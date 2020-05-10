#coding: utf-8
import os
import string
import numpy as np
from scipy.special import logsumexp, comb
import warnings

from ..utils import Params
from ..utils import ConvergenceMonitor
from ..utils import flush_progress_bar
from ..utils import handleKeyError
from ..utils import handleRandomState
from ..utils import has_not_attrs
from ..utils import normalize, log_normalize, log_mask_zero
from ..utils import log_multivariate_normal_density
from ..utils import decompress_based_on_covariance_type
from ..utils import compress_based_on_covariance_type_from_tied_shape
from ..utils import iter_from_variable_len_samples
from ..clib import c_hmm

from .EM import KMeans

DECODER_ALGORITHMS = ["viterbi", "map",]
DECODER_FUNC_NAMES = ["_decode_viterbi", "_decode_map"]
COVARIANCE_TYPES   = ["spherical", "diag", "full", "tied"]

class BaseHMM(Params):
    """ Hidden Markov Model.
    @params n_hstates    : (int) Number of hidden states.
    @params init         : (str) 'path/to/params.json', or 'random', or (dict)
    @params algorithm    : (str) 'viterbi', or 'map'. How to decode the input state sequences.
    @params up_params    : (str) Which parameters are updated in the training process.
    @params random_state : (int) random state for initialization.
    @attr initial   : Initial hidden states probabilities. shape=(n_hstates)
    @attr transit   : transition probabilities between hidden states. shape=(n_hstates, n_hstates)

    ~~~ Depending on the model Class

    * Discrete Multinomial HMM (ite)
        + @attr n_features : (int) Number of possible symbols emitted by the model.
        + @attr emission   : The emission probability per each hidden states. shape=(n_hstates, n_states)
        + @attr n_states   : (int) Number of possible symbols emitted by the model.

    ~~~~~~
    @attr covariance_type  : (str) The type of covariance parameters to use.
        | "spherical" | single variance value that applies to all features.     |
        | "diag"      | diagonal covariance matrix.                             |
        | "full"      | full (i.e. unrestricted) covariance matrix.             |
        | "tied"      | All hidden states use "the same" full covariance matrix.|
    ~~~~~~

    * Gaussian Mixture HMM (itwmc)
        + @attr n_features : (int) Dimensionality of the Gaussian emissions.
        + @attr n_mix      : (int) Number of states in the GMM.
        + @attr weights    : Mixture weights for each state. shape=(n_hstates, n_mix)
        + @attr means      : Mean parameters for each mixture component in each state. shape=(n_hstates, n_mix)
        + @attr covars     : Covariance parameters for each mixture components in each state.
                             The shape depends on `covariance_type`
                                 - shape=(n_hstates, n_mix)                         if "spherical"
                                 - shape=(n_hstates, n_mix, n_features)             if "diag"
                                 - shape=(n_hstates, n_mix, n_features, n_features) if "full"
                                 - shape=(n_hstates, n_features, n_features)        if "tied"

    * Gaussian HMM (itmc)
        + @attr n_features : (int) Dimensionality of the Gaussian emissions.
        + @attr means      : Mean parameters for each state. shape=(n_hstates, n_features)
        + @attr covars     : Covariance parameters for each state.
                             The shape depends on `covariance_type`
                                 - shape=(n_hstates)                         if "spherical"
                                 - shape=(n_hstates, n_features)             if "diag"
                                 - shape=(n_hstates, n_features, n_features) if "full"
                                 - shape=(n_features, n_features)            if "tied"

    ============================================================================

    When implementing a hidden Markov model with a new emission probability,
    you need to define the following methods:

    - Method which needs to be implemented.
        - _get_params_size(self)
        - _generate_sample(self, hstate, random_state=None)
        - _compute_log_likelihood(self, X)
        - _check_input_and_get_HOGEHOGE(self, X)
    - Method which needs to be overrided.
        - _check_params_validity(self)
        - _init_params(self, X, init)
        - _init_statistics(self)
        - _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta)
        - _Mstep(statistics):

    If you need a new instance variable `HOGEHOGE`, use the following method.

    ```
    def _check_input_and_get_HOGEHOGE():
        ...
        return HOGEHOGE

    def _init_params(self, X, init):
        self.HOGEHOGE = self._check_input_and_get_HOGEHOGE(X)
        super()._init_params(X, init)

        if isinstance(init, str) and init=="random":
            self.PIYO = self.rnd.rand(...
    ```
    """
    def __init__(self, n_hstates=3, init="random", algorithm="viterbi",
                 up_params=string.ascii_letters, random_state=None):
        super().__init__()
        self.n_hstates = n_hstates
        self.init = init
        self.algorithm = algorithm
        self.up_params = up_params
        self.rnd = handleRandomState(random_state)
        self.disp_params = ["n_hstates", "init", "algorithm", "up_params"]
        self.model_params = ["initial", "transit"]

    # @overrided
    def _check_params_validity(self):
        """ Validates model parameters.
        This method checks `self.initial` and `self.transit` only, and
        each sub class must override this method, as followings:
        ```
        def _check_params_validity():
            super()._check_params_validity()
            ...
        ```
        """
        self.initial = initial = np.asarray(self.initial)
        if initial.shape[0] != self.n_hstates:
            raise ValueError(f"self.initial must have length n_hstates, ({initial.shape[0]} != {self.n_hstates})")
        if not np.allclose(initial.sum(), 1.0):
            raise ValueError(f"self.initial must sum to 1.0 (but got {initial.sum():.5f})")

        self.transit = transit = np.asarray(self.transit)
        if transit.shape != (self.n_hstates, self.n_hstates):
            raise ValueError(f"self.transit must have shape (n_hstates, n_hstates)",
                             f"({transit.shape} != {self.n_hstates,self.n_hstates})")
        if not np.allclose(transit.sum(axis=1), 1.0):
            raise ValueError(f"self.transit must sum to 1.0 (but got {transit.sum(axis=1)})")

    def _init_params_by_json(self, json_path):
        """ Initialize parameter by json. """
        self.load_params(json_path)
        remain_attrs = has_not_attrs(self, self.model_params)
        if len(remain_attrs)>0:
            for attr in remain_attrs:
                warnings.warn(f"`{attr}` is not described in {json_path}, please specify it in the file",
                              f"or self.{attr}=SOMETHING to define it.")

    def _init_params_by_dict(self, params_dict):
        """ Initialize parameter by dictionaly. """
        self.__dict__.update(params_dict)
        self.format_params(list_params=["disp_params"])
        remain_attrs = has_not_attrs(self, self.model_params)
        if len(remain_attrs)>0:
            for attr in remain_attrs:
                warnings.warn(f"`{attr}` is not described in dictionaly, please specify it in the file",
                              f"or self.{attr}=SOMETHING to define it.")

    def _init_params_by_random(self):
        """ Initialize the parameters by random (only support for `initial` and `transit`). """
        self.initial = self.rnd.rand(self.n_hstates)
        normalize(self.initial)
        self.transit = self.rnd.rand(self.n_hstates, self.n_hstates) + np.diag(np.ones(shape=self.n_hstates)*0.3)
        normalize(self.transit, axis=1)

    # @overrided
    def _init_params(self, X, init):
        """
        This method initialize parameters before training. If you initialize
        parameters by random, it only handles with `initial`, and `transit`,
        so each sub class must override this method, as followings.
        ```
        def _init_params(self, X, init):
            self.hoge = self._check_and_get_hogehoge(X)
            super()._init_params(X, init)

            if isinstance(init, str) and init=="random":
                ...
        ```
        ================================================
        @params X    : Multiple connected samples. shape=(n_samples, n_features)
        @params init : (str) 'path/to/params.json', or 'random', or (dict)
        """
        params_scale_dict = self._get_params_size()
        n_trainable_params = sum([params_scale_dict[p] for p in self.up_params])
        if X.size < n_trainable_params:
            warnings.warn(f"Fitting a model with {n_trainable_params} trainable "
                          f"parameters with only {X.size} data. It will result in "
                          "a degenerate solution.")

        if isinstance(init, dict):
            self._init_params_by_dict(init)
        elif isinstance(init, str):
            if init=="random":
                self._init_params_by_random()
            else:
                self._init_params_by_json(init)
        else:
            raise ValueError(f"the `init` parameter should be 'random', or 'path/to/json', but got {init}")

    # @overrided
    def _init_statistics(self):
        """ Initialize the statistics storage.
        This method initialize the statistics storage. These information is used
        to update the model parameters.
        ```
        def _init_statistics():
            statistics = super()._init_statistics()
            ...
        ```
        """
        return {
            "n_sample": 0,
            "initial" : np.zeros(shape=(self.n_hstates)),
            "xi" : np.zeros(shape=(self.n_hstates, self.n_hstates)),
        }

    # @overrided
    def _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta):
        """ Update the statistics storage.
        @params statistics     : (dict) statistics storage.
        @params X              : Multiple connected samples. shape=(n_samples, n_features)
        @params log_cond_prob  : Log conditional probabilities. shape=(n_samples, n_hstates)
        @params posterior_prob : Posterior probabilities. shape=(n_samples, n_hstates)
        @params log_alpha      : Log scaled alpha. shape=(n_samples, n_hstates)
        @params log_beta       : Log scaled beta.  shape=(n_samples, n_hstates)
        """
        statistics['n_sample'] += 1
        if 'i' in self.up_params:
            # (PRML 13.18) pi_k = gamma(z_{ik}) / sum_j(gamma(z_{ij}))
            statistics['initial'] += posterior_prob[0]
        if 't' in self.up_params:
            if log_cond_prob.shape[0]>1:
                statistics['xi'] += self._compute_posteriors_xi_sum(log_alpha, log_beta, log_cond_prob)

    # @abstractmethod
    def _get_params_size(self):
        """
        This method is used to check whether there is enough data or not.
        Please refer to `_init_params` method for more details.
        @return (dict) model paramete size per parameter.
        """
        raise NotImplemented("`_get_params_size` method is not implemented.")

    # @abstractmethod
    def _generate_sample(self, hstate, random_state=None):
        """ Generate sample from given hidden state. """
        raise NotImplemented("`_generate_sampl` method is not implemented.")

    # @abstractmethod
    def _compute_log_likelihood(self, X):
        """ Computes log probability per hidden state under the model.
        @params X             : Multiple connected samples. shape=(n_samples, n_features)
        @params log_cond_prob : Log conditional probabilities. shape=(n_samples, n_hstates)
        """
        raise NotImplemented("`_compute_log_likelihood` method is not implemented.")

    # @overrided
    def _Mstep(self, statistics):
        """ Mstep of Hidden Markov Model. """

        if 's' in self.up_params:
            # (PRML 13.18) pi_k = gamma(z_{1k}) / sum_{j=1}^K gamma(z_{ij})
            self.initial = np.where(statistics['initial']==0, 0, statistics['initial'])
            normalize(self.initial)
        if 't' in self.up_params:
            # (PRML 13.19) A_jk = sum_{n=2}^N xi(z_{n-1,j},z_{nk}) / sum_{l=1}^K sum_{n=2}^N xi(z_{n-1,j},z_{nl})
            self.transit = np.where(statistics['xi']==0, 0, statistics['xi'])
            normalize(self.transit, axis=1)

    def fit(self, X, lengths=None, max_iter=10, tol=1e-4, verbose=1):
        """ Baum-Welch Algorithm
        @params X       : Multiple connected samples. shape=(n_samples, n_features)
        @params lengths : int array. shape=(n_sequences)
        """
        self._init_params(X, self.init)
        self._check_params_validity()
        # self.monitor = ConvergenceMonitor(tol, max_iter=max_iter, verbose=verbose)

        for it in range(max_iter):
            statistics = self._init_statistics()
            current_log_prob = 0.
            for i,j in iter_from_variable_len_samples(X, lengths):
                log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
                log_prob, log_alpha = self._Estep_log_forward(log_cond_prob_ij)
                current_log_prob += log_prob
                log_beta = self._Estep_log_backward(log_cond_prob_ij)
                posterior_prob = self._compute_posteriors_gamma(log_alpha, log_beta)
                self._update_statistics(statistics, X[i:j], log_cond_prob_ij, posterior_prob, log_alpha, log_beta)

            self._Mstep(statistics)
            self.statistics = statistics
            flush_progress_bar(it, max_iter, metrics={"log probability": current_log_prob}, barname=f"{self.__class__.__name__} (Baum-Welch)", verbose=verbose)
            if it>0 and prev_log_prob - current_log_prob < tol:
                break
            prev_log_prob = current_log_prob

            # self.monitor.report(current_log_prob,metrics={"log probability": current_log_prob}, barname="Baum-Welch Algorithm")
            # if self.monitor.converged:
            #     break

        if (self.transit.sum(axis=1) == 0).any():
            warnings.warn("Some rows of transit have zero sum because no transition from the hidden state was ever observed.")

    def score_samples(self, X, lengths=None):
        """ Compute the posterior probability for each hidden state.
        @params X       : Multiple connected samples. shape=(n_samples, n_features)
        @params lengths : int array. shape=(n_sequences)
        """
        self._check_params_validity()
        n_samples = X.shape[0]
        log_prob = 0.
        posterior_prob = np.empty(shape=(n_samples, self.n_hstates))
        for i,j in iter_from_variable_len_samples(X, lengths):
            log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
            log_prob_ij, log_alpha = self._Estep_log_forward(log_cond_prob_ij)
            log_prob += log_prob_ij

            log_beta = self._Estep_log_backward(log_cond_prob_ij)
            posterior_prob[i:j] = self._compute_posteriors_gamma(log_alpha, log_beta)
        return log_prob, posterior_prob

    def score(self, X, lengths=None):
        """ Compute the log likelihood under the model parameters.
        @params X       : Multiple connected samples. shape=(n_samples, n_features)
        @params lengths : int array. shape=(n_sequences)
        """
        self._check_params_validity()
        log_prob = 0.
        for i,j in iter_from_variable_len_samples(X, lengths):
            log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
            log_prob_ij, _ = self._Estep_log_forward(log_cond_prob_ij)
            log_prob += log_prob_ij
        return log_prob

    def _decode_viterbi(self, X):
        """
        Decode by viterbi algorithm.
        The viterbi algorithm computes the probability that an HMM generates an
        observation sequence in the "best path".
        """
        log_cond_prob = self._compute_log_likelihood(X)
        # logprob, ml_hstates = self._viterbi(log_cond_prob)
        return self._viterbi(log_cond_prob)

    def _decode_map(self, X):
        """
        Decode by MAP(Maximum A Posteriori) estimation.
        * If we have no prior information,
          the MAP estimate becomes identical to the ML estimate.
        The forward algorithm computes the probability that an HMM generates an
        observation sequence by summing up the probabilities of "all possible" paths.
        """
        _, posterior_prob = self.score_samples(X)
        log_prob = np.max(posterior_prob, axis=1).sum()
        ml_hstates = np.argmax(posterior_prob, axis=1)
        return log_prob, ml_hstates

    def decode(self, X, lengths=None, algorithm=None):
        """ Find the maximum likelihood hidden state sequences. """
        self._check_params_validity()
        # The value of the preceding variable takes precedence.
        # Therefore, this code is same as the followings.
        # `algorithm = self.algorithm if algorithm is None else algoritm`
        algorithm = (algorithm or self.algorithm).lower()
        if algorithm not in DECODER_ALGORITHMS:
            handleKeyError(DECODER_ALGORITHMS, algorithm=algorithm)

        decode_func = dict(zip(
            DECODER_ALGORITHMS,
            [self.__getattribute__(func_name) for func_name in DECODER_FUNC_NAMES]
        ))[algorithm]

        n_samples = X.shape[0]
        log_prob = 0
        ml_hstates = np.empty(n_samples, dtype=int)
        print(f"Algorithm: {algorithm}")
        for i, j in iter_from_variable_len_samples(X, lengths):
            log_prob_ij, ml_hstates_ij = decode_func(X[i:j])
            log_prob += log_prob_ij
            ml_hstates[i:j] = ml_hstates_ij
        return log_prob, ml_hstates

    def predict(self, X, lengths=None, algorithm=None):
        """ Find Maximum likelihood hidden state sequences. """
        _, ml_hstates = self.decode(X, lengths, algorithm=algorithm)
        return ml_hstates

    def predict_proba(self, X, lengths=None):
        """ Compute the posterior probability for each hidden state. """
        _, posterior_prob = self.score_samples(X, lengths)
        return posterior_prob

    def sample(self, n_samples=1, random_state=None, verbose=1):
        """Generate random samples from the model.
        @params n_samples    : (int) Number of samples to generate.
        @params random_state : (int) seed.
        """
        self._check_params_validity()
        rnd = handleRandomState(random_state) if random_state is not None else self.rnd

        transit = self.transit
        hidden_states = np.arange(self.n_hstates)

        # Initialization.
        h_state_sequences = np.empty(shape=n_samples, dtype=int)
        h_state_sequences[0] = h_state = self.initial.argmax()
        samples = [self._generate_sample(h_state, random_state=rnd)]
        flush_progress_bar(0, n_samples, metrics={"hidden state": h_state})
        for n in range(1, n_samples):
            h_state_sequences[n] = h_state = rnd.choice(a=hidden_states, p=transit[h_state])
            samples.append(self._generate_sample(h_state, random_state=rnd))
            flush_progress_bar(n, n_samples, metrics={"hidden state": h_state})

        self.rnd = rnd
        return np.atleast_2d(samples), np.asarray(h_state_sequences, dtype=int)

    def _viterbi(self, log_cond_prob):
        """ viterbi algorithm.
        @params log_cond_prob    : Log conditional probabilities. shape=(n_samples, n_hstates)
        @return log_prob         : Log scaled probability if the states were emitted from `ml_hstates`.
        @return ml_hstates       : Maximum likelihood hidden state sequences. shape=(n_samples)
        """
        ml_hstates, log_prob = c_hmm._viterbi(
            log_mask_zero(self.initial),
            log_mask_zero(self.transit),
            log_cond_prob
        )
        return log_prob, ml_hstates

    def _Estep_log_forward(self, log_cond_prob):
        """ Log-scaled forward algorithm.
        @params log_cond_prob : Log conditional probabilities. p(x_n,z_n) shape=(n_samples, n_hstates)
        @return log_prob      : Log scaled probability. p(X)
        @return log_alpha     : Log scaled alpha. shape=(n_samples, n_hstates)
        """
        log_alpha = c_hmm._log_forward(
            log_mask_zero(self.initial),
            log_mask_zero(self.transit),
            log_cond_prob
        )
        with np.errstate(under="ignore"):
            # (PRML 13.42) p(X) = sum_{z_N}(alpha(z_N))
            log_prob = logsumexp(log_alpha[-1])
            return log_prob, log_alpha

    def _Estep_log_backward(self, log_cond_prob):
        """ Log-scaled backward algorithm.
        @params log_cond_prob : Log conditional probabilities. p(x_n,z_n) shape=(n_samples, n_hstates)
        @return log_beta      : Log scaled beta. shape=(n_samples, n_hstates)
        """
        log_beta = c_hmm._log_backward(
            log_mask_zero(self.initial),
            log_mask_zero(self.transit),
            log_cond_prob
        )
        return log_beta

    def _compute_posteriors_gamma(self, log_alpha, log_beta):
        """ Compute gamma using the results of forward-backward algorithm.
        * All parameters have the shape (n_samples, n_hstates).
        @params log_alpha : alpha(z_n) = p(x_1,...,x_n,z_n)
        @params log_beta  :  beta(z_n) = p(x_{n+1},...,x_N|z_n)
        @return gamma     : gamma(z_n) = p(z_n|X,theta^{old})
        """
        # (PRML 13.33) gamma(z_n) = alpha(z_n)*beta(z_n) / p(X)
        log_gamma = log_alpha + log_beta
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _compute_posteriors_xi_sum(self, log_alpha, log_beta, log_cond_prob):
        """ Compute sum_{n=2}^N xi(z_{n-1},z_n) using the results of forward-backward algorithm.
        @params log_cond_prob : p(x_n,z_n). shape=(n_samples, n_hstates)
        @params log_alpha     : alpha(z_n) = p(x_1,...,x_n,z_n). shape=(n_samples, n_hstates).
        @params log_beta      :  beta(z_n) = p(x_{n+1},...,x_N|z_n). shape=(n_samples, n_hstates).
        @return xi_sum        : sum_{n=2}^N xi(z_{n-1},z_n) = sum_{n=2}^N p(z_{n-1},z_n|X,theta^{old}). shape=(n_hstates, n_hstates)
        """
        log_xi_sum = c_hmm._compute_log_xi_sum(
            log_alpha, log_beta, log_mask_zero(self.transit), log_cond_prob
        )
        with np.errstate(under="ignore"):
            return np.exp(log_xi_sum)

    def heatmap_params(self, **plot_kwargs):
        """ Not necessary !! """
        try:
            import os
            import json
            import seaborn as sns
            import matplotlib.pyplot as plt
            from ..utils import UTILS_DIR_PATH
            abs_path = os.path.join(UTILS_DIR_PATH, "default_params", "sns_heatmap_kwargs.json")
            with open(abs_path, 'r') as f:
                DEFAULT_KWARGS = json.load(f)
        except:
            raise ImportError("You Need to install `seaborn`")

        n_hstates = self.n_hstates
        n_rows = n_hstates + 1
        mk_ticklabels = lambda n:["$z_{" + f"{n},{k}" + "}$" for k in range(n_hstates)]

        DEFAULT_KWARGS["center"] = 1/n_hstates
        DEFAULT_KWARGS.update(plot_kwargs)

        fig = plt.figure(figsize=(2*n_hstates, 2*n_rows))
        fig.suptitle('Initial hidden state probs\n&\ntransit probs')
        axi = plt.subplot2grid((n_rows, n_hstates), (0, 0), colspan=n_hstates)
        axt = plt.subplot2grid((n_rows, n_hstates), (1, 0), colspan=n_hstates, rowspan=n_hstates)

        sns.heatmap(
            self.initial.reshape(1,-1), ax=axi, **DEFAULT_KWARGS,
            yticklabels=False, xticklabels=mk_ticklabels(0)
        )
        sns.heatmap(
            self.transit, ax=axt, **DEFAULT_KWARGS,
            yticklabels=mk_ticklabels("n-1"), xticklabels=mk_ticklabels("n")
        )
        plt.show()

class MultinomialHMM(BaseHMM):
    """ Hidden Markov Model with multinomial (discrete) emissions """
    def __init__(self, n_hstates=3, init="random", algorithm="viterbi",
                 up_params="ite", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)
        self.disp_params.extend(["n_states"])
        self.model_params.extend(["emission"])

    def _get_params_size(self):
        nh = self.n_hstates
        ns = self.n_states
        return {
            "i": nh - 1,
            "t": nh * (nh - 1),
            "e": nh * (ns - 1),
        }

    def _generate_sample(self, hstate, random_state=None):
        rnd = handleRandomState(random_state)
        return rnd.choice(a=np.arange(self.n_states), p=self.emission[hstate, :])

    def _compute_log_likelihood(self, X):
        return log_mask_zero(self.emission)[:, np.concatenate(X)].T

    def _check_params_validity(self):
        super()._check_params_validity()
        self.emission = emission = np.asarray(self.emission)
        if emission.shape != (self.n_hstates, self.n_states):
            raise ValueError(f"self.emission must have shape (n_hstates, n_states)",
                             f"({emission.shape} != {n_hstates, n_states})")
        if not np.allclose(emission.sum(axis=1), 1.0):
            raise ValueError(f"self.emission must sum to 1.0 (but got {emission.sum(axis=1)})")

    def _check_input_and_get_nstates(self, X):
        """ Check if ``X`` is a sample from a Multinomial distribution and get `n_states`. """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        return X.max() + 1

    def _init_params(self, X, init):
        self.n_states = self._check_input_and_get_nstates(X)
        super()._init_params(X, init)

        if isinstance(init, str) and init=="random":
            self.emission = self.rnd.rand(self.n_hstates, self.n_states)
            normalize(self.emission, axis=1)

    def _init_statistics(self):
        statistics = super()._init_statistics()
        statistics['observation'] = np.zeros((self.n_hstates, self.n_states))
        return statistics

    def _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta):
        super()._update_statistics(statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta)

        if 'e' in self.up_params:
            for n,symbol in enumerate(np.concatenate(X)):
                # posterior_prob has the shape (n_samples, n_hstates), and usually called "gamma".
                # (PRML 13.23) sum_n(gamma(z_{nk})*x_{ni}) / sum_n(gamma(z_{nk}))
                # statistics['observation'] has the shape (n_hstates, n_states)
                # and means that sum_n( gamma(z_{nk})*x_{ni}) )
                statistics['observation'][:, symbol] += posterior_prob[n]

    def _Mstep(self, statistics):
        super()._Mstep(statistics)

        if 'e' in self.up_params:
            self.emission = statistics['observation'] / statistics['observation'].sum(axis=1)[:, np.newaxis]

    def MCK(self, X, lengths=None):
        self._check_params_validity()

        MCKs = np.empty(shape=(0,self.n_states, self.n_hstates))
        for i, j in iter_from_variable_len_samples(X, lengths):
            log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
            log_prob, log_alpha = self._Estep_log_forward(log_cond_prob_ij)
            log_beta = self._Estep_log_backward(log_cond_prob_ij)
            posterior_prob = self._compute_posteriors_gamma(log_alpha, log_beta)
            MCK = 1/(j-i) * np.eye(self.n_states)[np.concatenate(X[i:j])].T.dot(posterior_prob) # (M,N)@(N,K)=(M,K)
            MCKs = np.r_[MCKs, MCK[np.newaxis, :, :]]
        return MCKs

class BernoulliHMM(MultinomialHMM):
    def __init__(self, n_hstates=3, init="random", algorithm="viterbi",
                 up_params="ite", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)

    def _check_input_and_get_nstates(self, X):
        """ Check if ``X`` is a sample from a Bernoulli distribution and get `n_states`. """
        if not np.issubdtype(X.dtype, np.integer) or X.min() < 0 or X.max() > 1:
            raise ValueError("Symbols should be integers 0 or 1.")
        return 2

class BinomialHMM(BaseHMM):
    """ Hidden Markov Model with binomial (discrete) emissions
    @input     X : shape=(n_samples, 2)
        - X[:, 0] (yes) The number of success/yes/true/one experiments.
        - X[:, 1] (no)  The number of failure/no/false/zero experiments.
    """
    def __init__(self, n_hstates=3, init="random", algorithm="viterbi",
                 up_params="itθ", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)
        self.model_params.extend(["thetas"])

    def _get_params_size(self):
        nh = self.n_hstates
        return {
            "i": nh - 1,
            "t": nh * (nh - 1),
            "θ": nh,
        }

    def _generate_sample(self, hstate, random_state=None):
        raise NotImplementedError("Binomial distribution needs a population of size N. B(N,theta)")

    def _compute_log_likelihood(self, X):
        """ Return `log_cond_prob`[n][k] = p(x_n|z_{nk}) """
        num_comb = comb(X.sum(1), X[:,0])
        log_cond_prob = log_mask_zero([num_comb * np.power(theta, X[:,0]) * np.power(1-theta, X[:,1]) for theta in self.thetas]).T
        return np.ascontiguousarray(log_cond_prob)

    def _check_params_validity(self):
        super()._check_params_validity()

        self.thetas = thetas = np.asarray(self.thetas)
        if len(thetas) != self.n_hstates:
            raise ValueError(f"self.thetas must have length n_hstates, ({len(thetas)} != {self.n_hstates})")

        if not np.all([0<=theta<=1 for theta in thetas]):
            raise ValueError("self.thetas must be greater than 0 and less than 1.")

    def _check_input(self, X):
        """
        Check the input data ``X`` is valid sample or not.
        @params X :  shape=(n_samples, 2)
        ex.) Detection of methylated regions.
            - X[:, 0] The number of reads of 'unconverted' C on bisulfite-seq at each CpC site.
            - X[:, 1] The number of reads of 'converted' C on bisulfite-seq at each CpC site.
        """
        if not X.shape[1] == 2:
            raise ValueError("Input data must have the shape (n_samples, 2).")
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")

    def _init_params(self, X, init):
        self._check_input(X)
        super()._init_params(X, init)

        if isinstance(init, str) and init=="random":
            self.thetas = self.rnd.rand(self.n_hstates)

    def _init_statistics(self):
        statistics = super()._init_statistics()
        statistics['observation'] = np.zeros(shape=(2,self.n_hstates))
        return statistics

    def _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta):
        super()._update_statistics(statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta)

        if 'θ' in self.up_params:
            statistics['observation'] += X.T.dot(posterior_prob)

    def _Mstep(self, statistics):
        super()._Mstep(statistics)

        if 'θ' in self.up_params:
            A,B = statistics['observation']
            self.thetas = A/(A+B)

class GaussianHMM(BaseHMM):
    def __init__(self, n_hstates=3, covariance_type="diag", min_covariances=1e-3,
                 means_prior=0, means_weight=0, covariances_prior=1e-2, covariances_weight=1,
                 init="random", algorithm="viterbi",
                 up_params="itmc", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)
        self.disp_params.extend(["n_features"])
        self.model_params.extend(["means", "covariances"])
        self.covariance_type = covariance_type
        self.min_covariances = min_covariances
        self.means_prior  = means_prior
        self.means_weight = means_weight
        self.covariances_prior  = covariances_prior
        self.covariances_weight = covariances_weight

    @property
    def covariances(self):
        return decompress_based_on_covariance_type(
            self._covariances, self.covariance_type, self.n_hstates, self.n_features
        )

    def _get_params_size(self):
        nh = self.n_hstates
        nf = self.n_features
        return {
            "i": nh - 1,
            "t": nh * (nh - 1),
            "m": nh * nf,
            "c": {
                "spherical": nh,
                "diag": nh * nf,
                "full": nh * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _generate_sample(self, hstate, random_state=None):
        rnd = handleRandomState(random_state)
        return rnd.multivariate_normal(
            self.means[hstate], self._covariances[hstate]
        )

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(X, self.means, self._covariances, self.covariance_type)

    def _check_params_validity(self):
        super()._check_params_validity()

        self.means = means = np.asarray(self.means)
        if means.shape != (self.n_hstates, self.n_features):
            raise ValueError(f"self.means must have length n_hstates, ({initial.shape[0]} != {self.n_hstates})")

        self._covariances = covariance = np.asarray(self._covariances)
        expected_shape = {
            "spherical" : (self.n_hstates,),
            "diag"      : (self.n_hstates, self.n_features),
            "full"      : (self.n_hstates, self.n_features, self.n_features),
            "tied"      : (self.n_features, self.n_features),
        }[self.covariance_type]
        if covariance.shape != expected_shape:
            raise ValueError(f"if covariance type is {self.covariance_type}, self._covariances must have shape "\
                             f"{expected_shape}, but got {covariance.shape}")

    def _check_input_and_get_nfeatures(self, X):
        _, n_features = X.shape
        if hasattr(self, "n_features") and self.n_features != n_features:
            raise ValueError(f"Unexpected number of dimensions, got {n_features} but expected {self.n_features}")
        return n_features

    def _init_params(self, X, init):
        self.n_features = self._check_input_and_get_nfeatures(X)
        super()._init_params(X, init)

        if isinstance(init, str) and init=="random":
            kmeans = KMeans(n_clusters=self.n_hstates, init="k++", random_state=self.rnd)
            kmeans.fit(X, verbose=-1)
            self.means = kmeans.centroids

            # shape=(n_features, n_features)
            cv = np.cov(X.T) + self.min_covariances * np.eye(self.n_features)
            self._covariances = compress_based_on_covariance_type_from_tied_shape(cv, covariance_type=self.covariance_type, n_gaussian=self.n_hstates)

    def _init_statistics(self):
        statistics = super()._init_statistics()

        statistics['posterior'] = np.zeros(self.n_hstates)
        statistics['observation'] = np.zeros(shape=(self.n_hstates, self.n_features))
        statistics['observation_squared'] = np.zeros(shape=(self.n_hstates, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            statistics['posterior @ observation @ observation.T'] = np.zeros(shape=(self.n_hstates, self.n_features, self.n_features))
        return statistics

    def _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta):
        super()._update_statistics(statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta)

        if sum(self.up_params.count(p) for p in ['m', 'c']) > 1:
            statistics['posterior'] += posterior_prob.sum(axis=0)
            statistics['observation'] += np.dot(posterior_prob.T, X)

        if 'c' in self.up_params:
            if self.covariance_type in ('spherical', 'diag'):
                statistics['observation_squared'] += np.dot(posterior_prob.T, X**2)
            elif self.covariance_type in ('tied', 'full'):
                statistics['posterior @ observation @ observation.T'] += np.einsum('ij,ik,il->jkl', posterior_prob, X, X)

    def _Mstep(self, statistics):
        super()._Mstep(statistics)

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = statistics['posterior'][:, np.newaxis]
        if 'm' in self.up_params:
            self.means = (means_weight*means_prior+statistics['observation']) / (means_weight+denom)

        if 'c' in self.up_params:
            covars_prior = self.covariances_prior
            covars_weight = self.covariances_weight
            meandiff = self.means - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2 + statistics['observation_squared'] - 2*self.means*statistics['observation'] + self.means**2*denom)
                cv_den = max(covars_weight-1, 0) + denom
                covariances = (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covariances = covariances.mean(1)
                else:
                    self._covariances = covariances
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty(shape=(self.n_hstates, self.n_features, self.n_features))
                for c in range(self.n_hstates):
                    obsmean = np.outer(statistics['observation'][c], self.means[c])
                    cv_num[c] = means_weight*np.outer(meandiff[c], meandiff[c]) \
                              + statistics['posterior @ observation @ observation.T'][c] - obsmean \
                              - obsmean.T + np.outer(self.means[c], self.means[c]) * statistics['posterior'][c]
                cvweight = max(covars_weight-self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covariances = (covars_prior + cv_num.sum(axis=0)) / (cvweight + statistics['posterior'].sum())
                elif self.covariance_type == 'full':
                    self._covariances = (covars_prior + cv_num) / (cvweight + statistics['posterior'][:, None, None])

class GaussianMixtureHMM(BaseHMM):
    """
    There is no joint conjugate prior density. We can assume different components
    of the HMM to be mutually independent, so that the optimization can different
    components of the HMM to be mutually independent, so that the optimization can
    be split into different subproblems involving only a single component of the
    parameter set.
    """
    def __init__(self, n_hstates=3, n_mix=1, covariance_type="diag", min_covariances=1e-3,
                 means_prior=0, means_weight=0, covariances_prior=1e-2, covariances_weight=1,
                 init="random", algorithm="viterbi",
                 up_params="itmcw", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)
        self.disp_params.extend(["n_features", "n_mix"])
        self.model_params.extend(["means", "covariances", "weights"])

class MSSHMM(BaseHMM):
    """ HMM for Maximum Segment Sum.
    ex.) Detection of methylated regions.
    @input     X : shape=(n_samples, 2)
        - X[:, 0] (m) The number of reads of 'unconverted' C on bisulfite-seq at each CpC site.
        - X[:, 1] (u) The number of reads of 'converted' C on bisulfite-seq at each CpC site.
    @param theta : (float) This parameter controling
        "How much methylated C is likely to apper in the Hypermethylated region."
        p(m_n,u_n | total_n, Hypermethylated) = t_nCm_n * theta^{m_n} * (1-theta)^{u_n}
        p(m_n,u_n | total_n, Hypomethylated)  = t_nCm_n * (1-theta)^{m_n} * theta^{u_n}
    """
    def __init__(self, n_hstates=2, init="random", algorithm="viterbi",
                 up_params="itθ", random_state=None):
        if n_hstates != 2:
            raise ValueError(
                "This Model is a special case of the `BinomialHMM`.\
                There are only 2 hidden states (A,B), and theta_A = 1-theta_B, \
                so if you want to increase the hidden state, please define it yourself, \
                pr use BinomialHMM`."
            )
        super().__init__(n_hstates=2, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)
        self.disp_params.extend(["theta"])
        self.model_params.extend(["theta"])

    def _get_params_size(self):
        nh = self.n_hstates
        return {
            "i": nh - 1,
            "t": 1,
            "θ": 1,
        }

    def _generate_sample(self, hstate, random_state=None):
        raise NotImplementedError("Binomial distribution needs a population of size N. B(N,theta)")

    def _compute_log_likelihood(self, X):
        return log_mask_zero(comb(X.sum(1), X[:,0])[:,np.newaxis] * np.c_[
            np.prod(np.power(np.asarray([self.theta, 1-self.theta])[np.newaxis,:], X), axis=1),
            np.prod(np.power(np.asarray([1-self.theta, self.theta])[np.newaxis,:], X), axis=1),
        ])

    def _check_params_validity(self):
        super()._check_params_validity()
        if not isinstance(self.theta, float):
            raise ValueError("self.theta must be a float value.")

        if not 0<=self.theta<=1:
            raise ValueError("self.theta must be greater than 0 and less than 1.")

    def _check_input(self, X):
        """
        Check the input data ``X`` is valid sample or not.
        @params X :  shape=(n_samples, 2)
        ex.) Detection of methylated regions.
            - X[:, 0] The number of reads of 'unconverted' C on bisulfite-seq at each CpC site.
            - X[:, 1] The number of reads of 'converted' C on bisulfite-seq at each CpC site.
        """
        if not X.shape[1] == 2:
            raise ValueError("Input data must have the shape (n_samples, 2).")
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")

    def _init_params(self, X, init):
        self._check_input(X)
        super()._init_params(X, init)

        if isinstance(init, str) and init=="random":
            self.theta = self.rnd.rand()

    def _init_statistics(self):
        statistics = super()._init_statistics()
        statistics['observation'] = np.zeros(shape=(2))
        return statistics

    def _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta):
        super()._update_statistics(statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta)

        if 'θ' in self.up_params:
            statistics['observation'][0] += np.sum(X*posterior_prob)
            statistics['observation'][1] += np.sum(X[:,::-1]*posterior_prob)

    def _Mstep(self, statistics):
        super()._Mstep(statistics)

        if 'θ' in self.up_params:
            A,B = statistics['observation']
            self.theta = A/(A+B)

"""
If you want to add your original HMM, please follow the format below.
~~~~~~~
class KerasyHMM(BaseHMM):
    def __init__(self, n_hstates=3, kerasy="kerasy", init="random",
                 algorithm="viterbi", up_params="kerasy", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, random_state=random_state)
        self.disp_params.extend([])
        self.model_params.extend(["kerasy"])
        self.kerasy = kerasy

    def _get_params_size(self):
    def _generate_sample(self, hstate, random_state=None):
    def _compute_log_likelihood(self, X):
    def _check_params_validity(self):
    def _check_input_and_get_kerasy_params(self, X):
        return kerasy_params
    def _init_params(self, X, init):
        self.kerasy_params = self._check_input_and_get_kerasy_params(X)
        super()._init_params(X, init)

        if isinstance(init, str) and init=="random":
            self.kerasy_model_params = self.rnd.rand(...
    def _init_statistics(self):
    def _update_statistics(self, statistics, X, log_cond_prob, posterior_prob, log_alpha, log_beta):
    def _Mstep(self, statistics):
"""
