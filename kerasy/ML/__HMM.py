#coding: utf-8
import os
import string
import numpy as np
from scipy.special import logsumexp
import warnings

from ..utils import Params
from ..utils import flush_progress_bar
from ..utils import handleKeyError
from ..utils import handle_random_state
from ..utils import has_not_attrs
from ..utils import normalize, log_normalize
from ..clib import c_hmm

DECODER_ALGORITHMS = ["map", "viterbi"]

def iter_from_variable_len_samples(X, lengths=None):
    """ yield starts and ends indexes of `X` per variable sample.
    Each sample should have the `n_features` features.
    np.sum(`lengths`) must be equal to `n_samples`
    =============================================================
    @params X       : Multiple connected samples. shape=(n_samples, n_features)
    @params lengths : int array. shape=(n_sequences)
    """
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}".format(n_samples, lengths))
        for i in range(len(lengths)):
            yield start[i], end[i]

class BaseHMM(Params):
    """ Hidden Markov Model.
    @params n_hstates    : (int) Number of hidden states.
    @params init         : (str) 'path/to/params.json', or 'random', or (dict)
    @params algorithm    : (str) 'viterbi', or 'map'. How to decode the input state sequences.
    @params up_params    : (str) Which parameters are updated in the training process.
    @params init_params  : (str) Which parameters are initialized prior to training.
    @params random_state : (int) random state for initialization.
    @attr initial   : Initial hidden states probabilities. shape=(n_hstates)
    @attr transit   : transition probabilities between hidden states. shape=(n_hstates, n_hstates)

    ~~~ Depending on the model Class

    - Discrete Multinomial HMM (ite)
        - @attr n_features : (int) Number of possible symbols emitted by the model.
        - @attr emission   : The emission probability per each hidden states. shape=(n_hstates, n_states)
        - @attr n_states   : (int) Number of possible symbols emitted by the model.

    - Gaussian Mixture HMM (itwmc)
        - @attr n_features : (int) Dimensionality of the Gaussian emissions.
        - @attr n_mix      : (int) Number of states in the GMM.
        - @attr weights    : Mixture weights for each state. shape=(n_hstates, n_mix)
        - @attr means      : Mean parameters for each mixture component in each state. shape=(n_hstates, n_mix)
        - @attr covars     : Covariance parameters for each mixture components in each state.
                             The shape depends on `covariance_type`
                                 - shape=(n_hstates, n_mix)                         if "spherical"
                                 - shape=(n_hstates, n_mix, n_features)             if "diag"
                                 - shape=(n_hstates, n_mix, n_features, n_features) if "full"
                                 - shape=(n_hstates, n_features, n_features)        if "tied"

    - Gaussian HMM (itmc)
        - @attr n_features : (int) Dimensionality of the Gaussian emissions.
        - @attr means      : Mean parameters for each state. shape=(n_hstates, n_features)
        - @attr covars     : Covariance parameters for each state.
                             The shape depends on `covariance_type`
                                 - shape=(n_hstates)                         if "spherical"
                                 - shape=(n_hstates, n_features)             if "diag"
                                 - shape=(n_hstates, n_features, n_features) if "full"
                                 - shape=(n_features, n_features)            if "tied"

    """
    def __init__(self, n_hstates=3, init="random", algorithm="viterbi",
                 up_params=string.ascii_letters, init_params=string.ascii_letters,
                 random_state=None):
        super().__init__()
        self.n_hstates = n_hstates
        self.init = init
        self.algorithm = algorithm
        self.up_params = up_params
        self.init_params = init_params
        self.seed = random_state
        self.disp_params = ["n_hstates", "init", "algorithm", "up_params", "init_params"]

    # @overrided
    def _check_params_validity():
        pass

    # @overrided
    def _init_params(self, X, init):
        """ Initialize the parameters """
        self.n_states = self._check_and_get_n_states(X)
        if init=="random":
            rnd = np.random.RandomState(self.seed)
            self.initial = rnd.rand(n_hstates)
            normalize(self.initial)
            self.transit = rnd.rand(self.n_hstates, self.n_hstates) + np.diag(np.ones(shape=self.n_hstates)*0.3)
            normalize(self.transit, axis=1)
            self.emission = rnd.rand(self.n_hstates, self.n_states)
            normalize(self.emission)
        elif os.path.exists(init):
            self.load_params(init)
            remain_attrs = has_not_attrs(self, "initial", "transit", "emission")
            if len(remain_attrs)>0:
                for attr in remain_attrs:
                    warnings.warn(f"`{}` is not described in {init}, please specify it in the file, or self.{}=SOMETHING to define it.")
        else:
            raise ValueError(f"the `init` parameter should be 'random', or 'path/to/json', but got {init})

    # @pverrided
    def _initialize_sufficient_statistics(self):
        pass

    # @overrided
    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                              posteriors, fwdlattice, bwdlattice):
        pass

    @abstractmethod
    def _generate_sample(self, hstate, random_state=None):
        """aa
        """
        raise NotImplementedError("This class is Abstract")

    @abstractmethod
    def _compute_log_likelihood(self, X):
        """ Computes per-component log probability under the model.
        @params X             : Multiple connected samples. shape=(n_samples, n_features)
        @params log_cond_prob : Log conditional probabilities. shape=(n_samples, n_hstates)
        """
        raise NotImplementedError("This class is Abstract")

    # @overrided
    def _mstep(self):
        """ Mstep of Hidden Markov Model.
        """
        raise NotImplementedError("This class is Abstract.")

    def fit(self, X, lengths=None, max_iter=10, tol=1e-4, verbose=1):
        """ Baum-Welch Algorithm
        @params X       : Multiple connected samples. shape=(n_samples, n_features)
        @params lengths : int array. shape=(n_sequences)
        """
        self._init_params(X, self.init)
        self._check_params_validity()
        for it in range(max_iter):
            current_log_prob = 0.
            for i,j in iter_from_X_lengths(X, lengths):
                # log conditional probability (start with ith)
                log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
                log_prob, log_alpha = self._Estep_log_forward(log_cond_prob_ij)
                current_log_prob += log_prob
                log_beta = self._Estep_log_backward(log_cond_prob_ij)
                posteriors = self._compute_posteriors(log_alpha, bwdlattice)
                # TODO: Update parameters.
                # self._accumulate_sufficient_statistics(stats, X[i:j], log_cond_prob_ij, posteriors, log_alpha, log_beta)
            flush_progress_bar(it, max_iter, metrics={"log probability": current_log_prob}, barname="Baum-Welch Algorithm")
            self._mstep()
            if "CONDITION":
                break
        return

    def score_samples(self, X, length=None):
        """ Compute the posterior probability for each hidden state.
        @params X       : Multiple connected samples. shape=(n_samples, n_features)
        @params lengths : int array. shape=(n_sequences)
        """
        self._check_params_validity()
        n_samples = X.shape
        log_prob = 0.
        posterior_prob = np.empty(shape=(n_samples, self.n_hstates))
        for i,j in iter_from_X_lengths(X, lengths):
            log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
            log_prob, log_alpha = self._Estep_log_forward(log_cond_prob_ij)
            log_prob += log_prob_ij

            log_beta = self._Estep_log_backward(log_cond_prob_ij)
            posterior_prob[i:j] = self._compute_posteriors(log_alpha, log_beta)
        return log_prob, posterior_prob

    def score(self, X, length=None):
        """ Compute the log likelihood under the model parameters.
        @params X       : Multiple connected samples. shape=(n_samples, n_features)
        @params lengths : int array. shape=(n_sequences)
        """
        self._check_params_validity()
        log_prob = 0.
        for i,j in iter_from_X_lengths(X, lengths):
            log_cond_prob_ij = self._compute_log_likelihood(X[i:j])
            log_prob_ij, _ = self._Estep_log_forward(log_cond_prob_ij)
            log_prob += log_prob_ij
        return log_prob

    def _decode_viterbi(self, X):
        """ Decode by viterbi algorithm. """
        log_cond_prob = self._compute_log_likelihood(X)
        # logprob, ml_hstates = self._viterbi(log_cond_prob)
        return self._viterbi(log_cond_prob)

    def _decode_map(self, X):
        """ Decode by MAP(Maximum A Posteriori) estimation """
        _, posterior_prob = self.score_samples(X)
        log_prob = np.max(posterior_prob, axis=1).sum()
        ml_hstates = np.argmax(posteriors, axis=1)
        return logprob, ml_hstates

    def decode(self, X, lengths=None, algorithm=None):
        """ Find the maximum likelihood hidden state sequences. """
        self.check()
        # The value of the preceding variable takes precedence.
        # Therefore, this code is same as the followings.
        # `algorithm = self.algorithm if algorithm is None else algoritm`
        algorithm = (algorithm or self.algorithm).lower()
        if algorithm not in ["viterbi", ]:
            handleKeyError(DECODER_ALGORITHMS, algorithm=algorithm)
        decode_func = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        n_samples = X.shape[0]
        log_prob = 0
        ml_hstates = np.empty(n_samples, dtype=int)
        for i, j in iter_from_variable_len_samples(X, lengths):
            log_prob_ij, ml_hstates_ij = decode_func(X[i:j])
            log_prob += log_prob_ij
            state_sequence[i:j] = ml_hstates_ij
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
        rnd = handle_random_state(self.seed if random_state is None else random_state)

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

        return np.atleast_2d(samples), np.asarray(h_state_sequences, dtype=int)

    def _viterbi(self, log_cond_prob):
        """ viterbi algorithm.
        @params log_cond_prob    : Log conditional probabilities. shape=(n_samples, n_hstates)
        @return log_prob         : Log scaled probability if the states were emitted from `ml_hstates`.
        @return ml_hstates       : Maximum likelihood hidden state sequences. shape=(n_samples)
        """
        ml_hstates, log_prob = c_hmm._log_forward(
            log_mask_zero(self.initial),
            log_mask_zero(self.transit),
            log_cond_prob
        )
        return logprob, ml_hstates

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

    def _compute_posteriors(self, log_alpha, log_beta):
        """ Compute log-scaled gamma(z_n) = p(z_n|X) = p(X|z_n)*p(z_n) / p(X) """
        # (PRML 13.33) gamma(z_n) = alpha(z_n)*beta(z_n) / p(X)
        log_gamma = log_alpha + log_beta
        log_normalize(log_gamma, axos=1)
        with np.errstate(under="ifnore"):
            return np.exp(log_gamma)

class MultinomialHMM(BaseHMM):
    """ Hidden Markov Model with multinomial (discrete) emissions """
    def __init__(self, n_hstates=3, init="random", algorithm="viterbi",
                 up_params="ite", init_params="ite", random_state=None):
        super().__init__(n_hstates=n_hstates, init=init, algorithm=algorithm,
                         up_params=up_params, init_params=init_params,
                         random_state=random_state)
        self.disp_params.extend([])

    def _check_and_get_n_states(self, X):
        """  """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if hasattr(self, "n_states"):
            if self.n_states-1 < X.max():
                raise ValueError(
                    "Largest symbol is {} but the model only emits "
                    "symbols up to {}"
                    .format(X.max(), self.n_states-1))
        n_states = X.max()+1
        return n_states
