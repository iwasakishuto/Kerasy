# coding: utf-8
import numpy as np
import scipy.stats as stats

from ..utils import flush_progress_bar

class EMmodel():
    def __init__(self, K=3, random_state=None):
        self.K=K
        self.seed=random_state
        self.history=[]
    def train_initialize(self, X):
        """
        @param X: shape=(N,D)
        """
        self.history=[]
        N,D  = X.shape
        Xmin = np.min(X, axis=0); Xmax = np.max(X, axis=0) #shape=(D,)
        self.mu = np.random.RandomState(self.seed).uniform(low=Xmin, high=Xmax, size=(self.K, D))
    def fit(self):
        raise NotImplementedError()
    def predict(self):
        raise NotImplementedError()
    def Estep(self):
        raise NotImplementedError()
    def Mstep(self):
        raise NotImplementedError()

class KMeans(EMmodel):
    def __init__(self, K=3, random_state=None):
        super().__init__(K=K, random_state=random_state)
        self.mu=None

    def fit(self, X, max_iter=100, memorize=False):
        """
        @param X: shape=(N,D)
        """
        # Initialization.
        self.train_initialize(X)
        # EM algorithm.
        for it in range(max_iter):
            idx = self.Estep(X)
            if memorize: self.memorize_param(idx)
            self.Mstep(X, idx)
            J = np.sum([np.sum(np.square(X[idx==k]-self.mu[k])) for k in range(self.K)])
            flush_progress_bar(it, max_iter, metrics=f"distortion measure: {J:.3f}")
            if it>0 and not np.any(pidx!=idx): break
            pidx=idx
        if memorize: self.memorize_param(idx)
        print()

    def memorize_param(self, idx):
        self.history.append([
            idx,
            np.copy(self.mu),
        ])

    def predict(self, X):
        """ Same with Estep """
        idx = self.Estep(X)
        return idx

    def Estep(self, X):
        """ X.shape=(N,D) """
        idx = np.asarray([np.argmin([np.sum(np.square(x-mu)) for mu in  self.mu]) for x in X], dtype=int)
        return idx

    def Mstep(self, X, idx):
        """ Assign a new average(mu) """
        for k in np.unique(idx):
            self.mu[k] = np.mean(X[idx==k], axis=0)

class MixedGaussian(EMmodel):
    def __init__(self, K=3, random_state=None):
        super().__init__(K=K, random_state=random_state)
        self.mu=None
        self.S=None
        self.pi=None

    def fit(self, X, max_iter=100, memorize=False, tol=1e-5):
        # Initialization.
        self.train_initialize(X) # Initialize the mean value `self.mu` within data space.
        self.S  = [1*np.eye(2) for k in range(self.K)] # Initialize with Diagonal matrix
        self.pi = np.ones(self.K)/self.K # Initialize with Uniform.
        # EM algorithm.
        for it in range(max_iter):
            gamma = self.Estep(X)
            if memorize: self.memorize_param(gamma)
            self.Mstep(X, gamma)
            ll = self.loglikelihood(X)
            flush_progress_bar(it, max_iter, metrics=f"Log Likelihood: {ll:.3f}")
            mus = np.copy(self.mu.ravel())
            if it>0 and np.mean(np.linalg.norm(mus-pmus)) < tol: break
            pmus = mus
        if memorize: self.memorize_param(gamma)
        print()

    def memorize_param(self, gamma):
        self.history.append([
            np.argmax(gamma, axis=1),
            (np.copy(self.mu), np.copy(self.S), np.copy(self.pi))
        ])

    def predict(self, X):
        gamma = self.Estep(X, normalized=False)
        return gamma

    def Estep(self, X, normalized=True):
        N,_ = X.shape
        gamma = np.zeros(shape=(N, self.K))
        for k in range(self.K):
            gamma[:,k] = self.pi[k] * stats.multivariate_normal(self.mu[k], self.S[k]).pdf(X)
        if normalized: gamma /= np.sum(gamma, axis=1)[:,None]
        return gamma

    def Mstep(self, X, gamma):
        N,_ = X.shape
        for k in range(self.K):
            Nk = np.sum(gamma[:, k])
            self.mu[k] = gamma[:,k].dot(X)/Nk
            self.S[k]  = (np.expand_dims(gamma[:,k], axis=1)*(X-self.mu[k])).T.dot(X-self.mu[k]) / Nk
            self.pi[k] = Nk/N

    def loglikelihood(self, X):
        return np.sum(np.log(np.sum([self.pi[k]*stats.multivariate_normal(self.mu[k], self.S[k]).pdf(X) for k in range(self.K)], axis=0)))
