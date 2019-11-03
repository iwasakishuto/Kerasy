# coding: utf-8
import numpy as np
import scipy.stats as stats

class MixedGaussian():
    def __init__(self, K=3, random_state=None):
        self.K = K
        self.N = None
        self.M = None
        self.mu = None
        self.S  = None
        self.pi = None
        self.epoch = 0
        self.seed=random_state

    def fit(self, data, memo_span=5):
        self.history=[]
        self.N, self.M = data.shape
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)

        #=== Initialization ===
        self.mu = np.random.RandomState(self.seed).uniform(low=mins, high=maxs, size=(self.K, self.M))
        self.S  = [1*np.eye(2) for k in range(self.K)] # Initialize with Diagonal matrix
        self.pi = np.ones(self.K)/self.K # Initialize with Uniform.

        pmus = np.copy(self.mu.ravel())
        while True:
            gamma = self.Estep(data)
            if self.epoch % memo_span ==0: self.memorize(data, gamma)
            self.Mstep(data, gamma)
            # Stop Condition.
            if np.linalg.norm(self.mu.ravel()-pmus)/(self.K*self.M) < 1e-3:
                self.memorize(data, gamma)
                break
            pmus = np.copy(self.mu.ravel())
            self.epoch+=1

    def memorize(self, data, gamma):
        self.history.append([
            self.epoch,
            np.argmax(gamma, axis=1),
            (np.copy(self.mu), np.copy(self.S), np.copy(self.pi)),
            self.loglikelihood(data)
        ])

    def predict(self, data):
        self.N, _ = data.shape
        gamma = self.Estep(data, normalized=False)
        return gamma

    def Estep(self, data, normalized=True):
        gamma = np.zeros(shape=(self.N, self.K))
        for k in range(self.K):
            gamma[:,k] = self.pi[k] * stats.multivariate_normal(self.mu[k], self.S[k]).pdf(data)
        if normalized: gamma /= np.sum(gamma, axis=1)[:,None]
        return gamma

    def Mstep(self, data, gamma):
        for k in range(self.K):
            Nk = np.sum(gamma[:, k])
            self.mu[k] = gamma[:,k].dot(data)/Nk
            self.S[k]  = (np.expand_dims(gamma[:,k], axis=1)*(data-self.mu[k])).T.dot(data-self.mu[k]) / Nk
            self.pi[k] = Nk/self.N

    def loglikelihood(self, data):
        return np.sum(np.log(np.sum([self.pi[k]*stats.multivariate_normal(self.mu[k], self.S[k]).pdf(data) for k in range(self.K)], axis=0)))
