# coding: utf-8
import numpy as np

class KMeans():
    def __init__(self, K=3, random_state=None):
        self.K=K
        self.N=None
        self.M=None
        self.mu=None
        self.seed=random_state

    def fit(self, data, exact=False, threshold=1e-4):
        if exact: threshold=1
        self.N, self.M = data.shape
        #=== Initialization ===
        np.random.seed(self.seed)
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        self.mu = np.random.uniform(low=mins, high=maxs, size=(self.K, self.M))

        idx = self.Estep(data)
        pll = self._loglikelihood(data, idx)
        #=== EM algorithm ===
        while True:
            idx = self.Estep(data)
            self.Mstep(data, idx)
            ll = self._loglikelihood(data, idx)
            if 1-ll/pll <= threshold: break
            pll = ll

    def predict(self, data):
        idx = self.Estep(data)
        return (idx, self.mu)

    def Estep(self, data):
        idx = list(map(self._nearlest, data))
        return np.array(idx)

    def Mstep(self, data, idx):
        for k in np.unique(idx):
            self.mu[k] = np.mean(data[idx==k], axis=0)

    def _nearlest(self, x):
        return np.argmin(np.sum((self.mu-x)**2, axis=1))

    def _loglikelihood(self, data, idx):
        dst = [np.sum(np.linalg.norm(data[idx==k]-self.mu[k], axis=1)) for k in range(self.K)]
        return np.sum(dst)/(self.N*self.M)
