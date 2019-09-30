# coding: utf-8
import numpy as np

class KMeans():
    def __init__(self, K=3, random_state=None):
        self.K=K
        self.N=None
        self.M=None
        self.mu = None
        self.seed=random_state
        self.history=[]

    def fit(self, data):
        self.N, self.M = data.shape
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)

        #=== Initialization ===
        np.random.seed(self.seed)
        self.mu = np.random.uniform(low=mins, high=maxs, size=(self.K, self.M))
        pidx = np.full(fill_value=self.K, shape=(self.N,))
        while True:
            idx = self.Estep(data)
            self.history.append([idx, np.copy(self.mu)])
            self.Mstep(data, idx)
            if np.all((idx-pidx)==0): break
            pidx = idx

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
