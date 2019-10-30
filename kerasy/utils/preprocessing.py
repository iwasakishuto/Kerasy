import numpy as np

class GaussianBaseFunction():
    def __init__(self, mu, sigma=0.1):
        """
        @param mu   : (ndarray) shape=(M,D)
        @param sigma: (float)
        """
        if mu.ndim == 1: mu=np.expand_dims(mu,axis=1)
        self.mu = mu
        self.sigma = sigma

    def _gaussian(self, x, mu):
        """
        @param x:  (ndarray) shape=(N,D)
        @param mu: (ndarray) shape=(D,)
        """
        return np.exp(-1/2 * np.sum(np.square(x-mu), axis=-1) / self.sigma**2)

    def transform(self, x):
        """
        @param  x           : (ndarray) shape=(N,D)
        @return x_tranformed: (ndarray) shape=(N,M)
        """
        if x.ndim == 1: x = np.expand_dims(x,axis=1)
        x_tranformed = np.asarray([self._gaussian(x, mu) for mu in self.mu]).transpose()
        return x_tranformed
