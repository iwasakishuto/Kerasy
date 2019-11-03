import numpy as np

class GaussianBaseTransformer():
    __name__ = "Gaussian"

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

class SigmoidBaseTransformer():
    __name__ = "Sigmoid"

    def __init__(self, mu, sigma=0.1):
        """
        @param mu   : (ndarray) shape=(M,D)
        @param sigma: (float)
        """
        if mu.ndim == 1: mu=np.expand_dims(mu,axis=1)
        self.mu = mu
        self.sigma = sigma

    def _logistic(self, x, mu):
        """
        @param x:  (ndarray) shape=(N,D)
        @param mu: (ndarray) shape=(D,)
        """
        return 1 / (1 + np.exp(- (x-mu)/self.sigma ))

    def transform(self, x):
        """
        @param  x           : (ndarray) shape=(N,D)
        @return x_tranformed: (ndarray) shape=(N,M)
        """
        if x.ndim == 1: x = np.expand_dims(x,axis=1)
        x_tranformed = np.asarray([self._logistic(x, mu) for mu in self.mu]).transpose()[0]
        return x_tranformed

class PolynomialBaseTransformer():
    __name__ = "Polynomial"

    def __init__(self, M):
        self.M = M

    def transform(self, x):
        """
        @param  x           : (ndarray) shape=(N,D)
        @return x_tranformed: (ndarray) shape=(N,M)
        """
        x_tranformed = np.asarray([x ** i for i in range(self.M+1)]).transpose()
        return x_tranformed
