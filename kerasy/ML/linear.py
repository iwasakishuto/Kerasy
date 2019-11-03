# coding: utf-8
import numpy as np

class BayesianLinearRegression():
    def __init__(self, alpha=1, beta=25):
        self.alpha=alpha
        self.beta=beta
        self.SN=None
        self.mN=None

    def fit(self, train_x, train_y):
        N,M = train_x.shape
        self.SN = np.linalg.inv(self.alpha*np.eye(M) + self.beta*train_x.T.dot(train_x))
        self.mN = self.beta*self.SN.dot(train_x.T.dot(train_y))

    def predict(self, X):
        mu = np.array([self.mN.T.dot(x) for x in X])
        std = np.sqrt(np.array([1/self.beta + x.T.dot(self.SN).dot(x) for x in X]))
        return (mu,std)

class EvidenceApproxBayesianRegression(BayesianLinearRegression):
    def __init__(self, iter_max=100, alpha=1, beta=1):
        super().__init__(alpha=alpha, beta=beta)
        self.iter_max = iter_max
        self.gamma = None

    def fit(self, train_x, train_y):
        N,M = train_x.shape
        Phi = train_x.T.dot(train_x)
        eigvals = np.linalg.eigvals(Phi)
        for it in range(self.iter_max):
            params = [self.alpha, self.beta]
            super().fit(train_x, train_y)
            lambda_ = self.beta * eigvals
            self.gamma = np.sum(lambda_ / (self.alpha + lambda_))
            self.alpha = self.gamma / self.mN.dot(self.mN)
            self.beta  = (N-self.gamma) / np.sum( (train_y-train_x.dot(self.mN))**2 )
            if np.allclose(params, [self.alpha, self.beta]): break
        super().fit(train_x, train_y)

    def evidence(self, train_x, train_y):
        """ loglikelihood of marginalization ln p(y|α,β) PRML(3.86) """
        N,M = train_x.shape
        A = self.alpha*np.eye(M) + self.beta*train_x.T.dot(train_x)
        EmN = self.beta/2 * np.sum( (train_y - train_x.dot(self.mN))**2 ) + self.alpha/2 * self.mN.dot(self.mN)
        logmarg = M/2*np.log(self.alpha) + N/2*np.log(self.beta) - EmN - 1/2*np.log(np.linalg.det(A)) - N/2*np.log(2*np.pi)
        return logmarg
