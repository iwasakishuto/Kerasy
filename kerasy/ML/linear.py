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
