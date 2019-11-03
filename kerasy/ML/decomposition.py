#coding: utf-8
import numpy as np

class PCA():
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean = None
        self.S = None

    def fit(self,X):
        if self.n_components is None:
            self.n_components = min(X.shape)
        N,D = X.shape
        # X_mean = np.mean(X, axis=0) # X_ave.shape=(D,)
        # X_centralized = X-X_mean    # X_centralized.shape=(N,D)
        S = np.cov(X.T)               # S[i][j] = np.mean(X_centralized[:,i]*X_centralized[:,j])
        eigenvals, eigenvecs = np.linalg.eig(S)
        # NOTE: v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

        total_var = eigenvals.sum()
        explained_variance_ratio_ = eigenvals / total_var

        # Memorization.
        self.components_ = eigenvecs[:,:self.n_components].T # shape=(n_components,D)
        self.explained_variance_ = eigenvals[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        # self.mean = X_mean

    def transform(self,X):
        # X_centralized = X-self.mean
        X_transformed = np.dot(X, self.components_.T) # (N,D)@(M,D).T = (N,M)
        return X_transformed

class LDA():
    """ Linear Discriminant Analysis """
    def __init__(self):
        self.m1 = None; self.m2 = None # shape=(D,)
        self.SW = None # shape=(D,D)
        self.w  = None # shape=(D,)

    def fit(self, x1, x2, mean_only=False, normlized=True):
        """
        @param x1: shape=(N1, D)
        @param x2: shape=(N2, D)
        """
        _,D = x1.shape
        self.m1 = np.mean(x1, axis=0); self.m2 = np.mean(x2, axis=0)

        self.SW = np.zeros(shape=(D,D))
        for x in x1: self.SW += (x-self.m1)[:,None].dot((x-self.m1)[None,:]) # (D,1)@(1,D)
        for x in x2: self.SW += (x-self.m2)[:,None].dot((x-self.m2)[None,:]) # (D,1)@(1,D)
        if mean_only: self.SW = np.eye(D)

        self.w = np.linalg.inv(self.SW).dot(self.m2-self.m1) # (D,D)@(D,) = (D,)
        if normlized: self.w /= np.sqrt(np.sum(self.w**2))

    def transform(self, x):
        """
        @param  x            : shape=(N,D)
        @return x_transformed: shape=()
        """
        # Fisher's linear discriminant.
        x_transformed = x.dot(self.w) # (N,D)@(D,) = (N,)
        return x_transformed
