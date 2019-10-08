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
