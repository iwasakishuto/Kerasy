#coding: utf-8
import numpy as np

from ..utils import flush_progress_bar
from ..utils import pairwise_euclidean_distances

from ._kernel import kernel_handler

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
        X_mean = np.mean(X, axis=0).reshape(1,D) # X_ave.shape=(D,)
        X_centralized = X-X_mean      # X_centralized.shape=(N,D)
        S = np.cov(X_centralized.T)   # S[i][j] = np.mean(X_centralized[:,i]*X_centralized[:,j])
        eigenvals, eigenvecs = np.linalg.eig(S)
        # NOTE: v[:,i] is the eigenvector corresponding to the eigenvalue w[i].

        total_var = eigenvals.sum()
        explained_variance_ratio_ = eigenvals / total_var

        # Memorization.
        self.components_ = eigenvecs[:,:self.n_components].T # shape=(n_components,D)
        self.explained_variance_ = eigenvals[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        self.mean = X_mean

    def transform(self,X):
        X_centralized = X-self.mean
        X_transformed = np.dot(X_centralized, self.components_.T) # (N,D)@(M,D).T = (N,M)
        return X_transformed

class LDA():
    """ Linear Discriminant Analysis """
    def __init__(self):
        self.m1 = None; self.m2 = None # shape=(D,)
        self.SW = None # shape=(D,D)
        self.w  = None # shape=(D,)

    def fit(self, X1, X2, mean_only=False, normlized=True):
        """
        @param X1: shape=(N1, D)
        @param X2: shape=(N2, D)
        """
        _,D = X1.shape
        self.m1 = np.mean(X1, axis=0); self.m2 = np.mean(X2, axis=0)

        self.SW = np.zeros(shape=(D,D))
        for x in X1: self.SW += (x-self.m1)[:,None].dot((x-self.m1)[None,:]) # (D,1)@(1,D)
        for x in X2: self.SW += (x-self.m2)[:,None].dot((x-self.m2)[None,:]) # (D,1)@(1,D)
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

class tSNE():
    """Stochastic Neighbor Embedding.
    """

    def __init__(self,
                 initial_momentum = 0.5,
                 final_momoentum = 0.8,
                 eta = 500,
                 min_gain = 0.1,
                 tol = 1e-5,
                 prec_max_iter = 50):
        self.initial_momentum = initial_momentum
        self.final_momoentum = final_momoentum
        self.eta = eta
        self.min_gain = min_gain
        self.tol = tol
        self.prec_max_iter = prec_max_iter

    @staticmethod
    def dist2shannon(Di, beta):
        """Compute Pi and the Shannon Entropy of Pi from Dij.
        ~~~~~~~~~~~Arguments
        @param Di      : shape=(n_samples-1,). Dij means the distances between Xi and Xj. (j≠i)
        @param beta    : 1/2 * precision = 1/(2*sigma^2).
        @return Pi     : Pi[j] means that the Conditional Propbability of Xj given Xi.
        @return logPerp: The log scaled perplexity under a given beta.
        --------------------
        Perp(Pi)       = 2^H(Pi)
        H(Pi)          = Shannon Entropy of Pi = -Σj pj|i log_2(pj|i)
        log(Perp(Pi))  = H(Pi) * log2
                       = -Σj pj|i log_e(pj|i)
        """
        propto_pji = np.exp(-Di*beta)
        Pi = propto_pji/np.sum(propto_pji) # Pi[j] = p_j|i
        logPerp = np.sum(-Pi*np.log(Pi))
        # propto_pji = np.exp(-Di*beta)
        # pji_norm = np.sum(propto_pji)
        # logPerp = np.log(pji_norm) + beta*np.sum(Di*propto_pji)/pji_norm
        # Pi = propto_pji / pji_norm
        return Pi, logPerp

    def adjustPrecisions(self, X, perplexity):
        """Performs a binary search to get precisions (beta) so that each
        conditional Gaussian has the same perplexity.
        """
        print(f"Each conditional Gaussian has the same perplexity: {perplexity}")
        n_samples, _ = X.shape
        D = pairwise_euclidean_distances(X, squared=True)
        # Initialization.
        P = np.zeros((n_samples, n_samples)) # Pij means that pj|i
        beta = np.ones(shape=(n_samples))    # Precisions
        goallogPrep = np.log(perplexity)     # Log scaled perplexity (AIM)

        for i in range(n_samples):
            """ Adjust precision by binary search for each data. """
            betamin, betamax = (-np.inf, np.inf)
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] # Remove D[i,i] from D[i]
            Pi, logPerp = self.dist2shannon(Di, beta[i])
            Prep_diff = logPerp - goallogPrep
            it = 0
            for it in range(self.prec_max_iter):
                if np.abs(Prep_diff) < self.tol: break

                # Binary Search
                if Prep_diff>0:
                    # Prep_diff>0 → current Perp > goal Prep → decrease the prep → increase the precision
                    betamin = beta[i]
                    beta[i] = beta[i]*2 if (betamax == np.inf) or (betamax == -np.inf) else (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i]
                    beta[i] = beta[i]/2 if (betamin == np.inf) or (betamin == -np.inf) else (beta[i] + betamin) / 2.

                Pi, logPerp = self.dist2shannon(Di, beta[i])
                Prep_diff = logPerp - goallogPrep
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] = Pi
        print(f"Mean value of sigma: {np.mean(np.sqrt(1/2*beta)):.3f}")
        return P

    def fit_transform(self, X, n_components=2, initial_dims=50, perplexity=30.0, max_iter=10, random_state=None):
        # If the number of initial features are too large, using PCA to reduce the dimentions.
        n_samples, n_ori_features = X.shape
        if n_ori_features > initial_dims:
            print(f"Preprocessing the data using PCA to reduce the dimentions {n_ori_features}→{initial_dims}")
            model = PCA(n_components=initial_dims)
            model.fit(X)
            X = model.transform(X)

        n_samples, n_features = X.shape
        # Initialization.
        Y = np.random.RandomState(random_state).randn(n_samples, n_components)
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones(shape=Y.shape)

        # Compute the p-value.
        P = self.adjustPrecisions(X, perplexity)
        P = P + P.T
        P = P / np.sum(P)
        P = P * 4.
        P = np.maximum(P, 1e-12)

        max_digit = len(str(max_iter))
        print()
        for it in range(max_iter):
            print(f"{it+1}/{max_iter}")
            for inner_it in range(100):
                # equation (4)
                propto_qij = 1. / (1. + pairwise_euclidean_distances(Y, squared=True))
                propto_qij[range(n_samples), range(n_samples)] = 0.
                Q = propto_qij / np.sum(propto_qij)
                Q = np.maximum(Q, 1e-12)

                # Compute the gradient
                PQ = P - Q
                for i in range(n_samples):
                    """
                    dC/dyi = 4 Σj (pij - qij)(yi - yj)(1 + ||yi-yj||^2)^(-1)    (5)
                    PQ[i].shape            = (n_samples, )
                    (Y[i,:] - Y).shape     = (n_samples, n_components)
                    propto_qij[i,: ].shape = (n_samples, )
                    """
                    dY[i, :] = np.dot(PQ[i,:]*propto_qij[i,:], Y[i,]-Y)

                # Perform the update
                momentum = self.initial_momentum if it < 20 else self.final_momoentum
                gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
                gains[gains < self.min_gain] = self.min_gain

                iY = momentum * iY - self.eta * (gains * dY)
                Y += iY
                Y  = Y - np.tile(np.mean(Y, axis=0), (n_samples, 1))
                flush_progress_bar(inner_it, 100, metrics=f"KL(P||Q) = {np.sum(P * np.log(P / Q)):.3f}")
            # Stop lying about P-values
            if it == 0: P = P / 4.
            # Compute current value of cost function
        return Y

class KernelPCA():
    def __init__(self, n_components=None, kernel="gaussian", **kernelargs):
        self.kernel = kernel_handler(kernel, **kernelargs)
        self.n_components = n_components

    def fit_transform(self, X):
        """
        @param X: shape=(N,M)
        """
        if self.n_components is None:
            self.n_components = min(X.shape)
        N, M = X.shape
        K = np.array([[self.kernel(xi, xj) for xj in X] for xi in X])
        I = 1/N * np.ones(shape=(N,N))
        K_tilde = K - I.dot(K) - K.dot(I) + np.dot(I,np.dot(K,I)) # shape=(N,N)
        eigenvals, eigenvecs = np.linalg.eig(K_tilde) # K@a_i = lambda_i*N*a_i
        total_var = eigenvals.sum()
        explained_variance_ratio_ = eigenvals / total_var

        # Memorization.
        self.components_ = eigenvecs[:,:self.n_components].T # shape=(n_components,N)
        self.explained_variance_ = eigenvals[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        self.N,self.M = N,M
        self.X = X
        X_transformed = np.dot(K_tilde, self.components_.T) # (N,N)@(N,n_components) = (N,n_components)
        return X_transformed
