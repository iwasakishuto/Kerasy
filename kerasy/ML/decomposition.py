#coding: utf-8
import numpy as np
from scipy import optimize

from ..utils import flush_progress_bar
from ..utils import pairwise_euclidean_distances

from ._kernel import kernel_handler
from ..clib import c_decomposition

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

class tSNE():
    """Stochastic Neighbor Embedding.
    """

    def __init__(self,
                 initial_momentum = 0.5,
                 final_momoentum = 0.8,
                 eta = 500,
                 min_gain = 0.1,
                 tol = 1e-5,
                 prec_max_iter = 50,
                 random_state = None):
        self.initial_momentum = initial_momentum
        self.final_momoentum = final_momoentum
        self.eta = eta
        self.min_gain = min_gain
        self.tol = tol
        self.prec_max_iter = prec_max_iter
        self.random_state = random_state

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

    def fit_transform(self, X, n_components=2, initial_dims=50, perplexity=30.0, epochs=1000, verbose=1):
        # If the number of initial features are too large, using PCA to reduce the dimentions.
        n_samples, n_ori_features = X.shape
        if n_ori_features > initial_dims:
            print(f"Preprocessing the data using PCA to reduce the dimentions {n_ori_features}→{initial_dims}")
            model = PCA(n_components=initial_dims)
            model.fit(X)
            X = model.transform(X)

        n_samples, n_features = X.shape
        # Initialization.
        Y = np.random.RandomState(self.random_state).randn(n_samples, n_components)
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones(shape=Y.shape)

        # Compute the p-value.
        P = self.adjustPrecisions(X, perplexity)
        P = P + P.T
        P = P / np.sum(P)
        P = P * 4.
        P = np.maximum(P, 1e-12)

        max_digit = len(str(epochs))
        for epoch in range(epochs):
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
            momentum = self.initial_momentum if epoch < 20 else self.final_momoentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < self.min_gain] = self.min_gain

            iY = momentum * iY - self.eta * (gains * dY)
            Y += iY
            Y  = Y - np.tile(np.mean(Y, axis=0), (n_samples, 1))
            # Compute current value of cost function
            KL = np.sum(P * np.log(P / Q))
            flush_progress_bar(epoch, epochs, metrics={"KL(P||Q)": KL}, verbose=verbose)
            # Stop lying about P-values
            if epoch == 100:
                P = P / 4.
        return Y

class UMAP():
    """ Uniform Manifold Approximation and Projection
    Finds a low dimensional embedding of the data that approximates an underlying manifold.
    ~~~
    @params metric      : The metric to use to compute distances in high dimensional space.
    @params metric_kwds : The parameters of metrics.
    @params min_dist    : The effective minimum distance between embedded points.
                          Smaller values will result in a more clustered/clumped
                          embedding where nearby points on the manifold are drawn
                          closer together, while larger values will result on a
                          more even dispersal of points.
    @params spread      : The effective scale of embedded points.
    """
    def __init__(self, metric="euclidean", metric_kwds=None, min_dist=0.1, spread=1.0, a=None, b=None, random_state=None, sigma_iter=40, sigma_init=1.0, sigma_tol=1e-5, sigma_lower=0, sigma_upper=np.inf):
        self.metric=metric
        self.metric_kwds=metric_kwds
        self.min_dist=min_dist
        self.sigma_iter  = sigma_iter
        self.sigma_init  = sigma_init
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.sigma_tol   = sigma_tol
        self.spread = spread
        self.a=a
        self.b=b
        self.random_state=random_state
        self.history = []

    def fit_transform(self, X, n_components=2, n_neighbors=15, epochs=1, learning_rate=1, verbose=1):
        self.n_neighbors=n_neighbors
        self.n_components=n_components
        n_samples, n_features = X.shape

        x_distances = pairwise_euclidean_distances(X, squared=False)
        # n_neighbors Nearest Neighbor samples (Including myself.)
        knn_indices = np.asarray(np.argpartition(x_distances, kth=n_neighbors, axis=1)[:,:n_neighbors], dtype=np.int32, order="c")
        # knn_dists = np.asarray([[x_distances[i,idx] for idx in row_indices] for i,row_indices in enumerate(knn_indices)])
        knn_dists = np.asarray(np.partition(x_distances, kth=n_neighbors, axis=1)[:,:n_neighbors], dtype=float, order="c")
        rhos = np.asarray(np.partition(knn_dists, kth=1, axis=1)[:,1], dtype=float, order="c")

        # Binary search to adjust sigmas (normalization factor) value
        sigmas = self._find_sigmas(distances=knn_dists, rhos=rhos)

        self.knn_indices = knn_indices
        self.knn_dists = knn_dists
        self.rhos = rhos
        self.sigmas = sigmas

        graphs = self.compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

        self.graphs = graphs

        probs = np.exp(-np.maximum(x_distances-rhos[:,None], 0)/sigmas[:,None]).T
        # P's diagonal parts are all 1 (= 1+1-1).
        P = probs + probs.T - np.multiply(probs, probs.T)
        # Probabilities in Low dimensional space.
        if self.a is None or self.b is None:
            self._find_ab_params()
        a,b = (self.a, self.b)

        # TODO: Initialize the coordinates of low-dimensional embeddings with "Graph Laplacian."
        y = np.random.RandomState(self.random_state).normal(size=(n_samples, n_components))
        for epoch in range(epochs):
            # Yij = yi - yj
            Y = np.expand_dims(y, 1) - np.expand_dims(y, 0)
            y_squared_distances = pairwise_euclidean_distances(y, squared=True)
            Q = 1/(1 + a*y_squared_distances**b)

            # Calculate the Cross Entropy.
            CE = np.sum( -P*np.log(Q + 1e-4) - (1-P)*np.log(1-Q+1e-4) )

            # Calculate the Cross Entropy's gradients.
            # adij^{2(b-1)}P
            first_term = a*P*(1e-8+y_squared_distances)**(b-1)
            # 1-P(X) / dij^2
            second_term = np.dot(1-P, np.power(1e-3+y_squared_distances, -1))
            np.fill_diagonal(second_term, 0) # Calibrating the numerical errors.
            second_term = second_term / np.sum(second_term, axis=1, keepdims=True)
            y_gradients = 2*b*np.sum(np.expand_dims(first_term-second_term, 2)*np.expand_dims(Q, 2)*Y, axis=1)

            # Update TODE: Implement SGD.
            y -= learning_rate * y_gradients
            flush_progress_bar(epoch, epochs, metrics={"CE(P,Q)": CE}, verbose=verbose)
        if verbose: print()
        return y

    def _find_sigmas(self, distances, rhos=None):
        """
        @params distances : shape=(n_samples, n_neighbors)
        """
        if rhos is None:
            rhos = np.asarray(np.partition(distances, kth=1, axis=1)[:,1], dtype=float, order="c")
        # Buffer.
        sigmas = np.zeros_like(rhos)
        c_decomposition.binary_sigma_search(
            distances=distances, rhos=rhos, sigmas=sigmas,
            n_neighbors=self.n_neighbors, max_iter=self.sigma_iter,
            init=self.sigma_init, lower=self.sigma_lower, upper=self.sigma_upper,
            tolerance=self.sigma_tol
        )
        return sigmas

    def compute_membership_strengths(self, knn_indices, knn_dists, sigmas, rhos, apply_set_operations=True, set_op_mix_ratio=1.0):
        # n_samples * n_neighbors
        data_size = knn_indices.size
        rows = np.zeros(data_size, dtype=np.int32)
        cols = np.zeros(data_size, dtype=np.int32)
        vals = np.zeros(data_size, dtype=float)
        c_decomposition.compute_membership_strengths(
            knn_indices=knn_indices, knn_dists=knn_dists,
            sigmas=sigmas, rhos=rhos,
            rows=rows, cols=cols, vals=vals,
        )

        n_samples = sigmas.shape[0]
        result = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n_samples, n_samples)
        )
        result.eliminate_zeros()

        if apply_set_operations:
            transpose = result.transpose()
            pfi = result.multiply(transpose) # pure fuzzy intersection
            pfu = result+transpose-pfi # pure fuzzy union
            result = set_op_mix_ratio*pfu + (1.0-set_op_mix_ratio)*pfi
        result.eliminate_zeros()
        return result

    def _find_ab_params(self):
        """
        Fit a, b params for the differentiable curve used in lower dimensional
        fuzzy simplicial complex construction.
        TODO: Implement `scipy.optimize.curve_fit` by myself.

        f(x) = (1+a*x^{2b})^{-1} → 1 if x<min_dist else e^{-x}-min_dist
        """
        curve = lambda x,a,b: 1.0/(1.0 + a * x ** (2 * b))
        X = np.linspace(0,3*self.spread,300)
        Y = np.where(X<self.min_dist, 1, np.exp(-(X-self.min_dist)/self.spread))
        (self.a, self.b) , _ = optimize.curve_fit(curve, X, Y)

    @staticmethod
    def sigma2num_neighbors(sigma, distance, rho):
        """ Ref: 3.1 Graph Construction. """
        # High dimensional probability p_{i|j}
        prob = np.exp(- np.maximum(distance-rho, 0)/sigma)
        k = np.power(2, np.sum(prob))
        return k

    def adjustNeighbors(self, distances, rhos, n_neighbors):
        """
        @params distances  : shape=(N,N) distance matrix.
        @params rhos       : shape=(N,) distance to the first nearest neighbor
        @params n_neighbors: (int) fixed number of neighbors.
        """
        sigmas = np.zeros(shape=(len(distances)))
        for i,(distance,rho) in enumerate(zip(distances, rhos)):
            """ Adjust sigma by binary search for each data. """
            sigma_lower, sigma_upper = (self.sigma_lower, self.sigma_upper)
            for _ in range(self.sigma_iter):
                sigma = (sigma_lower + sigma_upper) / 2
                k = self.sigma2num_neighbors(sigma, distance, rho)
                if np.abs(k-n_neighbors) <= self.sigma_tol:
                    break
                if k<n_neighbors:
                    sigma_lower = sigma
                else:
                    sigma_upper = sigma
                if np.abs(k-n_neighbors) <= self.sigma_tol:
                    break
            sigmas[i] = sigma
        return sigmas
