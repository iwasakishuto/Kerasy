# coding: utf-8
import numpy as np
# import scipy.sparse as sp
import scipy.stats as stats

from ..utils import findLowerUpper
from ..utils import _check_sample_weight
from ..utils import flush_progress_bar
from ..utils import handleRandomState
from ..utils import paired_euclidean_distances
from ..utils import silhouette_plot
from ..clib import c_kmeans

class BaseEMmodel():
    def __init__(self, n_clusters=8, init="k++", random_state=None, metrics="euclid"):
        self.n_clusters=n_clusters
        self.rnd=handleRandomState(random_state)
        self.init=init
        self.history=[]
        self.metrics=metrics
        self.iterations_=0

    def _find_initial_centroids(self, X, n_clusters=8, init="k++", random_state=None):
        """Initialize the centroids.
        @params X         : array-like or sparse matrix, shape=(n_samples, n_features)
        @params n_clusters: (int) Number of centroids.
        @params init      : (str) 'k++', 'random', 'uniform'
                            (ndarray) shape=(n_clusters, n_features)
        @return centers   : (ndarray) centroids. shape=(n_clusters, n_features)
        """
        N,D = X.shape
        if hasattr(init, '__array__'):
            n_cluster, D_ = init.shape
            if n_cluster!=self.n_clusters or D_!=D:
                raise ValueError(
                    "If init is array, init.shape should be equal to (n_clusters, n_features)",
                    f"but init.shape=({n_cluster}, {D_}), and n_clusters={self.n_clusters}, n_features={D}")
            centers = init
        elif isinstance(init, str):
            rnd = handleRandomState(self.rnd if random_state is None else random_state)
            if init == "random":
                Xmin,Xmax = findLowerUpper(X, margin=0, N=None)
                centers = rnd.uniform(low=Xmin, high=Xmax, size=(self.n_clusters, D))
            elif init == 'k++':
                data_idx = np.arange(N)
                idx = rnd.choice(data_idx)
                centers = X[idx, None]
                for k in range(1,self.n_clusters):
                    min_distance = np.asarray([np.min([paired_euclidean_distances(x,c) for c in centers]) for x in X])
                    idx = rnd.choice(data_idx, p=min_distance/np.sum(min_distance))
                    centers = np.r_[centers, X[idx,None]]
            else:
                handleKeyError(["random", "k++"], initializer=initializer)
        else:
            raise ValueError(f"the `init` parameter should be 'k++', or 'random', or an array.shape=({n_cluster}, {D_})")
        return centers

    def _memorize_param(self, *args):
        self.history.append([np.copy(arg) if hasattr(arg, '__array__') else arg for arg in args])

class KMeans(BaseEMmodel):
    def __init__(self, n_clusters=8, init="k++", random_state=None, metrics="euclid"):
        super().__init__(n_clusters=n_clusters, init=init, random_state=random_state, metrics=metrics)
        self.centroids   = None
        self.iterations_ = 0
        self.labels_     = None
        self.inertia_    = None

    def fit(self, X, sample_weight=None, max_iter=300, memorize=False, tol=1e-4, verbose=1):
        X = X.astype(float)
        centroids = self._find_initial_centroids(X, n_clusters=self.n_clusters, init=self.init, random_state=self.rnd)
        sample_weight = _check_sample_weight(sample_weight, X)
        n_samples = X.shape[0]
        labels    = np.full(n_samples, -1, dtype=np.int32)
        distances = np.zeros(n_samples, dtype=X.dtype)
        for it in range(max_iter):
            if memorize:
                self._memorize_param(centroids)
            labels, inertia = self.Estep(X, centroids, labels=labels, distances=distances)
            new_centroids = self.Mstep(X, labels, sample_weight=sample_weight, distances=distances)
            center_shift_total = np.sum(np.sum((centroids-new_centroids)**2, axis=1))
            flush_progress_bar(
                it, max_iter, verbose=verbose, barname="KMeans Lloyd",
                metrics={
                    "average inertia" : f"{inertia/n_samples:.3f}",
                    "center shift total": f"{center_shift_total:.3f}",
                }
            )
            centroids = new_centroids
            if center_shift_total < tol:
                break
        if center_shift_total != 0:
            labels, inertia = self.Estep(X, centroids, labels=labels, distances=distances)
        self.centroids   = centroids
        self.iterations_ = it+1
        self.labels_     = labels
        self.inertia_    = inertia

    def predict(self, X):
        """ Same with Estep """
        labels, inertia = self.Estep(X, centroids=self.centroids)
        return labels

    def Estep(self, X, centroids, labels=None, distances=None):
        """ Compute the labels and the inertia of the given samples and centers. """
        if labels is None:
            n_samples = X.shape[0]
            labels = np.full(n_samples, -1, np.int32)
        if distances is None:
            # shape=(0,) means that don't store the distances.
            distances = np.zeros(shape=(0,), dtype=X.dtype)
        inertia = c_kmeans._kmeans_Estep(X, centroids, labels, distances)
        return labels, inertia

    def Mstep(self, X, labels, sample_weight=None, distances=None):
        sample_weight = _check_sample_weight(sample_weight, X)
        centers = c_kmeans._kmeans_Mstep(X, sample_weight, labels, self.n_clusters, distances=distances)
        return centers

    def silhouette(self, X, axes=None, set_style=True):
        labels = self.predict(X)
        silhouette_plot(X, labels, self.centroids, axes=axes, set_style=set_style)

class HamerlyKMeans(KMeans):
    def __init__(self, n_clusters=8, init="k++", random_state=None, metrics="euclid"):
        super().__init__(n_clusters=n_clusters, init=init, random_state=random_state, metrics=metrics)

    def fit(self, X, sample_weight=None, max_iter=300, memorize=False, tol=1e-4, verbose=1):
        init_centroids = self._find_initial_centroids(X, n_clusters=self.n_clusters, init=self.init, random_state=self.rnd)
        sample_weight = _check_sample_weight(sample_weight, X)
        self.centroids, self.inertia_, self.labels_, self.iterations_ = c_kmeans.k_means_hamerly(X, sample_weight, self.n_clusters, init_centroids, tol=tol, max_iter=max_iter, verbose=verbose)

class ElkanKMeans(KMeans):
    def __init__(self, n_clusters=8, init="k++", random_state=None, metrics="euclid"):
        super().__init__(n_clusters=n_clusters, init=init, random_state=random_state, metrics=metrics)

    def fit(self, X, sample_weight=None, max_iter=300, memorize=False, tol=1e-4, verbose=1):
        init_centroids = self._find_initial_centroids(X, n_clusters=self.n_clusters, init=self.init, random_state=self.rnd)
        sample_weight = _check_sample_weight(sample_weight, X)
        self.centroids, self.inertia_, self.labels_, self.iterations_ = c_kmeans.k_means_elkan(X, sample_weight, self.n_clusters, init_centroids, tol=tol, max_iter=max_iter, verbose=verbose)

class MixedGaussian(BaseEMmodel):
    def __init__(self, n_clusters=8, init="k++", random_state=None, metrics="euclid"):
        super().__init__(n_clusters=n_clusters, init=init, random_state=random_state, metrics=metrics)
        self.centroids=None
        self.S=None
        self.pi=None

    def fit(self, X, sample_weight=None, max_iter=300, memorize=False, tol=1e-4, verbose=1):
        self.centroids = self._find_initial_centroids(X, n_clusters=self.n_clusters, init=self.init, random_state=self.rnd) # Initialize the mean value `self.centroids` within data space.
        self.S  = [1*np.eye(2) for k in range(self.n_clusters)] # Initialize with Diagonal matrix
        self.pi = np.ones(self.n_clusters)/self.n_clusters # Initialize with Uniform.
        sample_weight = _check_sample_weight(sample_weight, X)
        # EM algorithm.
        for it in range(max_iter):
            gamma = self.Estep(X)
            if memorize: self._memorize_param(np.argmax(gamma, axis=1), self.centroids, self.S, self.pi)
            self.Mstep(X, gamma)
            ll = self.loglikelihood(X)
            flush_progress_bar(it, max_iter, metrics={"Log Likelihood": ll}, verbose=verbose)
            mus = np.copy(self.centroids.ravel())
            if it>0 and np.mean(np.linalg.norm(mus-pmus)) < tol: break
            pmus = mus
        if memorize: self._memorize_param(gamma)
        if verbose>0: print()

    def predict(self, X):
        gamma = self.Estep(X, normalized=False)
        labels = np.argmax(gamma, axis=1)
        return labels

    def Estep(self, X, normalized=True):
        N,_ = X.shape
        gamma = np.zeros(shape=(N, self.n_clusters))
        for k in range(self.n_clusters):
            gamma[:,k] = self.pi[k] * stats.multivariate_normal(self.centroids[k], self.S[k]).pdf(X)
        if normalized: gamma /= np.sum(gamma, axis=1)[:,None]
        return gamma

    def Mstep(self, X, gamma):
        N,_ = X.shape
        for k in range(self.n_clusters):
            Nk = np.sum(gamma[:, k])
            self.centroids[k] = gamma[:,k].dot(X)/Nk
            self.S[k]  = (np.expand_dims(gamma[:,k], axis=1)*(X-self.centroids[k])).T.dot(X-self.centroids[k]) / Nk
            self.pi[k] = Nk/N

    def loglikelihood(self, X):
        return np.sum(np.log(np.sum([self.pi[k]*stats.multivariate_normal(self.centroids[k], self.S[k]).pdf(X) for k in range(self.n_clusters)], axis=0)))
