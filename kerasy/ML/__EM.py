""" old version of EM.py This python file don't use the C-Extensions. """
# coding: utf-8
import numpy as np
import scipy.stats as stats

from ..utils import findLowerUpper
from ..utils import flush_progress_bar
from ..utils import paired_euclidean_distances
from ..utils import pairwise_euclidean_distances
from ..utils import silhouette_plot

class BaseEMmodel():
    def __init__(self, K, random_state=None, metrics="euclid"):
        self.K=K
        self.seed=random_state
        self.history=[]
        self.metrics=metrics

    def train_initialize(self, X, initializer="uniform"):
        """
        @param X: shape=(N,D)
        """
        self.history=[]
        N,D  = X.shape
        np.random.seed(self.seed)
        if initializer=="uniform":
            Xmin,Xmax = findLowerUpper(X, margin=0, N=None)
            self.mu = np.random.uniform(low=Xmin, high=Xmax, size=(self.K, D))
        elif initializer=="k++":
            data_idx = np.arange(N)
            idx = np.random.choice(data_idx)
            mu = X[idx, None]
            for k in range(1,self.K):
                min_distance = np.asarray([np.min([paired_euclidean_distances(x,m) for m in mu]) for x in X])
                idx = np.random.choice(data_idx, p=min_distance/np.sum(min_distance))
                mu = np.r_[mu, X[idx,None]]
            self.mu = mu
        else:
            handleKeyError(["uniform", "k++"], initializer=initializer)

    def fit(self, X, max_iter=100, memorize=False, initializer="uniform", verbose=1):
        raise NotImplementedError()
        #=== Initialization. ===
        self.train_initialize(X, initializer=initializer) # Initialize the mean value `self.mu` within data space.
        # If you need, you also initialize some parameters.
        #=== EM algorithm. ===
        for it in range(max_iter):
            responsibilities = self.Estep(X)
            if memorize: self.memorize_param(responsibilities)
            self.Mstep(X, responsibilities)
            ll = self.loglikelihood(X)
            flush_progress_bar(it, max_iter, metrics={"Log Likelihood": ll}, verbose=verbose)
            if it>0 and "Any Break Condition": break
        if memorize: self.memorize_param(responsibilities)
        if verbose>0: print()

    def predict(self, X):
        raise NotImplementedError()
        responsibilities = self.Estep(X)
        return responsibilities

    def Estep(self, X):
        raise NotImplementedError()

    def Mstep(self, X, responsibilities):
        raise NotImplementedError()

class KMeans(BaseEMmodel):
    def __init__(self, K, random_state=None):
        super().__init__(K=K, random_state=random_state)
        self.mu=None

    def fit(self, X, max_iter=100, memorize=False, initializer="uniform", verbose=1):
        """
        @param X: shape=(N,D)
        """
        # Initialization.
        self.train_initialize(X, initializer=initializer)
        # EM algorithm.
        for it in range(max_iter):
            idx = self.Estep(X)
            if memorize: self.memorize_param(idx)
            self.Mstep(X, idx)
            J = np.sum(paired_euclidean_distances(X, self.mu[idx]))
            flush_progress_bar(it, max_iter, metrics={"distortion measure": J}, verbose=verbose)
            if it>0 and not np.any(pidx!=idx): break
            pidx=idx
        if memorize: self.memorize_param(idx)
        if verbose>0: print()

    def memorize_param(self, idx):
        self.history.append([
            idx,
            np.copy(self.mu),
        ])

    def predict(self, X):
        """ Same with Estep """
        idx = self.Estep(X)
        return idx

    def Estep(self, X):
        """ X.shape=(N,D) """
        idx = np.asarray([np.argmin([np.sum(np.square(x-mu)) for mu in  self.mu]) for x in X], dtype=int)
        return idx

    def Mstep(self, X, idx):
        """ Assign a new average(mu) """
        for k in np.unique(idx):
            self.mu[k] = np.mean(X[idx==k], axis=0)

    def silhouette(self, X, axes=None, set_style=True):
        idx = self.Estep(X)
        silhouette_plot(X, idx, self.mu, axes=axes, set_style=set_style)

class HamerlyKMeans(KMeans):
    def __init__(self, K, random_state=None):
        super().__init__(K=K, random_state=random_state)
        self.lower=None   # shape=(N), Lower bound of distance to second nearest centroid.
        self.upper=None   # shape=(N), Upper bound of distance to nearest centroid.
        self.num_cls=None # shape=(K), Number of data belonging to class k.
        self.sum=None     # shape=(K,D)
        # self.mu=None      # shape=(K,D) mu = sum/num_cls

    def calcLowerBound(self, X, idx):
        """ Calculate the Lower Bounds for all data (Xi)
        @param X          : shape=(N,D)
        @param idx        : shape=(N,)
        @return second_min: shape=(N,) Minmum distance to the centroid which is not belong to.
        """
        data2centroids = pairwise_euclidean_distances(X, self.mu)
        return np.partition(data2centroids, kth=1, axis=1)[:,1]

    def Hamerly_initialize(self, X):
        """
        @param X: shape=(N,D)
        """
        N,D = X.shape
        idx = self.Estep(X)
        self.num_cls=np.zeros(shape=self.K)
        self.sum=np.zeros(shape=(self.K, D))
        for k in np.arange(self.K):
            self.num_cls[k]=np.count_nonzero(idx==k)
            self.sum[k]=np.sum(X[idx==k], axis=0)
        self.mu = self.sum/self.num_cls.reshape(-1,1)
        self.upper=paired_euclidean_distances(X, self.mu[idx])
        self.lower=self.calcLowerBound(X, idx)
        return idx

    def fit(self, X, max_iter=100, memorize=False, initializer="uniform", verbose=1):
        """ @param X: shape=(N,D) """
        N,D = X.shape
        dim = len(str(N))
        # Initialization.
        self.train_initialize(X, initializer=initializer)
        idx = self.Hamerly_initialize(X)
        if memorize: self.memorize_param(idx)
        # EM algorithm.
        for it in range(max_iter):
            not_meet, new_idx = self.SparseEstep(X, idx)
            changed = np.nonzero(new_idx!=idx[not_meet])[0]
            if len(changed)==0: break
            # Update centroid.
            dmu = self.Mstep(X[not_meet][changed], idx[not_meet][changed], new_idx[changed])
            # Update index.
            idx[not_meet] = new_idx
            self.updateUpperLower(idx, dmu)
            flush_progress_bar(it, max_iter, metrics={"changed": len(changed)}, verbose=verbose)
            if memorize: self.memorize_param(idx)
        if memorize: self.memorize_param(idx)
        if verbose>0: print()

    def SparseEstep(self, X, idx):
        """ Hamerly' Proposition
        Reduce the number of data to be calculated the distance using the Hamerly' Proposition.
        @param X  :
        @param idx:
        @return not_meet_again: shape=(?,) Index of data that did not meet the Hamerly' Proposition
        @return new_idx       : shape=(?,) X[not_meet_again]'s new idx.
        """
        pairwise_cent_dist = pairwise_euclidean_distances(self.mu, squared=False)
        pairwise_cent_dist += np.max(pairwise_cent_dist)*np.identity(self.K) # Add maximum to diagonal components.
        nearest_cent_dist = np.min(pairwise_cent_dist, axis=0)
        hamerly_right_side = np.maximum(nearest_cent_dist[idx]/2, self.lower)
        """ Hamerly' Proposition step.1 """
        not_meet = np.nonzero(self.upper>hamerly_right_side)[0] # Index of Xi who does not meet the Hamerly's Proposition.
        self.upper[not_meet] = paired_euclidean_distances(X[not_meet], self.mu[idx[not_meet]]) # Update the upper bound.
        """ Hamerly' Proposition step.2 """
        not_meet_again = not_meet[np.nonzero(self.upper[not_meet]>hamerly_right_side[not_meet])]
        # Apply Estep Only to `X[not_meet_again]`
        new_idx = self.Estep(X[not_meet_again])
        # Update lower bounds and upper bounds.
        self.upper[not_meet_again] = paired_euclidean_distances(X[not_meet_again], self.mu[new_idx])
        self.lower[not_meet_again] = self.calcLowerBound(X[not_meet_again], new_idx)

        changed = np.nonzero(new_idx==idx[not_meet_again])
        return not_meet_again, new_idx

    def Mstep(self, X, bid, aid):
        """Think about only changed data.
        @param X   : shape=(?, D)
        @param bid : shape=(?,) Before Id
        @param aid : shape=(?,) After Id
        @return dmu: shape=(K,) Delta distance of centoroids.
        """
        # Update the cluster.
        for k in np.arange(self.K):
            self.num_cls[k] -= np.count_nonzero(bid==k)
            self.sum[k]     -= np.sum(X[bid==k], axis=0)
            self.num_cls[k] += np.count_nonzero(aid==k)
            self.sum[k]     += np.sum(X[aid==k], axis=0)
        mu = self.sum/self.num_cls.reshape(-1,1)
        dmu = paired_euclidean_distances(mu, self.mu)
        self.mu = mu
        return dmu

    def updateUpperLower(self, idx, dmu):
        """ Update lower bounds and upper bounds. """
        max_dmu = np.max(dmu)
        self.upper += dmu[idx]
        self.lower -= max_dmu

class ElkanKMeans(KMeans):
    def __init__(self, K, random_state=None):
        super().__init__(K=K, random_state=random_state)
        self.lower=None   # shape=(N,K), Lower bound of distance to all centoroids.
        self.upper=None   # shape=(N), Upper bound of distance to nearest centroid.
        self.tight=None   # shape=(N), Whether Upper bounds are tight or not.
        self.num_cls=None # shape=(K), Number of data belonging to class k.
        self.sum=None     # shape=(K,D)
        # self.mu=None      # shape=(K,D) mu = sum/num_cls

    def Elkan_initialize(self, X):
        N,D = X.shape
        idx = self.Estep(X)
        self.num_cls=np.zeros(shape=self.K)
        self.sum=np.zeros(shape=(self.K, D))
        for k in np.arange(self.K):
            self.num_cls[k]=np.count_nonzero(idx==k)
            self.sum[k]=np.sum(X[idx==k], axis=0)
        self.mu = self.sum/self.num_cls.reshape(-1,1)
        self.upper=paired_euclidean_distances(X, self.mu[idx])
        self.lower=np.asarray([paired_euclidean_distances(x, self.mu) for x in X])
        self.tight=np.ones(shape=N, dtype=bool)
        return idx

    def fit(self, X, max_iter=100, memorize=False, initializer="uniform", verbose=1):
        """ @param X: shape=(N,D) """
        N,D = X.shape
        dim = len(str(N))
        # Initialization.
        self.train_initialize(X, initializer=initializer)
        idx = self.Elkan_initialize(X)
        if memorize: self.memorize_param(idx)
        # EM algorithm.
        for it in range(max_iter):
            changed,idx = self.SparseEstep(X, idx)
            if changed==0: break
            # Update centroid.
            dmu = self.Mstep()
            self.updateUpperLower(idx, dmu)
            flush_progress_bar(it, max_iter, metrics={"changed": changed}, verbose=verbose)
            if memorize: self.memorize_param(idx)
        if memorize: self.memorize_param(idx)
        if verbose>0: print()

    def SparseEstep(self, X, idx):
        changed = 0
        # 1) For all clusters c and c', compute d(c,c').
        pairwise_center_half_dist = 1/2 * pairwise_euclidean_distances(self.mu, squared=False)
        # For all centers, compute s(c) = 1/2 min_{c'≠c}d(c,c')
        nearest_center_dist = np.partition(pairwise_center_half_dist, kth=1, axis=0)[1]
        for i,x in enumerate(X):
            ub = self.upper[i]
            cls = idx[i]
            # 2) This means that the nearlest center is far away from the currently assigned center.
            if nearest_center_dist[cls] >= ub:
                continue

            for k in range(self.K):
                # 3. If this hold, then k is a good candidate fot the sample to be relabeled.
                if (k!=cls and (ub > self.lower[i,k]) and (ub > pairwise_center_half_dist[k,cls])):
                    # 3.a Recomputing the actual distance between sample and label.
                    if not self.tight[i]:
                        ub = paired_euclidean_distances(x,self.mu[cls])
                        self.lower[i,cls] = ub
                        self.tight[i] = True

                    # 3.b
                    if (ub>self.lower[i,k] or ub>pairwise_center_half_dist[cls,k]):
                        dist = paired_euclidean_distances(x,self.mu[k])
                        self.lower[i,k] = dist
                        if dist < ub:
                            self.num_cls[cls] -= 1; self.sum[cls] -= x
                            self.num_cls[k] += 1; self.sum[k] += x
                            cls = k
                            ub = dist
            self.upper[i] = ub
            if idx[i] != cls:
                changed += 1
                idx[i]   = cls
        return changed,idx

    def Mstep(self):
        mu  = self.sum/self.num_cls.reshape(-1,1)
        dmu = paired_euclidean_distances(mu, self.mu)
        self.mu = mu
        return dmu

    def updateUpperLower(self, idx, dmu):
        """ Update lower bounds and upper bounds. """
        self.lower = np.maximum(self.lower-dmu, 0)
        self.upper += dmu[idx]
        self.tight[:] = False

class MixedGaussian(BaseEMmodel):
    def __init__(self, K, random_state=None):
        super().__init__(K=K, random_state=random_state)
        self.mu=None
        self.S=None
        self.pi=None

    def fit(self, X, max_iter=100, memorize=False, tol=1e-5, initializer="uniform", verbose=1):
        # Initialization.
        self.train_initialize(X, initializer=initializer) # Initialize the mean value `self.mu` within data space.
        self.S  = [1*np.eye(2) for k in range(self.K)] # Initialize with Diagonal matrix
        self.pi = np.ones(self.K)/self.K # Initialize with Uniform.
        # EM algorithm.
        for it in range(max_iter):
            gamma = self.Estep(X)
            if memorize: self.memorize_param(gamma)
            self.Mstep(X, gamma)
            ll = self.loglikelihood(X)
            flush_progress_bar(it, max_iter, metrics={"Log Likelihood": ll}, verbose=verbose)
            mus = np.copy(self.mu.ravel())
            if it>0 and np.mean(np.linalg.norm(mus-pmus)) < tol: break
            pmus = mus
        if memorize: self.memorize_param(gamma)
        print()

    def memorize_param(self, gamma):
        self.history.append([
            np.argmax(gamma, axis=1),
            (np.copy(self.mu), np.copy(self.S), np.copy(self.pi))
        ])

    def predict(self, X):
        gamma = self.Estep(X, normalized=False)
        return gamma

    def Estep(self, X, normalized=True):
        N,_ = X.shape
        gamma = np.zeros(shape=(N, self.K))
        for k in range(self.K):
            gamma[:,k] = self.pi[k] * stats.multivariate_normal(self.mu[k], self.S[k]).pdf(X)
        if normalized: gamma /= np.sum(gamma, axis=1)[:,None]
        return gamma

    def Mstep(self, X, gamma):
        N,_ = X.shape
        for k in range(self.K):
            Nk = np.sum(gamma[:, k])
            self.mu[k] = gamma[:,k].dot(X)/Nk
            self.S[k]  = (np.expand_dims(gamma[:,k], axis=1)*(X-self.mu[k])).T.dot(X-self.mu[k]) / Nk
            self.pi[k] = Nk/N

    def loglikelihood(self, X):
        return np.sum(np.log(np.sum([self.pi[k]*stats.multivariate_normal(self.mu[k], self.S[k]).pdf(X) for k in range(self.K)], axis=0)))
