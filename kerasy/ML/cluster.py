#coding: utf-8

import numpy as np
from ..utils import flush_progress_bar
from ..utils import paired_euclidean_distances
from ..utils import pairwise_euclidean_distances

# Ref: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/cluster/_dbscan_inner.pyx#L19
def dbscan_inner(is_core, neighborhoods, labels, verbose=1):
    """Algorithms for DBSCAN.
    @params is_core      : shape=(N,) Whether ith data is core or not.
    @params neighborhoods: shape=(N,) ith data's neighbor
    @params labels       : shape=(N,) Cluster labels for each point. Noisy samples are given the label -1.
    """
    label = 0
    stack = []
    num_data = labels.shape[0]
    for i in range(num_data):
        flush_progress_bar(i, num_data, metrics={"num cluster": label+2}, verbose=verbose)
        if labels[i] != -1 or not is_core[i]:
            continue
        # Depth-first search starting from i,
        # ending at the non-core points.
        while True:
            if labels[i] == -1:
                labels[i] = label
                if is_core[i]:
                    for j in neighborhoods[i]:
                        if labels[j] == -1:
                            stack.append(j)
            if len(stack) == 0:
                break
            i = stack.pop(-1)
        label+=1
    return labels

# Ref: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/cluster/_dbscan.py#L148
class DBSCAN():
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.
    """
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X, verbose=1):
        if not self.eps > 0.0: raise ValueError("`eps` must be positive.")
        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.int)
        # Distance Matrix of All-to-All.
        distances = pairwise_euclidean_distances(X, squared=False)
        neighborhoods = np.asarray([np.where(neighbors)[0] for neighbors in distances<=self.eps])
        n_neighbors = np.asarray([np.count_nonzero(dist<self.eps) for dist in distances], dtype=np.int)
        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        labels = dbscan_inner(core_samples, neighborhoods, labels, verbose=verbose)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
