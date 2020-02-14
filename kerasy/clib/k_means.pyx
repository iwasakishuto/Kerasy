# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport sqrt
from ..utils import pairwise_euclidean_distances
from ..utils import flush_progress_bar

cdef floating euclidean_dist(floating* a, floating* b, int n_features) nogil:
    """
    Compute the euclidean distance between `a` and `b`. (Each data has the `n_features` features.)
    As the function start with `nogil`, this function will be executed without GIL.
    """
    cdef floating result, diff
    result = 0
    cdef int i
    for i in range(n_features):
        diff = (a[i] - b[i])
        result += diff * diff
    return sqrt(result)

cdef update_labels_distances_inplace(
        floating* X, floating* centers,floating[:, :] half_cent2cent,
        int[:] labels, floating[:, :] lower_bounds, floating[:] upper_bounds,
        Py_ssize_t n_samples, int n_features, int n_clusters):
    """
    @params X              : The input data. shape=(n_samples, n_features)
    @params centers        : The cluster centers. shape=(n_clusters, n_features)
    @params half_cent2cent : The half of the distance between any 2 clusters centers. shape=(n_clusters, n_clusters)
    @params labels         : The label for each sample. shape(n_samples)
    @params lower_bounds   : The lower bound on the distance between a sample and each cluster center. shape=(n_samples, n_clusters)
    @params upper_bounds   : The upper bound on the distance of each sample from its closest cluster center. shape=(n_samples,)
    @params n_samples      : The number of samples.
    @params n_features     : The number of features.
    @params n_clusters     : The number of clusters.
    """
    cdef floating* x
    cdef floating* c
    cdef floating d_c, dist
    cdef int c_x, k
    cdef Py_ssize_t idx
    for idx in range(n_samples):
        # first cluster (exception)
        c_x = 0
        x = X + idx * n_features
        d_c = euclidean_dist(x, centers, n_features) # temporal upper bound.
        lower_bounds[idx, 0] = d_c
        for k in range(1, n_clusters):
            # If this hold, then k is a good candidate fot the sample to be relabeled.
            if d_c > half_cent2cent[c_x, k]:
                c = centers + k * n_features
                dist = euclidean_dist(x, c, n_features)
                lower_bounds[idx, k] = dist
                if dist < d_c:
                    d_c = dist
                    c_x = k
        labels[idx] = c_x
        upper_bounds[idx] = d_c

def _centers_dense(
        np.ndarray[floating, ndim=2] X,
        np.ndarray[floating, ndim=1] sample_weight,
        np.ndarray[int, ndim=1]      labels,
        int n_clusters,
        np.ndarray[floating, ndim=1] distances):
    """ M step of the K-means EM algorithm
    @params X             : Input data. shape=(n_samples, n_features)
    @params sample_weight : The weights for each observation in X. shape=(n_samples,)
    @params labels        : Current label assignment. shape=(n_samples,)
    @params n_clusters    : Number of desired clusters.
    @params distances     : Distance to closest cluster for each sample. shape=(n_samples)
    @return centers       : The resulting centers. shape=(n_clusters, n_features)
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int i, j, c
    cdef np.ndarray[floating, ndim=2] centers = np.zeros((n_clusters, n_features), dtype=dtype)
    cdef np.ndarray[floating, ndim=1] weight_in_cluster = np.zeros((n_clusters,), dtype=dtype)

    for i in range(n_samples):
        weight_in_cluster[labels[i]] += sample_weight[i]
    empty_clusters = np.where(weight_in_cluster==0)[0]

    # If there is a cluster to which no data belongs, the furthest data is considered to be the new center.
    if len(empty_clusters):
        far_from_centers = distances.argsort()[::-1]
        for i,k in enumerate(empty_clusters):
            far_index = far_from_centers[i]
            new_center = X[far_index] * sample_weight[far_index]
            centers[k] = new_center
            weight_in_cluster[k] = sample_weight[far_index]

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i,j]*sample_weight[i]

    centers /= weight_in_cluster[:, np.newaxis]
    return centers

def k_means_elkan(np.ndarray[floating, ndim=2, mode='c'] X_,
                  np.ndarray[floating, ndim=1, mode='c'] sample_weight,
                  int n_clusters,
                  np.ndarray[floating, ndim=2, mode='c'] init,
                  float tol=1e-4, int max_iter=30, verbose=False):
    """Run Elkan's k-means.
    @params X_            : The input data. shape=(n_samples, n_features)
    @params sample_weight : The weights for each observation in X. shape=(n_samples,)
    @params n_clusters    : Number of clusters to find.
    @params init          : Initial position of centers.
    @params tol           : The relative increment in cluster means before declaring convergence.
    @params max_iter      : Maximum number of iterations of the k-means algorithm. (>0)
    @params verbose       : Whether to be verbose.
    """
    dtype = np.float32 if floating is float else np.float64

    # Initialization.
    cdef np.ndarray[floating, ndim=2, mode='c'] centers_ = init
    cdef floating* centers_p = <floating*>centers_.data
    cdef floating* X_p = <floating*>X_.data
    cdef floating* x_p
    cdef Py_ssize_t n_samples = X_.shape[0]
    cdef Py_ssize_t n_features = X_.shape[1]
    cdef Py_ssize_t idx
    cdef int k, label
    cdef floating ub, dist
    cdef floating[:, :] half_cent2cent = pairwise_euclidean_distances(centers_) / 2.
    cdef floating[:, :] lower_bounds = np.zeros((n_samples, n_clusters), dtype=dtype)
    cdef floating[:] nearest_center_dist
    labels_ = np.empty(n_samples, dtype=np.int32)
    cdef int[:] labels = labels_
    upper_bounds_ = np.empty(n_samples, dtype=dtype)
    cdef floating[:] upper_bounds = upper_bounds_

    # Get the initial set of upper bounds and lower bounds for each sample.
    update_labels_distances_inplace(X_p, centers_p, half_cent2cent,
                                    labels, lower_bounds, upper_bounds,
                                    n_samples, n_features, n_clusters)
    cdef np.uint8_t[:] is_tight = np.ones(n_samples, dtype=bool)
    cdef np.ndarray[floating, ndim=2, mode='c'] new_centers

    col_indices = np.arange(half_cent2cent.shape[0], dtype=np.int)
    for it in range(max_iter):
        # 1) For all clusters c and c', compute d(c,c').
        pairwise_center_half_dist = np.asarray(half_cent2cent)
        # nearest_center_dist[k] means the distance from center k to nearest center j(≠k)
        nearest_center_dist = np.partition(pairwise_center_half_dist, kth=1, axis=0)[1]
        for idx in range(n_samples):
            ub = upper_bounds[idx]
            label = labels[idx]
            # 2) This means that the nearlest center is far away from the currently assigned center.
            if nearest_center_dist[label] >= ub:
                continue

            x_p = X_p + idx * n_features
            for k in range(n_clusters):
                # 3) If this hold, then k is a good candidate fot the sample to be relabeled.
                if (k!=label and (ub > lower_bounds[idx, k]) and (ub > half_cent2cent[k, label])):
                    # 3.a Recomputing the actual distance between sample and label.
                    if not is_tight[idx]:
                        ub = euclidean_dist(x_p, centers_p + label*n_features, n_features)
                        lower_bounds[idx, label] = ub
                        is_tight[idx] = True

                    # 3.b
                    if (ub > lower_bounds[idx, k] or (ub > half_cent2cent[label, k])):
                        dist = euclidean_dist(x_p, centers_p + k*n_features, n_features)
                        lower_bounds[idx, k] = dist
                        if dist < ub:
                            label = k
                            ub = dist
            upper_bounds[idx] = ub
            labels[idx] = label

        # compute new centers
        new_centers = _centers_dense(X_, sample_weight, labels_, n_clusters, upper_bounds_)
        is_tight[:] = False

        center_shift = np.sqrt(np.sum((centers_ - new_centers) ** 2, axis=1))
        center_shift_total = np.sum(center_shift ** 2)

        # Update lower bounds and upper bounds.
        lower_bounds = np.maximum(lower_bounds - center_shift, 0)
        upper_bounds = upper_bounds + center_shift[labels_]

        # Reassign centers
        centers_ = new_centers
        centers_p = <floating*>new_centers.data
        half_cent2cent = pairwise_euclidean_distances(centers_) / 2.

        flush_progress_bar(
            it, max_iter, verbose=verbose,
            metrics={
                "inertia" : np.sum((X_ - centers_[labels]) ** 2 * sample_weight[:,np.newaxis]),
                "center shift": center_shift_total,
            }
        )

        if center_shift_total < tol:
            break

    # We need this to make sure that the labels give the same output as
    # predict(X)
    if center_shift_total > 0:
        update_labels_distances_inplace(X_p, centers_p, half_cent2cent,
                                        labels, lower_bounds, upper_bounds,
                                        n_samples, n_features, n_clusters)
    return centers_, labels_, it+1
