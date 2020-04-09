# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

from cython cimport floating

def binary_sigma_search(
        np.ndarray[floating, ndim=2, mode='c'] distances,
        np.ndarray[floating, ndim=1, mode='c'] rhos,
        np.ndarray[floating, ndim=1, mode='c'] sigmas,
        int n_neighbors, max_iter=20,
        float init=1.0, lower=0.0, upper=INFINITY, tolerance=1e-5, bandwidth=1.0):

    cdef int i,j,it
    cdef float curt_val, curt_target, d, lb, ub
    cdef float target = np.log2(n_neighbors)*bandwidth

    for i in range(sigmas.shape[0]):
        lb = lower
        ub = upper
        curt_val = init

        # <START> BINARY SEARCH
        for it in range(max_iter):
            # curt_target = func(curt_val, dists)
            curt_target = -1.0 # Because distances contains self data point.
            for j in range(n_neighbors):
                d = distances[i,j] - rhos[i]
                if d > 0:
                    curt_target += np.exp(-(d/curt_val))
                else:
                    curt_target += 1.0

            if np.fabs(curt_target - target) < tolerance:
                break
            if curt_target > target:
                ub = curt_val
                curt_val = (lb+ub)/2.0
            else:
                lb = curt_val
                if ub == upper:
                    curt_val *= 2
                else:
                    curt_val = (lb+ub)/2.0
        # <END> BINARY SEARCH

        sigmas[i] = curt_val

def compute_membership_strengths(
        np.ndarray[np.int32_t, ndim=2, mode='c'] knn_indices,
        np.ndarray[floating,   ndim=2, mode='c'] knn_dists,
        np.ndarray[floating,   ndim=1, mode='c'] sigmas, rhos,
        np.ndarray[np.int32_t, ndim=1, mode='c'] rows, cols,
        np.ndarray[floating,   ndim=1, mode='c'] vals):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    ----------
    @params knn_indices: The indices on the `n_neighbors` closest points in the dataset.
    @params knn_dists  : The distances to the `n_neighbors` closest points in the dataset.
    @params sigmas     : The normalization factor derived from the metric tensor approximation.
    @params rhos       : The local connectivity adjustment.
    @buffer rows       : Row data for the resulting sparse matrix (coo format)
    @buffer clos       : Column data for the resulting sparse matrix (coo format)
    @buffer vals       : Entries for the resulting sparse matrix (coo format)
    -------
    """
    cdef int n_samples = knn_indices.shape[0]
    cdef int n_neighbors = knn_indices.shape[1]
    cdef int i,j
    cdef float val

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i,j] == i:
                val = 0.0
            elif knn_dists[i,j]-rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i,j] - rhos[i]) / (sigmas[i])))

            rows[i*n_neighbors + j] = i
            cols[i*n_neighbors + j] = knn_indices[i,j]
            vals[i*n_neighbors + j] = val
