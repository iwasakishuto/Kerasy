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
            curt_target = 0.0
            for j in range(1, n_neighbors):
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
