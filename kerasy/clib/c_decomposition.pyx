# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

from cython cimport floating

from ..utils import flush_progress_bar
from _cutils cimport clip

cdef tau_rand_int(np.ndarray[np.int64_t, ndim=1, mode='c'] state):
    """A fast (pseudo)-random number generator.
    state: The internal state of the rng. array of int64, shape (3,)
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )
    return state[0] ^ state[1] ^ state[2]

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

def optimize_layout(
    np.ndarray[floating,   ndim=2, mode='c'] head_embedding, tail_embedding,
    np.ndarray[np.int32_t, ndim=1, mode='c'] head, tail,
    np.ndarray[floating,   ndim=1, mode='c'] epochs_per_sample,
    np.ndarray[np.int64_t, ndim=1, mode='c'] rng_state,
    int epochs, n_vertices, float a, b,
    float gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0, int verbose=1):

    cdef int i,j,k,p
    cdef int dim = head_embedding.shape[1]
    cdef int move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    cdef float alpha = initial_alpha

    cdef np.ndarray[floating, ndim=1, mode='c'] epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    cdef np.ndarray[floating, ndim=1, mode='c'] epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    cdef np.ndarray[floating, ndim=1, mode='c'] epoch_of_next_sample = epochs_per_sample.copy()

    for epoch in range(epochs):
        for i in range(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= epochs:
                j = head[i]
                k = tail[i]
                current = head_embedding[j]
                other = tail_embedding[k]
                dist_squared = sum((current-other)**2)

                if dist_squared > 0.0:
                    grad_coeff = -2.0 * a * b * pow(dist_squared, b-1.0)
                    grad_coeff /= a*pow(dist_squared, b)+1.0
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    grad_d = clip(val=grad_coeff*(current[d]-other[d]), v_min=-4.0, v_max=4.0)
                    current[d] += grad_d * alpha
                    if move_other:
                        other[d] += -grad_d * alpha
                epoch_of_next_sample[i] += epochs_per_sample[i]
                n_neg_samples = int((epochs-epoch_of_next_negative_sample[i])/epochs_per_negative_sample[i])

                for p in range(n_neg_samples):
                    k = tau_rand_int(rng_state) % n_vertices
                    other = tail_embedding[k]
                    dist_squared = sum((current-other)**2)

                    if dist_squared > 0.0:
                        grad_coeff = 2.0 * gamma * b
                        grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                    elif j == k:
                        continue
                    else:
                        grad_coeff = 0.0

                    for d in range(dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(val=grad_coeff*(current[d]-other[d]), v_min=-4.0, v_max=4.0)
                        else:
                            grad_d = 4.0
                        current[d] += grad_d * alpha

                epoch_of_next_negative_sample[i] += n_neg_samples*epochs_per_negative_sample[i]
        alpha = initial_alpha*(1.0-(float(epoch)/float(epochs)))
        flush_progress_bar(epoch, epochs, verbose=verbose)
    if verbose>0: print
