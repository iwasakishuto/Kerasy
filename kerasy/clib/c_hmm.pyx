# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport view, floating
from numpy.math cimport INFINITY
from _cutils cimport c_argmax, c_max, c_logsumexp, c_logaddexp

def _forward(
        np.ndarray[floating, ndim=1, mode='c'] log_initial_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_transit_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_prob):
    """ Forward Algorhtm for HMM. alpha(z_n) = p(x_1, ..., x_n, z_n)
    @params log_initial_prob : Log initial hidden states probabilities. shape=(n_hstates)
    @params log_transit_prob : Log transition probabilities. shape=(n_hstates, n_hstates)
    @params log_prob         : Log probabilities. shape=(n_samples, n_hstates)
    @return log_alpha        : Log shape=(n_samples, n_hstates)
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = log_prob.shape[0]
    cdef int n_hstates = log_prob.shape[1]
    cdef np.ndarray[floating, ndim=2, mode='c'] log_alpha = np.empty(shape=(n_samples, n_hstates), dtype=dtype)
    cdef np.ndarray[floating, ndim=1, mode='c'] buffer = np.empty(shape=(n_hstates), dtype=dtype)

    cdef int n, i, j
    # (PRML 13.37) alpha(z_1) = p(x_1,z_1) = p(z_1)*p(x_1|z_1)
    for i in range(n_hstates):
        log_alpha[0, i] = log_initial_prob[i] + log_prob[0, i]

    for n in range(1, n_samples):
        for j in range(n_hstates):
            for i in range(n_hstates):
                buffer[i] = log_alpha[n-1, i] + log_transit_prob[i, j]
            # (PRML 13.36) alpha(z_n) = p(x_n|z_n) sum(alpha(z_{n-1})*p(z_n|z_{n-1}))
            log_alpha[n, j] = c_logsumexp(buffer) + log_prob[n, j]
    return log_alpha

def _backward(int n_samples, int n_hstates,
              double[:] log_initial_prob,
              double[:, :] log_transit_prob,
              double[:, :] log_prob,
              double[:, :] bwdlattice):

    cdef int t, i, j
    cdef double[::view.contiguous] work_buffer = np.zeros(n_hstates)

    with nogil:
        for i in range(n_hstates):
            bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_hstates):
                for j in range(n_hstates):
                    work_buffer[j] = (log_transit_prob[i, j]
                                      + log_prob[t + 1, j]
                                      + bwdlattice[t + 1, j])
                bwdlattice[t, i] = c_logsumexp(work_buffer)


def _compute_log_xi_sum(int n_samples, int n_hstates,
                        double[:, :] log_alpha,
                        double[:, :] log_transit_prob,
                        double[:, :] bwdlattice,
                        double[:, :] log_prob,
                        double[:, :] log_xi_sum):

    cdef int t, i, j
    cdef double[:, ::view.contiguous] work_buffer = \
        np.full((n_hstates, n_hstates), -INFINITY)
    cdef double logprob = c_logsumexp(log_alpha[n_samples - 1])

    with nogil:
        for t in range(n_samples - 1):
            for i in range(n_hstates):
                for j in range(n_hstates):
                    work_buffer[i, j] = (log_alpha[t, i]
                                         + log_transit_prob[i, j]
                                         + log_prob[t + 1, j]
                                         + bwdlattice[t + 1, j]
                                         - logprob)

            for i in range(n_hstates):
                for j in range(n_hstates):
                    log_xi_sum[i, j] = c_logaddexp(log_xi_sum[i, j],
                                                  work_buffer[i, j])


def _viterbi(int n_samples, int n_hstates,
             double[:] log_initial_prob,
             double[:, :] log_transit_prob,
             double[:, :] log_prob):

    cdef int i, j, t, where_from
    cdef double logprob

    cdef int[::view.contiguous] state_sequence = \
        np.empty(n_samples, dtype=np.int32)
    cdef double[:, ::view.contiguous] viterbi_lattice = \
        np.zeros((n_samples, n_hstates))
    cdef double[::view.contiguous] work_buffer = np.empty(n_hstates)

    with nogil:
        for i in range(n_hstates):
            viterbi_lattice[0, i] = log_initial_prob[i] + log_prob[0, i]

        # Induction
        for t in range(1, n_samples):
            for i in range(n_hstates):
                for j in range(n_hstates):
                    work_buffer[j] = (log_transit_prob[j, i]
                                      + viterbi_lattice[t - 1, j])

                viterbi_lattice[t, i] = c_max(work_buffer) + log_prob[t, i]

        # Observation traceback
        state_sequence[n_samples - 1] = where_from = \
            c_argmax(viterbi_lattice[n_samples - 1])
        logprob = viterbi_lattice[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_hstates):
                work_buffer[i] = (viterbi_lattice[t, i]
                                  + log_transit_prob[i, where_from])

            state_sequence[t] = where_from = c_argmax(work_buffer)

    return np.asarray(state_sequence), logprob
