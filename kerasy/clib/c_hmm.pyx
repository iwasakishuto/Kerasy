# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport view, floating
from numpy.math cimport INFINITY
from _cutils cimport c_argmax, c_max, c_logsumexp, c_logaddexp

# Log-scaled forward algorithm
def _log_forward(
        np.ndarray[floating, ndim=1, mode='c'] log_initial_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_transit_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_cond_prob):
    """ Forward Algorhtm for HMM. alpha(z_n) = p(x_1, ..., x_n, z_n)
    @params log_initial_prob : Log initial hidden states probabilities. shape=(n_hstates)
    @params log_transit_prob : Log transition probabilities. shape=(n_hstates, n_hstates)
    @params log_cond_prob    : Log conditional probabilities. shape=(n_samples, n_hstates)
    @return log_alpha        : Log scaled alpha. shape=(n_samples, n_hstates)
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = log_cond_prob.shape[0]
    cdef int n_hstates = log_cond_prob.shape[1]
    cdef np.ndarray[floating, ndim=2, mode='c'] log_alpha = np.empty(shape=(n_samples, n_hstates), dtype=dtype)
    cdef np.ndarray[floating, ndim=1, mode='c'] buffer = np.empty(shape=(n_hstates), dtype=dtype)

    cdef int n, i, k
    # (PRML 13.37) alpha(z_1) = p(x_1,z_1) = p(z_1)*p(x_1|z_1)
    for i in range(n_hstates):
        log_alpha[0, i] = log_initial_prob[i] + log_cond_prob[0, i]

    for n in range(1, n_samples):
        for k in range(n_hstates):
            for i in range(n_hstates):
                buffer[i] = log_alpha[n-1,i] + log_transit_prob[i,k]
            # (PRML 13.36) alpha(z_n) = p(x_n|z_n) sum(alpha(z_{n-1})*p(z_n|z_{n-1}))
            log_alpha[n,k] = c_logsumexp(buffer) + log_cond_prob[n,k]
    return log_alpha

# Log-scaled backward algorithm
def _log_backward(
        np.ndarray[floating, ndim=1, mode='c'] log_initial_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_transit_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_cond_prob):
    """ Backward Algorhtm for HMM. beta(z_n) = p(x_{n+1}, ..., x_N | z_n)
    @params log_initial_prob : Log initial hidden states probabilities. shape=(n_hstates)
    @params log_transit_prob : Log transition probabilities. shape=(n_hstates, n_hstates)
    @params log_cond_prob    : Log conditional probabilities. shape=(n_samples, n_hstates)
    @return log_beta         : Log scaled beta. shape=(n_samples, n_hstates)
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = log_cond_prob.shape[0]
    cdef int n_hstates = log_cond_prob.shape[1]
    cdef np.ndarray[floating, ndim=2, mode='c'] log_beta = np.empty(shape=(n_samples, n_hstates), dtype=dtype)
    cdef np.ndarray[floating, ndim=1, mode='c'] buffer = np.empty(shape=(n_hstates), dtype=dtype)

    cdef int n, i, k
    # (PRML 13.39) p(z_N|X) = p(X,z_N)*beta(z_n) / p(X) ∴ beta[z_N]=1
    for i in range(n_hstates):
        log_beta[n_samples-1, i] = 0.0

    # (PRML 13.38) beta(z_n) = sum_{z_{n+1}}(beta(z_{n+1})*p(x_{n+1}|z_{n+1})*p(z_{n+1}|z_n))
    for n in range(n_samples-2, -1, -1):
        for k in range(n_hstates):
            for i in range(n_hstates):
                buffer[i] = log_transit_prob[k,i] + log_cond_prob[n+1,i] + log_beta[n+1,i]
            log_beta[n,k] = c_logsumexp(buffer)
    return log_beta

# Log-scaled xi
def _compute_log_xi_sum(
        np.ndarray[floating, ndim=2, mode='c'] log_alpha,
        np.ndarray[floating, ndim=2, mode='c'] log_beta,
        np.ndarray[floating, ndim=2, mode='c'] log_transit_prob,
        np.ndarray[floating, ndim=2, mode='c'] log_cond_prob):
    """ Compute log-scaled xi(z_{n-1},z_n) from log-scaled alpha, and beta.
    @params log_alpha        : Log scaled alpha. shape=(n_samples, n_hstates)
    @params log_beta         : Log scaled beta.  shape=(n_samples, n_hstates)
    @params log_transit_prob : Log transition probabilities. shape=(n_hstates, n_hstates)
    @params log_cond_prob    : Log conditional probabilities. shape=(n_samples, n_hstates)
    @return log_xi_sum       : Log scaled xi. ( xi.shape=(n_samples-1, n_hstates, n_hstates) )
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = log_cond_prob.shape[0]
    cdef int n_hstates = log_cond_prob.shape[1]
    cdef np.ndarray[floating, ndim=2, mode='c'] log_xi_sum = np.full(shape=(n_hstates, n_hstates), fill_value=-INFINITY)
    # (PRML 13.42) p(X) = sum_{z_N}(alpha(z_N))
    cdef floating log_prob = c_logsumexp(log_alpha[n_samples-1])

    cdef int n, i, k
    # (PRML 13.43) xi(z_{n-1},z_n) = alpha(z_{n-1})*p(x_n|z_n)*p(z_n|z_{n-1})*beta(x_n) / p(X)
    for n in range(n_samples-1):
        for k in range(n_hstates):
            for i in range(n_hstates):
                # log-version "+="
                log_xi_sum[k,i] = c_logaddexp(
                    log_xi_sum[k,i],
                    log_alpha[n,k] + log_cond_prob[n+1,i] + log_transit_prob[k,i] + log_beta[n+1,i] - log_prob
                )
    return log_xi_sum

# Log-scaled viterbi algorithm.
def _viterbi(
         np.ndarray[floating, ndim=1, mode='c'] log_initial_prob,
         np.ndarray[floating, ndim=2, mode='c'] log_transit_prob,
         np.ndarray[floating, ndim=2, mode='c'] log_cond_prob):
    """ Viterbi Algorhtm.
    @params log_initial_prob : Log initial hidden states probabilities. shape=(n_hstates)
    @params log_transit_prob : Log transition probabilities. shape=(n_hstates, n_hstates)
    @params log_cond_prob    : Log conditional probabilities. shape=(n_samples, n_hstates)
    @return ml_hstates       : Maximum likelihood hidden state sequences. shape=(n_samples)
    @return log_prob         : Log scaled probability if the states were emitted from `ml_hstates`.
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = log_cond_prob.shape[0]
    cdef int n_hstates = log_cond_prob.shape[1]
    cdef np.ndarray[floating, ndim=2, mode='c'] omega = np.zeros((n_samples, n_hstates), dtype=dtype)
    cdef np.ndarray[int,      ndim=1, mode='c'] ml_hstates = np.empty(n_samples, dtype=np.int32)
    cdef np.ndarray[floating, ndim=1, mode='c'] buffer = np.empty(shape=(n_hstates), dtype=dtype)
    cdef int n, i, k, where_from

    # Dynamic Programming
    # Initialization
    # (PRML 13.69) omega(z_1) = log(p(z_1)) + log(p(x_1|z_1))
    for k in range(n_hstates):
        omega[0,k] = log_initial_prob[k] + log_cond_prob[0,k]
    # Reccursion
    # (PRML 13.68) omega(z_{n+1}) = log(p(x_{n+1}|z_{n+1})) + max_{z_n}(log(p(x_{n+1}|z_n)) + omega(z_n))
    for n in range(1, n_samples):
        for k in range(n_hstates):
            for i in range(n_hstates):
                buffer[i] = log_transit_prob[i,k] + omega[n-1,i]
            omega[n,k] = log_cond_prob[n,k] + c_max(buffer)

    # TraceBack
    # Because there is a unique path coming into that state we can trace the path
    # back to step N-1 to see what state it occupied at that time, and so on back
    # through the lattice to the state n=1.
    ml_hstates[n_samples-1] = where_from = c_argmax(omega[n_samples-1])
    cdef floating log_prob = omega[n_samples-1, where_from]
    for n in range(n_samples-2, -1, -1):
        for k in range(n_hstates):
            buffer[k] = omega[n,k] + log_transit_prob[k,where_from]
        ml_hstates[n] = where_from = c_argmax(buffer)

    return ml_hstates, log_prob
