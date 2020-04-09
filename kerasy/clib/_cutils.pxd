# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

# Ref: <math.h> https://www.qnx.com/developers/docs/6.4.1/dinkum_en/c99/math.html
from numpy.math cimport expl, logl, log1pl, isinf, fabsl, INFINITY
from libc.math import sqrt, ceil

cdef inline double clip(double val, v_min, v_max):
    if val > v_max:
        return v_max
    elif val < v_min:
        return v_min
    else:
        return val

# np.argmax(X)
cdef inline int c_argmax(double[:] X) nogil:
    cdef double X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos

# np.max(X)
cdef inline double c_max(double[:] X) nogil:
    return X[c_argmax(X)]

# log(sum_i(exp(X_i)))
cdef inline double c_logsumexp(double[:] X) nogil:
    """ Prevent X[i] from overflowing and underflowing.
    log(sum_i(exp(X_i))) = log( exp(Xmax) * sum_i(exp(X_i - Xmax)))
                         = Xmax + log(sum_i(exp(X_i - Xmax)))
    """
    cdef double Xmax = c_max(X)
    if isinf(Xmax):
        return -INFINITY

    cdef double sumexp_ = 0
    for i in range(X.shape[0]):
        sumexp_ += expl(X[i] - Xmax)

    return logl(sumexp_) + Xmax

# log(exp(a) + exp(b))
cdef inline double c_logaddexp(double a, double b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))

cdef inline c_maxdivisor(int n):
    cdef int i
    cdef int max_val = int(ceil(sqrt(n)))

    for i in range(2, max_val):
        if n%i==0:
            return n//i
    return 1
