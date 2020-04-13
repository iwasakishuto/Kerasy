# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport sqrt
from libc.stdio cimport printf
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from scipy.linalg.cython_blas cimport sdot, ddot

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

cdef floating _dot(int n, floating *x, int incx, floating *y, int incy) nogil:
    """ x.T.y """
    if floating is float:
        return sdot(&n, x, &incx, y, &incy)
    else:
        return ddot(&n, x, &incx, y, &incy)

cdef floating euclidean_distance(floating* x, floating* y, int n_features) nogil:
    """
    Compute the euclidean distance between `x` and `y`. (Each data has the `n_features` features.)
    As the function start with `nogil`, this function will be executed without GIL.
    """
    cdef floating result, diff
    result = 0
    cdef int i
    for i in range(n_features):
        diff = x[i]-y[i]
        result += diff * diff
    return sqrt(result)

def paired_euclidean_distances(
        np.ndarray[floating, ndim=2, mode='c'] X,
        np.ndarray[floating, ndim=2, mode='c'] Y):
    """ Compute the paired euclidean distances between X and Y.
    @params X,Y      : shape=(n_samples, n_features) should be the same.
    @return distances: shape=(n_samples)
    """
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef floating* X_p = <floating*>X.data
    cdef floating* Y_p = <floating*>Y.data
    cdef Py_ssize_t idx
    distances = np.empty(n_samples, dtype=dtype)

    for idx in range(n_samples):
        distances[idx] = euclidean_distance(X_p+idx*n_features, Y_p+idx*n_features, n_features)
    return distances

def extract_continuous_area(np.ndarray[INT, ndim=1, mode='c'] bool_arr):
    """
    ex.)
    @params bool_arr = [T, F,F,F, T,T,T, F,F,F, T,T]
    @return start_and_length = [[0,1],[4,3],[10,2]]
    """
    cdef pair[int, int] p
    cdef vector[pair[int, int]] start_and_length

    cdef int len = 0
    cdef int val
    for i,val in enumerate(bool_arr):
        if len>0:
            if val:
                len+=1
            else:
                p.second = len
                start_and_length.push_back(p)
                len = 0
        elif val:
            len = 1
            p.first = i

    return start_and_length
