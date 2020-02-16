# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport sqrt
from libc.stdio cimport printf

def conv2D_forward(np.ndarray[floating, ndim=3, mode='c'] Xin,
                   np.ndarray[floating, ndim=3, mode='c'] Xout,
                   np.ndarray[floating, ndim=4, mode='c'] kernel,
                   int kh, kw, sh, sw):
    cdef Py_ssize_t OH = Xout.shape[0]
    cdef Py_ssize_t OW = Xout.shape[1]
    cdef Py_ssize_t i,j

    for i in range(OH//sh):
        for j in range(OW//sw):
            Xout[i,j,:] = np.sum(Xin[sh*i:(sh*i+kh), sw*j:(sw*j+kw), :, None] * kernel, axis=(0,1,2))
