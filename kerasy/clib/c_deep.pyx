# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport sqrt
from libc.stdio cimport printf

def conv2D_forward_gil(
        np.ndarray[floating, ndim=3, mode='c'] Xin,
        np.ndarray[floating, ndim=3, mode='c'] Xout,
        np.ndarray[floating, ndim=4, mode='c'] kernel,
        int kh, kw, sh, sw):
    cdef Py_ssize_t OH = Xout.shape[0]
    cdef Py_ssize_t OW = Xout.shape[1]
    cdef Py_ssize_t i,j

    for i in range(OH//sh):
        for j in range(OW//sw):
            Xout[i,j,:] = np.sum(Xin[sh*i:(sh*i+kh), sw*j:(sw*j+kw), :, None] * kernel, axis=(0,1,2))

def conv2D_forward_nogil(
        double[:,:,:] Xin,
        double[:,:,:] Xout,
        double[:,:,:,:] kernel,
        int OH, OW, OF, F, kh, kw, sh, sw, SH):
    """ Convolution.
    @params Xin    : Input Image.  shape=(H,W,F)
    @params Xout   : Output Image. shape=(OH,OW,OF)
    @params kernel : Kernel.       shape=(kh,kw,F,OF)
    """
    cdef int i, j, k, n, m, c, offset_h, offset_w
    cdef double buff

    with nogil:
        for i in range(OH//sh):
            offset_h = sh*i
            for j in range(OW//sw):
                offset_w = sw*j
                for k in range(OF):
                    buff = 0.
                    for n in range(kh):
                        for m in range(kw):
                            for c in range(F):
                                buff += Xin[offset_h+n, offset_h+m, c]
                    Xout[i,j,k] = buff
