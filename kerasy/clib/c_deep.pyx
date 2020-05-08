# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport sqrt
from libc.stdio cimport printf

ctypedef np.float64_t DOUBLE

cdef int _backprop_mask(int i,j,m,n,sh,sw,kh,kw,OH,OW):
    # Kernel IndexError Handler & Backprop IndexError Handler
    return (i%sh+m<kh and j%sw+n<kw) and ((OH > (i-m)//sh >= 0) and (OW > (j-n)//sw >= 0))


def Conv2D_forward(
    np.ndarray[DOUBLE, ndim=3, mode='c'] a,
    np.ndarray[DOUBLE, ndim=3, mode='c'] Xin,
    np.ndarray[DOUBLE, ndim=4, mode='c'] kernel,
    int OH, OW, sh, sw, kh, kw):

    cdef int i,j
    for i in range(OH//sh):
        for j in range(OW//sw):
            a[i,j,:] = np.sum(Xin[sh*i:(sh*i+kh), sw*j:(sw*j+kw), :, None]*kernel, axis=(0,1,2))

def Conv2D_backprop(
    np.ndarray[DOUBLE, ndim=3, mode='c'] dEda,
    np.ndarray[DOUBLE, ndim=3, mode='c'] dEdXin,
    np.ndarray[DOUBLE, ndim=4, mode='c'] dEdw,
    np.ndarray[DOUBLE, ndim=3, mode='c'] Xin,
    np.ndarray[DOUBLE, ndim=4, mode='c'] kernel,
    int padH, padW, F, OH, OW, OF, sh, sw, kh, kw, trainable=1):

    cdef int i,j,c
    for c in range(F):
        for i in range(padH):
            for j in range(padW):
                dEdXin[i,j,c] = np.mean([
                    dEda[(i-m)//sh, (j-n)//sw, :] * kernel[i%sh+m, j%sw+n, c, :] \
                    for m in range(0,kh,sh) \
                    for n in range(0,kw,sw) \
                    if _backprop_mask(i,j,m,n,sh,sw,kh,kw,OH,OW)
                ])

    if trainable:
        for m in range(kh):
            for n in range(kw):
                for c in range(F):
                    for c_ in range(OF):
                        dEdw[m,n,c,c_] = np.sum(dEda[:,:,c_] * Xin[m:m+OH:sh, n:n+OW:sw, c])

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

# def conv2D_forward_nogil(
#         double[:,:,:] Xin,
#         double[:,:,:] Xout,
#         double[:,:,:,:] kernel,
#         int OH, OW, OF, F, kh, kw, sh, sw, SH):
#     """ Convolution.
#     @params Xin    : Input Image.  shape=(H,W,F)
#     @params Xout   : Output Image. shape=(OH,OW,OF)
#     @params kernel : Kernel.       shape=(kh,kw,F,OF)
#     """
#     cdef int i, j, k, n, m, c, offset_h, offset_w
#     cdef double buff
#
#     with nogil:
#         for i in range(OH//sh):
#             offset_h = sh*i
#             for j in range(OW//sw):
#                 offset_w = sw*j
#                 for k in range(OF):
#                     buff = 0.
#                     for n in range(kh):
#                         for m in range(kw):
#                             for c in range(F):
#                                 buff += Xin[offset_h+n, offset_h+m, c]
#                     Xout[i,j,k] = buff
