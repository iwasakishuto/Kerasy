# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.stdio cimport printf
from ..utils import flush_progress_bar

def forward(np.ndarray[floating, ndim=1, mode='c'] R, int limit, verbose=1):
    """ Forward Algorithm for Maxsets Dynamic Programming.
    @params R     : Real Numbers Sequence. shape=(n_sequence)
    @params limit : (int) Minimum n_sequence of Segment Sets.
    @var S        : Score. shape=(2,n_sequence+1)
    @var T        : TraceBack Pointer. shape=(2*(n_sequence+1))
    """
    dtype = np.float32 if floating is float else np.float64

    cdef int n_sequence = len(R)
    cdef int n_memory = n_sequence+1
    cdef np.ndarray[floating, ndim=2, mode='c'] S = np.zeros(shape=(2,n_memory), dtype=dtype)
    cdef np.ndarray[int, ndim=1] T = np.zeros(shape=(2*n_memory), dtype=np.int32)

    # Reccursion
    cdef floating s0
    cdef Py_ssize_t idx
    for n in range(limit):
        S[1,n] = -np.inf
        flush_progress_bar(n, n_memory, barname="forward DP", metrics={"Maximum segment score": 0}, verbose=verbose)

    for n in range(limit, n_memory):
        # Sn^0
        S[0,n] = max(S[0,n-1], S[1,n-1])
        if S[0,n] == S[0,n-1]:
            T[n] = n-1
        else:
            T[n] = n_memory+(n-1)

        # Sn^L
        s0 = S[0,n-limit]+np.sum(R[n-limit:n])
        S[1,n] = max(s0, S[1,n-1]+R[n-1])
        if S[1,n] == s0:
            T[n_memory+n] = n-limit
        else:
            T[n_memory+n] = n_memory+(n-1)

        flush_progress_bar(n, n_memory, barname="forward DP", metrics={"Maximum segment score": max(S[:,n])}, verbose=verbose)

    if verbose>0: printf("\n")
    return S,T


def traceback(np.ndarray[int, ndim=1, mode='c'] T, Py_ssize_t tp, verbose=1):
    """ Forward Algorithm for Maxsets Dynamic Programming.
    @params T  : TraceBack Pointer. shape=(2*(n_sequence+1))
    @params tp : (int) Initial traceback pointer
    """
    cdef n_memory = len(T)//2
    cdef np.ndarray[int, ndim=1] isin_sets = np.zeros(shape=(n_memory-1), dtype=np.int32)
    cdef Py_ssize_t new_tp, new_idx, idx=tp%n_memory-1
    cdef int new_isL, isL=tp>=n_memory

    if tp>=n_memory: isin_sets[idx] = 1
    flush_progress_bar(0, n_memory, barname="traceback", verbose=verbose)

    while idx!=-1:
        new_tp  = T[tp]
        new_idx = new_tp%n_memory-1
        new_isL = new_tp>=n_memory

        if (not new_isL) and isL:
            # When Sn = ^0_{new_idx} and SL^{idx} = S^0_{new_idx} + sum(r[])
            isin_sets[new_idx+1:idx] = 1
        elif new_isL:
            # When Sn = S^Ln
            isin_sets[new_idx] = 1

        tp  = new_tp
        idx = new_idx
        isL = new_isL

        flush_progress_bar(n_memory-(idx+1), n_memory, barname="traceback", verbose=verbose)

    if verbose>0: printf("\n")
    return isin_sets
