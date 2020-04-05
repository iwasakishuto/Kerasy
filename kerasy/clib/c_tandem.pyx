# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

cimport cython
from cython cimport floating
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from .c_prefix import LP_create
from .c_prefix import lptext_create

# Cut out the exact tandem area, and calculate the score.
cdef inline _calc_tandem_score(int i, j, score):
    """
    Cut out the exact tandem area.
    ex)
    i             j<-- score=14+4 ->
    [tandem block][tandem block][tan
                  [tandem block][tan
    --> 14*2 = 28
    ~~~
    @params i,j          : The position of i,j at the
    @params score        : Score in DP matrix.
    @return tandem score : score//len_tandem*len_tandem
    """
    # No overlap
    if i+score<j:
        return 0
    cdef int len_tandem = j-i
    return (score//len_tandem+1)*len_tandem

# Dynamic Programming for finding tandem repeats.
def _DPrecursion_Tandem(np.ndarray[np.int32_t, ndim=2] DPmatrix, str X):
    """ Dynamic Programming for finding tandem repeats.
    @params DPmatrix : shape=(n+1,n+1)
    @params X        : Sequence. len(X) = n
    @return max_val          : maximum tandem score.
    @return tandem_pos_lists : store start, and end point of tandem.
    """
    cdef int max_val, score, len_tandem
    cdef int n = len(X)
    cdef pair[int, int] p
    cdef vector[pair[int, int]] tandem_pos_lists

    max_val = 0
    for i in range(n):
        for j in range(n):
            if i>=j:
                continue
            score_ = DPmatrix[i][j]
            if X[i]==X[j]:
                DPmatrix[i+1][j+1] = score_+1
            elif score_ > 0:
                score = _calc_tandem_score(i,j,score_)
                if score > max_val:
                    # Initialization
                    tandem_pos_lists.clear()
                    max_val = score
                    # Add pair.
                    p.first = i-score_
                    p.second = j-score_
                    tandem_pos_lists.push_back(p)
                elif score == max_val:
                    p.first = i-score_
                    p.second = j-score_
                    tandem_pos_lists.push_back(p)

    return max_val, tandem_pos_lists

def _SAIS_Tandem(list U):
    """
    @params U: LZ_factorization
    """
    cdef int k=len(U)
    cdef Py_ssize_t i
    cdef int m_i,n_i
    cdef str t_i,u_i,v_i
    cdef list tandems_right = []
    cdef list tandems_left = []

    for i in range(1, k):
        # O(|u_{i-1}|+|u_i|) for each iteration.
        t_i = "".join(U[:i])[-(2*len(U[i-1])+len(U[i])):]
        u_i = U[i]
        v_i = t_i+u_i

        m_i = len(t_i)
        n_i = len(u_i)
        # type1. rightmax
        _find_type1(t_i,u_i,m_i,n_i,tandems_right)
        # type1. rightmax
        _find_type1(u_i[::-1],t_i[::-1],n_i,m_i,tandems_left)
    tandems_left = [tandem[::-1] for tandem in tandems_left]
    return tandems_left+tandems_right

cdef _find_type1(str t, str u, int m, int n, list tandems):
    cdef np.ndarray[np.int32_t, ndim=1] LP, lppattern, LS
    cdef str v = t+u
    cdef int j
    # Function LP is computed in time linear in |u|, O(n)
    # shape=(n+1) is same as "'shape=n' and 'np.append(LP, 0)'"
    LP=np.zeros(shape=(n+1), dtype=np.int32)
    LP_create(u, LP, n)

    # Function LS is computed in time linear in |v| O(n+m)
    lppattern=np.zeros(shape=m, dtype=np.int32)
    LP_create(t, lppattern, m)
    LS=np.zeros(shape=n, dtype=np.int32)
    lptext_create(
        pattern=t[::-1], lppattern=lppattern[::-1],
        text=v[::-1], lptext=LS, n=n
    )
    LS = LS[::-1]
    for j in range(1,n):
        if LS[j]+LP[j+1]>=j+1:
            # max_reps = v[m-LS[j]:m+(j+1)+LP[j+1]]
            # tandem = max_reps[:(j+1)]
            # tandems.append(tandem)
            tandems.append(v[m-LS[j]:m-LS[j]+(j+1)])

# # Log-scaled forward algorithm
# def _DPrecursion_NeedlemanWunshGotoh(
#     np.ndarray[floating, ndim=2, mode='c'] DPmatrix,
#     np.ndarray[floating, ndim=1, mode='c'] TraceBack,
#     str X, Y, floating gap_opening, gap_extension, match, mismatch):
#     """ Dynamic Programming for NeedlemanWunshGotoh.
#     @params DPmatrix      : shape=(nx,ny,3)
#     @params TraceBack     : The matrix which records traceback pointers
#     @params X,  Y         : Sequences. len(X)=nx-1, len(Y)=ny-1
#     @params nx, ny        : The length of each string.
#     @params gap_opening   :
#     @params gap_extension :
#     @params match         :
#     @params mismatch      :
#     """
#     for i in range(len_sequence):
#     for j in range(len_sequence):
#         if i>=j: continue
#         if sequence[i]==sequence[j]:
#             DP[i+1][j+1] = DP[i][j]+1
#
#     cdef int nx = DPmatrix.shape[0]
#     cdef int ny = DPmatrix.shape[1]
#     cdef floating score
#
#     for i in range(1,nx):
#         for j in range(1,ny):
#             DPmatrix[i-1,j-1,:]
#             score = match if X[i]==Y[j] else mismatch
