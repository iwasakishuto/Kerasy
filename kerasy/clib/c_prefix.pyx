# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np

def LP_create(str string, np.ndarray[np.int32_t, ndim=1] z, int n):
    """ Ref: https://codeforces.com/blog/entry/3107
    This algorithm runs in O(n) time.
    Given a string S of length n, the Z Algorithm produces an array Z where
    Z[i] is the length of the longest substring starting from S[i]
    which is also a prefix of S (i.e. the maximum k such that S[j]=S[i+j] for all 0<=j<k)
    NOTE: Z[i]=0 means that S[0]≠S[i].
    ~~~
    @params string : (str) Input string.
    @params z      : (ndarray) buffer. (np.zeros(shape=(n), dtype=np.int32))
    @params n      : (int) The length of string (also z).
    """
    cdef int L,R,k
    cdef Py_ssize_t is
    L = R = 0
    # Now suppose we have the correct interval [L, R] for i-1 and all of the Z values up to i-1.
    # We will compute Z[i] and the new [L, R] by the following steps:
    for i in range(1,n):
        # If i>R, then there does not exist a prefix-substring of S that starts
        # before i and ends at or after i. If such a substring existed,
        # [L, R] would have been the interval for that substring rather than its
        # current value. Thus we "reset" and compute a new [L, R] by
        # comparing S[0...] to S[i...] and get Z[i] at the same time (Z[i]=R-L+1).
        if i>R:
            L = R = i
            while (R<n and string[R-L]==string[R]): R+=1
            z[i] = R-L; R-=1;
        # Otherwise, i≤R, so the current [L, R] extends at least to i. Let k=i-L.
        # We know that Z[i] ≥ min(Z[k], R-i+1) because S[i...] matches S[k...]
        # for at least R-i+1 characters (they are in the [L, R] interval which we
        # know to be a prefix-substring). Now we have a few more cases to consider.
        else:
            k = i-L
            # If Z[k] < R-i+1, then there is no longer prefix-substring starting
            # at S[i] (or else Z[k] would be larger), meaning Z[i]=Z[k] and [L, R]
            # stays the same. The latter is true because [L, R] only changes if
            # there is a prefix-substring starting at S[i] that extends beyond R,
            # which we know is not the case here.
            if (z[k]<R-i+1):
                z[i] = z[k]
            # If Z[k] ≥ R-i+1, then it is possible for S[i...] to match S[0...]
            # for more than R - i + 1 characters (i.e. past position R).
            # Thus we need to update [L, R] by setting L=i and matching from S[R+1]
            # forward to obtain the new R. Again, we get Z[i] during this.
            else:
                L = i;
                while (R<n and string[R-L]==string[R]): R+=1
                z[i] = R-L; R-=1
