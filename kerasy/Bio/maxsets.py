# coding: utf-8
import numpy as np
import time

from ..utils import printAlignment
from ..utils import flush_progress_bar

class MSS():
    """ Maximum Segment Sum
    ~~~~~ Vars.
    - Score (shape=(2,N+1))
        - S[0,:] memorizes S^0.
        - S[1,:] memorizes S^L.
        - S[:,0] memorizes dummy variables (Initialization)
        - S[:,i+1] memorizes the maximum score from r[0] to r[i]
    - TraceBack (shape=(L+1,N)): MaxSegmentSet includes r[i] if T[i]==1 else doesn't include.
        - T[k] memorizes the set of S^0(n-L+k).
        - T[L] memorizes the set of S^Ln-1.
    """
    def __init__(self):
        self.score = None # shape=(2,N) Memorize data shape.
        self.isin_sets = None # shape=(2,N) Memorize data shape.

    @staticmethod
    def maxargmax(s0,sL):
        """ Finding maximum value from numbers including inf """
        return (s0,0) if s0>=sL else (sL,1)

    def run(self,r,L,display=True, verbose=1, width=60):
        """
        @params r: Real numbers. 1-dimentional score sequence
        @params L: Minimum segment length. (Constraint)
        @return score:
        """
        r = np.asarray(r)
        N = len(r)

        if L==1:
            # There is no problem with passing the following function, but this is faster.
            isin_sets = np.where(r>=0, 1, 0)
            score = sum(np.where(np.array(r)>0,r,0))
        else:
            # Initialization
            S = np.zeros(shape=(2,N+1), dtype=float) # Score.
            S[1][0] = -np.inf
            T = np.zeros(shape=(L+1,N), dtype=int) # TraceBack.

            # Reccursion
            for n in range(N):
                tmpT = T[1:-1]
                """ Sn^0 """
                score,idx = self.maxargmax(*S[:,n])
                S[0,n+1]  = score
                updateS0T = T[(L-1)+idx,:]
                """ Sn^L """
                score,idx = self.maxargmax(S[-1,n], S[0,n-L+1]+np.sum(r[n-L+1:n]) if n-L+1>=0 else -np.inf)
                S[1,n+1]  = score + r[n]
                updateSLT = np.append(T[0,:n-L+1],[1 for _ in range(L)]+[0 for _ in range(N-n-1)]) if idx else T[-1,:]
                """ Update Trace Back """
                T = np.r_[
                    tmpT,
                    updateS0T.reshape(1,-1),
                    updateSLT.reshape(1,-1)
                ]
                T[-1,n] = 1
                flush_progress_bar(n,N,barname="DP",metrics={"Maximum segment score": max(S[:,n+1])},verbose=verbose)
            if verbose>=1: print()
            score,idx = self.maxargmax(*S[:,-1])
            isin_sets = T[-2+idx,:]
            # Memorize.
            self.score = S[:,1:]
            self.isin_sets = T[-2:,:]

        if display:
            printAlignment(
                sequences=["".join(np.where(isin_sets==1, "-", "*"))],
                indexes=[np.arange(N)],
                score=score,
                seqname=["S"],
                width=width,
                model=self.__class__.__name__
            )
        else:
            return score
