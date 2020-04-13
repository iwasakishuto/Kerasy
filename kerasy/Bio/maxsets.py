# coding: utf-8
import numpy as np
import time

from ..utils import printAlignment
from ..utils import flush_progress_bar

from ..clib import c_maxsets
from ..clib._pyutils import extract_continuous_area

class MSS():
    """ Maximum Segment Sum """
    def __init__(self):
        self.score = None # shape=(2,N) Memorize data shape.
        self.isin_sets = None # shape=(2,N) Memorize data shape.

    @staticmethod
    def max_argmax(s0, sL, offset=0, n_memory=1, limit=1):
        """ Return Score and TraceBack Pointer Simultaneously
        @params s0,sL    : score of S^0_n and S^L_n.
        @params n        : index.
        @params limit    : minimum length of segment sets.
        @params n_memory : The length of memory.
        """
        if s0>=sL:
            # if we think about s0, we sets limit=1.
            return s0,offset+1-limit
        else:
            return sL,offset+n_memory

    def run(self,R,limit,display=True,verbose=1,width=60):
        try:
            from ..clib import c_maxsets
        except ImportError:
            print("There is no c-extensions, so we run the `MSS.run`, which runs only with python")
            self.run(R, limit, display=display, verbose=verbose, width=width)

        R = np.asarray(R, dtype=float)
        n_sequence = len(R)
        n_memory   = n_sequence+1
        if limit==1:
            # There is no problem with passing the following function, but this is faster.
            isin_sets = np.where(R>=0, 1, 0)
            score = np.sum(np.where(R>=0, R, 0))
        else:
            S, T = c_maxsets.forward(R, limit, verbose=verbose)
            score, tp_init = self.max_argmax(*S[:,-1], offset=n_sequence, n_memory=n_memory)
            isin_sets = c_maxsets.traceback(T, tp_init)

        self.score = score
        self.isin_sets = isin_sets
        self.detected_area = np.asarray(extract_continuous_area(isin_sets), dtype=np.int32)

    def display(self, isin_sets=None, score=None, width=60):
        if isin_sets is None:
            isin_sets = self.isin_sets
            score = self.score
        printAlignment(
            sequences=["".join(np.where(isin_sets==1, "*", "-"))],
            indexes=[np.arange(len(isin_sets))],
            score=score,
            add_info="('*' means in sets)",
            scorename="Maximum Sets Score",
            seqname=["R"],
            model=self.__class__.__name__,
            width=width,
        )

    # def run(self,R,limit,display=True,verbose=1,width=60):
    #     """
    #     @params r: Real numbers. 1-dimentional score sequence
    #     @params L: Minimum segment length. (Constraint)
    #     @return score:
    #     """
    #     if verbose>0: print("preparing for DP...")
    #     R = np.asarray(R, dtype=float)
    #     n_sequence = len(R)
    #     n_memory   = n_sequence+1
    #     if limit==1:
    #         # There is no problem with passing the following function, but this is faster.
    #         isin_sets = np.where(R>=0, 1, 0)
    #         score = np.sum(np.where(R>=0, R, 0))
    #     else:
    #         S, T = self.forward(R, limit, verbose=verbose)
    #         score,tp_init = self.max_argmax(*S[:,-1], offset=n_sequence, n_memory=n_memory)
    #         isin_sets = self.TraceBack(T, tp_init)
    #
    #     if verbose>0: print("memorize the score as `self.score`...")
    #     self.score = score
    #     if verbose>0: print("memorize the maximum segmet sets as `self.isin_sets`...")
    #     self.isin_sets = isin_sets
    #
    #     if display:
    #         self.display(isin_sets=isin_sets, score=score, width=width)
    #     else:
    #         return score
    #
    # def forward(self, R, limit, verbose=1):
    #     n_sequence = len(R)
    #     n_memory   = n_sequence+1
    #     S = np.zeros(shape=(2,n_memory), dtype=np.float)
    #     T = np.zeros(shape=(2*n_memory), dtype=np.int32)
    #     # Reccursion
    #     # (while n<=limit)
    #     for n in range(limit):
    #         S[1,n] = -np.inf
    #         flush_progress_bar(n, n_memory, barname="forward DP", metrics={"Maximum segment score": 0}, verbose=verbose)
    #     # (while n>limit, when we think about segment sets.)
    #     for n in range(limit, n_memory):
    #         S[0,n], T[n]          = self.max_argmax(S[0,n-1], S[1,n-1], offset=n-1, n_memory=n_memory, limit=1)
    #         S[1,n], T[n+n_memory] = self.max_argmax(S[0,n-limit]+np.sum(R[n-limit:n]), S[1,n-1]+R[n-1], offset=n-1, n_memory=n_memory, limit=limit)
    #         flush_progress_bar(n, n_memory, barname="forward DP", metrics={"Maximum segment score": max(S[:,n])}, verbose=verbose)
    #     if verbose>=1: print()
    #     return S,T
    #
    # def TraceBack(self, T, tp):
    #     n_memory  = len(T)//2
    #     isin_sets = np.zeros(n_memory-1)
    #
    #     idx = tp%n_memory-1
    #     isL = tp>=n_memory
    #     if tp>=n_memory: isin_sets[idx] = 1
    #
    #     while idx!=-1:
    #         new_tp  = T[tp]
    #         new_idx = new_tp%n_memory-1
    #         new_isL = new_tp>=n_memory
    #
    #         if (not new_isL) and isL:
    #             # When Sn = ^0_{new_idx} and SL^{idx} = S^0_{new_idx} + sum(r[])
    #             isin_sets[new_idx+1:idx] = 1
    #         elif new_isL:
    #             # When Sn = S^Ln
    #             isin_sets[new_idx] = 1
    #
    #         tp  = new_tp
    #         idx = new_idx
    #         isL = new_isL
    #     return isin_sets
