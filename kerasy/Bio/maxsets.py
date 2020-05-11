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
            isin_sets = c_maxsets.traceback(T, tp_init, verbose=verbose).astype(np.int32)

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
