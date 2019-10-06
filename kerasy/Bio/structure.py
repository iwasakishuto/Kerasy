# coding: utf-8
from __future__ import absolute_import
from ..utils.params import Params

import numpy as np

class BaseHandler(Params):
    __name__ = ""

    def _printAlignment(self, score, seq, pairs, width=60):
        print(f"\033[31m\033[07m {self.__name__} \033[0m\nScore: \033[34m{score}\033[0m\n")
        print("="*(width+3))
        print("\n\n".join([f"seq: {seq[i: i+width]}\n     {pairs[i: i+width]}" for i in range(0, len(seq), width)]))
        print("="*(width+3))

class Nussinov(BaseHandler):
    __remove_params__ = ["gamma"]
    __name__ = "Nussinov Algorithm"

    def __init__(self):
        self.type=None
        self.Watson_Crick=None
        self.Wobble=None

    def _is_bp(self,si,sj):
        """check if i to j forms a base-pair."""
        bases = si+sj
        flag = False
        if self.Watson_Crick:
            WC_1 = ("A" in bases) and ("T" in bases) if self.type=="DNA" else ("A" in bases) and ("U" in bases)
            WC_2 = ("G" in bases) and ("C" in bases)
            flag = flag or (WC_1 or WC_2)
        if self.Wobble:
            Wob  = ("G" in bases) and ("U" in bases)
            flag = flag or Wob
        return flag

    def fit(self, sequence, memorize=False):
        N = len(sequence)
        gamma = np.zeros(shape=(N,N),dtype=int)
        for ini_i in reversed(range(N)):
            diff = N-ini_i
            for i in reversed(range(ini_i)):
                j = i+diff
                delta = 1 if self._is_bp(sequence[i],sequence[j]) and (j-i>3) else 0
                gamma[i][j] = max(
                    gamma[i+1][j],
                    gamma[i][j-1],
                    gamma[i+1][j-1] + delta,
                    max([gamma[i][k] + gamma[k+1][j] for k in range(i,j)]) # Bifurcation
                )
        if memorize: self.gamma = gamma
        # TraceBack
        pairs = self.traceback(gamma, N, sequence)
        self._printAlignment(gamma[0][-1], sequence, pairs, width=60)

    def traceback(self,gamma,N,sequence):
        """trackback to find which bases form base-pairs."""
        bp = []
        stack = [(0,N-1)]
        while(stack):
            i,j = stack.pop(0)
            delta = 1 if self._is_bp(sequence[i],sequence[j]) and (j-i>3) else 0

            if (i>=j): continue
            elif gamma[i+1][j] == gamma[i][j]: stack.append((i+1,j))
            elif gamma[i][j-1] == gamma[i][j]: stack.append((i,j-1))
            elif gamma[i+1][j-1]+delta == gamma[i][j]: bp.append((i,j)); stack.append((i+1,j-1))
            else:
                # Bifurcation
                for k in range(i,j):
                    if gamma[i][k] + gamma[k+1][j] == gamma[i][j]:
                        stack.append((k+1,j))
                        stack.append((i,k))
                        break
        pairs = list(" " * N)
        for i,j in bp:
            pairs[i] = "("; pairs[j] = ")"
        return "".join(pairs)
