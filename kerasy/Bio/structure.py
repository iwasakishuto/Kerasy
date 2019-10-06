# coding: utf-8
from __future__ import absolute_import
from ..utils.bio import BaseHandler

import numpy as np

class Nussinov(BaseHandler):
    __remove_params__ = ["gamma", "omega","Z"]
    __name__ = "Nussinov Algorithm"

    def __init__(self):
        self.type=None
        self.Watson_Crick=None
        self.Wobble=None
        self.gamma=None
        self.omega=None
        self.Z=None

    def _printAsTerai(self, array, sequence, as_gamma=False):
        arr = array.astype(object)
        N,M = arr.shape
        digit = len(str(np.max(arr)))

        print("\ " +  " ".join(sequence))
        for i in range(N):
            cell = f"{sequence[i]} "
            for j in range(N):
                cell += f'{"-" if j+as_gamma<i else arr[i][j]:>{digit}} '
            print(cell)

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

    def fit(self, sequence, memorize=True, traceback=True):
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
        # TraceBack
        if traceback:
            pairs = self.traceback(gamma, N, sequence)
            score = gamma[0][-1]
            self._printAlignment(score, sequence, pairs, width=60, xlabel="seq", ylabel="")
        # Memorize or Return.
        if memorize: self.gamma=gamma
        else: self._printAsTerai(gamma, sequence, as_gamma=True)

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

    def outside(self, sequence, memorize=True):
        N = len(sequence)
        omega = np.zeros(shape=(N+1,N+1), dtype=int)
        # omega[i+1][j] and gamma[i][j] indicate same position (seq[i],seq[j])
        for ini_j in reversed(range(N-1)):
            for i in range(N-ini_j):
                j = ini_j+i
                delta = 1 if (i>0 and j+1<N) and self._is_bp(sequence[i-1],sequence[j+1]) and (j-i>3) else 0
                omega[i+1,j]=max(
                    omega[i][j],
                    omega[i+1][j+1],
                    omega[i][j+1] + delta,
                    max([0]+[omega[k+1][j] + self.gamma[k][i-1] for k in range(i)]),
                    max([0]+[self.gamma[j+1][k] + omega[i+1][k] for k in range(j+1,N)])
                )
        omega = omega[1:,:-1]
        if memorize: self.omega=omega
        else: self._printAsTerai(omega, sequence)

    def ConstrainedMaximize(self, sequence, gamma=None, omega=None, memorize=False):
        if gamma is None:
            self.fit(sequence, memorize=True, traceback=False)
            gamma = self.gamma
        if omega is None:
            self.outside(sequence)
            omega = self.omega

        N=len(sequence)
        Z = np.zeros(shape=(N,N),dtype=int)
        for ini_i in reversed(range(N)):
            diff = N-ini_i
            for i in reversed(range(ini_i)):
                j = i+diff
                Z[i][j] = gamma[i+1][j-1]+1+omega[i][j] if self._is_bp(sequence[i],sequence[j]) and (j-i>3) else 0
        if memorize:  self.Z=Z
        else: self._printAsTerai(Z, sequence)
