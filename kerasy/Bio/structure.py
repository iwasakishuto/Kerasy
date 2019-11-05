# coding: utf-8
from __future__ import absolute_import
from ..utils.bio import BaseHandler

import numpy as np

class Nussinov(BaseHandler):
    __hidden_params__ = ["gamma", "omega","Z"]
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

    def predict(self, sequence, memorize=True, traceback=True):
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
        pairs = list("." * N)
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
            self.predict(sequence, memorize=True, traceback=False)
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


class Zuker(BaseHandler):
    __name__ = "Zuker Algorithm"
    __initialize_method__ = ['list2np', '_sort_stacking_cols', '_padding_stacking_score', 'replaceINF']
    __np_params__     = ["hairpin", "internal", "buldge", "stacking_score"]
    __inf_params__    = ["hairpin", "internal", "buldge"]
    __hidden_params__ = ["hairpin", "internal", "buldge", "stacking_cols", "stacking_score", "V", "M", "W", "sequence", "bp"]

    def __init__(self):
        #=== parameter ===
        self.a = None
        self.b = None
        self.c = None
        self.inf = None
        self.hairpin = None
        self.internal = None
        self.buldge = None
        self.stacking_cols = None
        self.stacking_score = None
        #=== Base Pair Type ===
        self.type=None
        self.Watson_Crick=None
        self.Wobble=None
        # Momory
        self.sequence = None
        self.W  = None # the minimum free energy of subsequence from i to j.
        self.V  = None # the minimum free energy of subsequence from i to j when i to j forms a base-pair.
        self.M  = None # the minimum free energy of subsequence when from i to j are in a multiloop and contain more than one base pairs which close it.
        # TraceBack.
        self.bp = []

    def _sort_stacking_cols(self):
        self.stacking_cols = np.array(["".join(sorted(bp)) for bp in self.stacking_cols])

    def _padding_stacking_score(self):
        N = len(self.stacking_score)
        tmp = np.full(shape=(N+1,N+1), fill_value=self.inf)
        tmp[:N,:N] = self.stacking_score
        self.stacking_score = tmp

    def _calW(self,i,j):
        return min(
            self.W[i+1][j],
            self.W[i][j-1],
            self.V[i][j],
            min([self.W[i][k] + self.W[k+1][j] for k in range(i,j)])
        )

    def _calV(self,i,j):
        return min(
            self._F1(i,j),
            min([self.inf]+[self._F2(i,j,h,l)+self.V[h][l] for h in range(i+1,j-1) for l in range(h+1,j) if self._is_bp(self.sequence[h],self.sequence[l])]),
            min([self.inf]+[self.M[i+1][k] + self.M[k+1][j-1] for k in range(i+1,j-1)]) + self.a + self.b,
        ) if self._is_bp(self.sequence[i],self.sequence[j]) else self.inf

    def _calM(self,i,j):
        return min(
            self.V[i][j] + self.b,
            self.M[i+1][j] + self.c,
            self.M[i][j-1] + self.c,
            min([self.inf]+[self.M[i][k] + self.M[k+1][j] for k in range(i,j)])
        )

    def _F1(self,i,j):
        """ i...j → length=3 → nt=2 (0-origin) """
        nt=(j-i-1)-1 # 0-origin.
        return self.hairpin[nt] if nt<30 else self.inf

    def _F2(self,i,j,h,l):
        """ i..h - l...j → length=2+3=5 → nt=4 """
        nt = (h-i-1)+(j-l-1)-1 # 0-origin.
        if nt>=30: val = self.inf
        elif (h==i+1) and (l==j-1):
            """ stacking """
            rcond = self.stacking_cols=="".join(sorted(self.sequence[i]  +self.sequence[j]))
            ccond = self.stacking_cols=="".join(sorted(self.sequence[i+1]+self.sequence[j-1]))
            ridx = np.argmax(rcond) if np.any(rcond) else -1
            cidx = np.argmax(ccond) if np.any(ccond) else -1
            val = self.stacking_score[ridx][cidx]
        elif (i+1<h<l<j-1):
            """ internal loop """
            val = self.internal[nt]
        else:
            """ bulge loop """
            val = self.buldge[nt]

        return val

    def predict(self,sequence):
        """ calcurate DP matrix. (Recursion) """
        N=len(sequence)
        self.sequence = sequence
        self.V  = np.full(shape=(N,N), fill_value=self.inf, dtype=float)
        self.M  = np.full(shape=(N,N), fill_value=self.inf, dtype=float)
        self.W  = np.full(shape=(N,N), fill_value=0, dtype=float)
        for ini_i in reversed(range(N)):
            diff = N-ini_i
            for i in reversed(range(ini_i)):
                j = i+diff
                self.V[i][j]  = self._calV(i,j)
                self.M[i][j]  = self._calM(i,j)
                self.W[i][j]  = self._calW(i,j)

        score = self.W[0][-1]
        pairs = self.traceback()
        self._printAlignment(score, self.sequence, pairs, width=60, xlabel="seq", ylabel="")

    def tracebackV(self, i, j):
        self.bp.append((i,j))
        if self.V[i][j] == self._F1(i,j): pass
        else:
            for h in range(i+1,j-1):
                for l in range(h+1,j):
                    if self._is_bp(self.sequence[h],self.sequence[l]):
                        if self.V[i][j] == self._F2(i,j,h,l) + self.V[h][l]:
                            self.tracebackV(h,l)
        for k in range(i+1,j-1):
            if self.V[i][j] == self.M[i+1][k] + self.M[k+1][j-1] + self.a+self.b:
                self.tracebackM(i+1,k)
                self.tracebackM(k+1,j-1)

    def tracebackM(self, i, j):
        if self.M[i][j] == self.V[i][j] + self.b:
            self.tracebackV(i,j)
        elif self.M[i][j] == self.M[i+1][j] + self.c: self.tracebackM(i+1,j)
        elif self.M[i][j] == self.M[i][j-1] + self.c: self.tracebackM(i,j-1)
        else:
            for k in range(i,j):
                if self.M[i][j] == self.M[i][k] + self.M[k+1][j]:
                    self.tracebackM(i,k)
                    self.tracebackM(k+1,j)

    def traceback(self):
        """ Main trackback to find which bases form base-pairs. """
        N = len(self.sequence)
        self.bp = []
        stack = [(0,N-1)]
        while(stack):
            i,j = stack.pop()
            if (i>=j): continue
            elif self.W[i][j] == self.W[i+1][j]: stack.append((i+1,j))
            elif self.W[i][j] == self.W[i][j-1]: stack.append((i,j-1))
            elif self.W[i][j] == self.V[i][j]: self.tracebackV(i,j)
            else:
                for k in range(i,j):
                    if self.W[i][j] == self.W[i][k]+self.W[k+1][j]:
                        stack.append((k+1,j))
                        stack.append((i,k))
                        break

        pairs = list("." * N)
        for c,(i,j) in enumerate(self.bp):
            pairs[i] = "("; pairs[j] = ")"
        return "".join(pairs)
