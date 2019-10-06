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


# class Zuker(BaseHandler):
#     __remove_params__ = ["gamma", "omega","Z"]
#     __name__ = "Zuker Algorithm"
#
#     def __init__(self):
#         self.type=None
#         self.Watson_Crick=None
#         self.Wobble=None
#         self.hairpin=None
#         self.stacking=None
#         self.internal_loop=None
#         self.buldge_loop
#         self.gamma=None
#         self.omega=None
#         self.Z=None
#
#     def _calW(self,i,j): return min(self.W[i+1][j],self.W[i][j-1],self.V[i][j],min([self.W[i][k] + self.W[k+1][j] for k in range(i,j)]))
#     def _calV(self,i,j,DNA,Wobble): return min(self._F1(i,j),min([self.inf]+[self._F2(i,j,h,l,DNA,Wobble)+self.V[h][l] for h in range(i+1,j) for l in range(h+1,j) if self._is_bp(h,l,DNA=DNA,Wobble=Wobble)<2]),self.M[i+1][j-1] + self.a + self.b) if self._is_bp(i,j,DNA=DNA,Wobble=Wobble)<2 else self.inf
#     def _calM(self,i,j): return min([self.M1[i][k] + self.M1[k+1][j] for k in range(i,j)])
#     def _calM1(self,i,j): return min(self.M[i][j], self.V[i][j]+self.b, self.M1[i+1][j]+self.c, self.M1[i][j-1]+self.c)
#
#     def _F1(self,i,j):
#         nt=j-i-1
#         return self.hairpin[nt] if nt<30 else self.inf
#     def _F2(self,i,j,h,l,DNA,Wobble):
#         nt = (h-i-1)+(j-l-1)
#         if nt>=30: val = self.inf
#         elif (h==i+1) and (l==j-1): val = self.stacking[self._is_bp(i,j,DNA=DNA,Wobble=Wobble)][self._is_bp(i+1,j-1,DNA=DNA,Wobble=Wobble)]
#         elif (i+1<h<l<j-1): val = self.internal_loop[nt]
#         else: val = self.buldge_loop[nt]
#
#         return val
#
#     def predict(self,sequence,):
#         """calcurate DP matrix.(recursion)"""
#         N=len(sequence)
#         self.V =
#         for ini_i in reversed(range(self.N)):
#             diff = self.N-ini_i
#             for i in reversed(range(ini_i)):
#                 j = i+diff
#                 self.V[i][j] = self._calV(i,j,DNA=DNA,Wobble=Wobble)
#                 self.M[i][j] = self._calM(i,j)
#                 self.M1[i][j] = self._calM1(i,j)
#                 self.W[i][j] = self._calW(i,j)
#
#
