# coding: utf-8
import numpy as np
from scipy.special import logsumexp

from ..utils import Params
from ..utils import bpHandler
from ..utils import printAlignment
from ..utils import handleKeyError
from ..utils import flush_progress_bar

class BaseStructureModel(Params):
    def __init__(self, **kwargs):
        self.disp_params = ["inf", "nucleic_acid", "WatsonCrick", "Wobble"]

    def predict(self, sequence, width=60, only_score=False, display=True, verbose=1):
        N = len(sequence)
        self.Initialization()
        # Recursion
        score = self.DynamicProgramming(sequence, verbose)
        if only_score:
            return score

        # TraceBack
        structure_info = self.TraceBack(sequence)
        if display:
            printAlignment(
                sequences=[sequence, structure_info],
                indexes=[np.arange(len(sequence)), np.arange(len(sequence))],
                seqname=['X', ' '],
                score=score,
                width=width,
                model=self.__class__.__name__
            )
        else:
            return (score,structure_info)

    def Initialization(self):
        """ Initialize function like `is_bp`, `bp_ps`. """
        raise NotImplementedError()

    def DynamicProgramming(self, sequence, verbose=1):
        """ Recursion of Dynamic Programming. """
        raise NotImplementedError()

    def TraceBack(self, sequence):
        """ Trace Back. """
        raise NotImplementedError()

class Nussinov(BaseStructureModel):
    """ Nussinov Algorithm """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disp_params += ["minspan"]
        self.gamma = None
        self.omega = None
        self.Z = None

    def Initialization(self):
        self.is_bp = bpHandler(bp2id=None, nucleic_acid=self.nucleic_acid, WatsonCrick=self.WatsonCrick, Wobble=self.Wobble)

    def DynamicProgramming(self, sequence, verbose=1):
        """ Recursion of Dynamic Programming. """
        N = len(sequence)
        self.gamma = np.zeros(shape=(N, N),dtype=int)
        max_iter = sum([i for i in range(N)])
        it = 0
        for ini_i in reversed(range(N)):
            diff = N-ini_i
            for i in reversed(range(ini_i)):
                j = i+diff
                delta = 1 if (j-i>self.minspan) and self.is_bp(sequence[i],sequence[j]) else 0
                self.gamma[i][j] = max(
                    self.gamma[i+1][j],
                    self.gamma[i][j-1],
                    self.gamma[i+1][j-1] + delta,
                    max([self.gamma[i][k] + self.gamma[k+1][j] for k in range(i,j)]) # Bifurcation
                )
                flush_progress_bar(it, max_iter, barname="γ", metrics={"max num of base-pairs": np.max(self.gamma)}, verbose=verbose)
                it+=1
        print()
        score = self.gamma[0][-1]
        return score

    def TraceBack(self, sequence):
        """trackback to find which bases form base-pairs."""
        N = len(sequence)
        bp = []
        stack = [(0,N-1)]
        while(stack):
            i,j = stack.pop(0)
            delta = 1 if (j-i>self.minspan) and self.is_bp(sequence[i],sequence[j]) else 0
            if (i>=j): continue
            elif self.gamma[i+1][j] == self.gamma[i][j]: stack.append((i+1,j))
            elif self.gamma[i][j-1] == self.gamma[i][j]: stack.append((i,j-1))
            elif self.gamma[i+1][j-1]+delta == self.gamma[i][j]: bp.append((i,j)); stack.append((i+1,j-1))
            else:
                # Bifurcation
                for k in range(i,j):
                    if self.gamma[i][k] + self.gamma[k+1][j] == self.gamma[i][j]:
                        stack.append((k+1,j))
                        stack.append((i,k))
                        break
        pairs = list("." * N)
        for i,j in bp:
            pairs[i] = "("; pairs[j] = ")"
        return "".join(pairs)

    def outside(self, sequence, verbose=1):
        N = len(sequence)
        omega = np.zeros(shape=(N+1,N+1), dtype=int)
        max_iter = sum([i for i in range(N+1)])-1
        it = 0
        # NOTE: omega[i+1][j] and gamma[i][j] indicate same position (seq[i],seq[j])
        for ini_j in reversed(range(N-1)):
            for i in range(N-ini_j):
                j = ini_j+i
                delta = 1 if (i>0 and j+1<N) and (j-i>self.minspan) and self.is_bp(sequence[i-1], sequence[j+1]) else 0
                omega[i+1,j]=max(
                    omega[i][j],
                    omega[i+1][j+1],
                    omega[i][j+1] + delta,
                    max([0]+[omega[k+1][j] + self.gamma[k][i-1] for k in range(i)]),
                    max([0]+[self.gamma[j+1][k] + omega[i+1][k] for k in range(j+1,N)])
                )
                flush_progress_bar(it, max_iter, barname="ω", metrics={"max num of base-pairs": np.max(omega)}, verbose=verbose)
                it+=1
        print()
        omega = omega[1:,:-1]
        self.omega=omega
        return omega

    def ConstrainedMaximize(self, sequence, verbose=1):
        """ Calcurate the probability of sequence when xi and yj form base-pairs. """
        if self.gamma is None:
            _ = self.predict(sequence, display=False, verbose=verbose)
        if self.omega is None:
            _ = self.outside(sequence, verbose=verbose)
        N = len(sequence)
        Z = np.zeros(shape=(N,N),dtype=int)
        for ini_i in reversed(range(N)):
            diff = N-ini_i
            for i in reversed(range(ini_i)):
                j = i+diff
                Z[i][j] = self.gamma[i+1][j-1]+1+self.omega[i][j] if (j-i>self.minspan) and self.is_bp(sequence[i],sequence[j]) else 0
        self.Z = Z
        return self.Z

class Zuker(BaseStructureModel):
    """ Zuker Algorithm """
    def __init__(self, **kwargs):
        self.inf = np.inf # You can adjust this value by parameters file.
        super().__init__(**kwargs)
        self.disp_params += ["hairpin","stacking_cols","stacking_score","buldge","internal","a","b","c"]

    def _padding_stacking_score(self):
        """ Padding the score array to calcurate the index by `self.is_bp` method. """
        N_arr  = len(self.stacking_score)
        N_cols = len(self.stacking_cols)
        if N_cols==N_arr:
            padd_statcking_score = np.full(shape=(N_cols+1,N_cols+1), fill_value=self.inf)
            padd_statcking_score[:N_cols,:N_cols] = self.stacking_score
            self.stacking_score = padd_statcking_score
        elif N_cols+1 == N_arr:
            # Already aranged.
            pass
        else:
            raise ValueError(f"`len(self.stacking_score)` - `len(self.stacking_cols)` should be 0 or 1. However {N_arr-N_cols}.")

    def _calW(self,i,j):
        return min(
            self.W[i+1][j],
            self.W[i][j-1],
            self.V[i][j],
            min([self.W[i][k] + self.W[k+1][j] for k in range(i,j)])
        )

    def _calV(self,sequence,i,j):
        return min(
            self._F1(i,j),
            min([self.inf]+[self._F2(sequence,i,j,h,l)+self.V[h][l] for h in range(i+1,j-1) for l in range(h+1,j) if self.is_bp(sequence[h],sequence[l])]),
            min([self.inf]+[self.M[i+1][k] + self.M[k+1][j-1] for k in range(i+1,j-1)]) + self.a + self.b,
        ) if self.is_bp(sequence[i], sequence[j]) else self.inf

    def _calM(self,i,j):
        return min(
            self.V[i][j]   + self.b,
            self.M[i+1][j] + self.c,
            self.M[i][j-1] + self.c,
            min([self.inf] + [self.M[i][k]+self.M[k+1][j] for k in range(i,j)])
        )

    def _F1(self,i,j):
        """ i...j → length=3 → nt=2 (0-origin) """
        nt=(j-i-1)-1 # 0-origin.
        return self.hairpin[nt] if nt<30 else self.inf

    def _F2(self,sequence,i,j,h,l):
        """ i..h - l...j → length=2+3=5 → nt=4 """
        nt = (h-i-1)+(j-l-1)-1 # 0-origin.
        if nt>=30:
            val = self.inf
        elif (h==i+1) and (l==j-1):
            # stacking
            val = self.stacking_score[self.bp_ps(sequence[i],sequence[j])][self.bp_ps(sequence[i+1],sequence[j-1])]
        elif (i+1<h<l<j-1):
            # internal loop
            val = self.internal[nt]
        else:
            # bulge loop
            val = self.buldge[nt]
        return val

    def Initialization(self):
        """ Initialize Dinamyc Programming Matrix. """
        self._padding_stacking_score()
        self.bp_ps = bpHandler(bp2id=self.stacking_cols)
        self.is_bp = bpHandler(nucleic_acid=self.nucleic_acid, WatsonCrick=self.WatsonCrick, Wobble=self.Wobble)

    def DynamicProgramming(self, sequence, verbose=1):
        """ Recursion of Dynamic Programming. """
        N = len(sequence)
        self.V  = np.full(shape=(N,N), fill_value=self.inf, dtype=float)
        self.M  = np.full(shape=(N,N), fill_value=self.inf, dtype=float)
        self.W  = np.full(shape=(N,N), fill_value=0,        dtype=float)
        max_iter = sum([i for i in range(N)])
        it = 0
        for ini_i in reversed(range(N)):
            diff = N-ini_i
            for i in reversed(range(ini_i)):
                j = i+diff
                self.V[i][j] = self._calV(sequence,i,j)
                self.M[i][j] = self._calM(i,j)
                self.W[i][j] = self._calW(i,j)
                flush_progress_bar(it, max_iter, metrics={"min energy": np.min(self.W)}, verbose=verbose)
                it+=1
        print()
        score = self.W[0][-1]
        return score

    def TraceBack(self, sequence):
        """ Main trackback to find which bases form base-pairs. """
        N = len(sequence)
        self.bp = []
        stack = [(0,N-1)]
        while(stack):
            i,j = stack.pop()
            if (i>=j): continue
            elif self.W[i][j] == self.W[i+1][j]: stack.append((i+1,j))
            elif self.W[i][j] == self.W[i][j-1]: stack.append((i,j-1))
            elif self.W[i][j] == self.V[i][j]: self.tracebackV(sequence, i,j)
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

    def tracebackV(self, sequence, i, j):
        self.bp.append((i,j))
        if self.V[i][j] == self._F1(i,j): pass
        else:
            for h in range(i+1,j-1):
                for l in range(h+1,j):
                    if self.is_bp(sequence[h],sequence[l]):
                        if self.V[i][j] == self._F2(sequence, i,j,h,l) + self.V[h][l]:
                            self.tracebackV(sequence,h,l)
        for k in range(i+1,j-1):
            if self.V[i][j] == self.M[i+1][k] + self.M[k+1][j-1] + self.a+self.b:
                self.tracebackM(sequence,i+1,k)
                self.tracebackM(sequence,k+1,j-1)

    def tracebackM(self, sequence, i, j):
        if self.M[i][j] == self.V[i][j] + self.b:
            self.tracebackV(sequence, i,j)
        elif self.M[i][j] == self.M[i+1][j] + self.c: self.tracebackM(sequence, i+1,j)
        elif self.M[i][j] == self.M[i][j-1] + self.c: self.tracebackM(sequence, i,j-1)
        else:
            for k in range(i,j):
                if self.M[i][j] == self.M[i][k] + self.M[k+1][j]:
                    self.tracebackM(sequence, i,k)
                    self.tracebackM(sequence, k+1,j)
