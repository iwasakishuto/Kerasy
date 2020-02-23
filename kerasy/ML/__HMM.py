""" old version of HMM.py This python file don't use the C-Extensions. """
#coding: utf-8
from __future__ import absolute_import

from ..utils import Params
from ..utils import flush_progress_bar

import numpy as np

class HMM(Params):
    __name__ = "Hidden Markov Model"
    __np_params__ = ["pi","A","phi"]
    __initialize_method__ = ['list2np','mkdict']

    def __init__(self):
        self.M = None
        self.K = None
        self.basetypes = None
        self.base2int = None
        self.int2base = None
        self.pi = None
        self.A = None
        self.phi = None
        self.epoch = 0
        self.history = []
        self.indent = 5+len(str(self.K))

    def mkdict(self):
        """ Make a dictionaly to connect base and int. """
        self.base2int = dict(zip(self.basetypes, range(self.M)))
        self.int2base = dict(zip(range(self.M), self.basetypes))

    def fit(self, sequences, epochs=1, verbose=1, rtol=1e-7):
        pMLL = 0 # post Mean Log Likelihood
        N = len(sequences)
        sequences = [np.array([self.base2int[s] for s in seq]) for seq in sequences]
        for epoch in range(epochs):
            pis  = np.zeros(shape=(0,*self.pi.shape))
            As   = np.zeros(shape=(0,*self.A.shape))
            phis = np.zeros(shape=(0,*self.phi.shape))
            for sequence in sequences:
                gamma, xi = self.Estep(sequence) # E step
                pi,A,phi = self._Mstep(gamma,xi,sequence) # M step (Not update now)
                pis = np.r_[pis,pi[None,:]]; As = np.r_[As,A[None,:,:]]; phis = np.r_[phis,phi[None,:,:]]

            # Update.
            self.pi  = np.mean(pis,  axis=0)
            self.A   = np.mean(As,   axis=0)
            self.phi = np.mean(phis, axis=0)

            LLs=0 # Log Likelihood
            for sequence in sequences:
                LLs+=self.Estep(sequence, only_LL=True)
            MLL = LLs/N
            flush_progress_bar(epoch, epochs, metrics={"Log likelihood": MLL}, verbose=verbose)
            self.epoch+=1
            self.history.append(MLL)
            if abs(MLL-pMLL)<rtol:
                break
            pMLL=MLL
        if verbose: print()

    def Estep(self,sequence, only_LL=False):
        """Calcurate the γ and ξ for M step."""
        N = len(sequence)
        alpha = np.zeros(shape=(N,self.K))
        beta  = np.zeros(shape=(N,self.K))
        c = np.zeros(shape=(N,)) # scaling factor.

        #=== Forward Algorithm ===
        # Initialization
        alpha[0] = self.pi * self.phi[:,sequence[0]] # α(z1) = p(z1)p(x1|z1) = p(x1,z1)
        c[0] = np.sum(alpha[0]) # c1 = ∑p(x1,z1) = p(x1)
        alpha[0] /= c[0] # α'(z1) = α(z1)/c1 = p(x1,z1)/p(x1) = p(z1|x1)
        # Recursion
        for i in range(1, N):
            # α(zn) = p(xn|zn) ∑p(zn-1|x1,...,xn-1)p(zn|zn-1) = p(xn,zn|x1,...,xn-1)
            alpha[i] = self.phi[:,sequence[i]] * alpha[i-1].dot(self.A)
            c[i] = np.sum(alpha[i]) # ci = ∑p(xn,zn|x1,...,xn-1) = p(xn|x1,...,xn-1)
            alpha[i] /= c[i] # α'(z1) = p(xn,zn|x1,...,xn-1)/p(xn|x1,...,xn-1) = p(zn|x1,...,xn)

        if only_LL: return np.sum(np.log(c)) # np.log(np.prod(c))

        #=== Backward Algorithm ===
        # Initialization
        beta[-1] = 1 # β(zN) = 1
        # Recursion
        for i in reversed(range(N-1)):
            # β(zn)  = ∑ β(zn+1)p(xn+1|zn+1)p(zn+1|zn)
            beta[i]  = (beta[i+1]*self.phi[:,sequence[i+1]]).dot(self.A.T)
            beta[i] /= c[i+1]

        gamma  = alpha * beta # γi = α(zi)β(zi) shape=(N,K)
        xi  =  (1/c[1:, None, None]) * alpha[:-1,:,None] * self.phi[:,sequence[1:]].T[:,None,:] * self.A[None,:,:] * beta[1:,None,:] # shape=(N-1,Kb,Ka)
        return gamma, xi

    def Mstep(self,gamma,xi,sequence):
        """ Update the params on the fly. """
        # Update the "initial state"; πk=γ_1k/∑γ_1k
        self.pi = gamma[0] / np.sum(gamma[0])

        # Update the "transition probability"; Ajk=∑ξ_ijk/∑∑ξijk
        self.A  = np.sum(xi, axis=0) / np.sum(xi, axis=(0,2))[:,None] # shape=(Kb,Ka) / shape=(Kb)[:,None] → shape=(Kb,1)

        # Update the emission probability; ϕ_kj=∑γ_ik*x_ij/∑∑γ_ik * x_ij
        x   = gamma[:,None,:] * (np.eye(self.M)[sequence])[:,:,None] # shape=(N,K,M)
        self.phi = (np.sum(x, axis=0) / np.sum(gamma, axis=0)).T # shape=(K,M)

    def _Mstep(self,gamma,xi,sequence):
        """ Don't update on the fly """
        pi  = gamma[0] / np.sum(gamma[0])
        A   = np.sum(xi, axis=0) / np.sum(xi, axis=(0,2))[:,None]
        x   = gamma[:,None,:] * (np.eye(self.M)[sequence])[:,:,None]
        phi = (np.sum(x, axis=0) / np.sum(gamma, axis=0)).T
        return (pi,A,phi)

    def predict(self, sequences, asai=True):
        """
        Viterbi algorithm: DP for finding the most likely sequence of hidden states.
        ============================================================================
        @val A[i][j]: The transition probability (s_i → s_j)
        @val φ[i][j]: The emission probability (s_i emmits jth alhpabet.)
        """
        predictions=[]
        sequences = [np.array([self.base2int[s] for s in seq]) for seq in sequences]
        for sequence in sequences:
            N = len(sequence)
            v = np.zeros(shape=(N, self.K))
            v[0] = self.phi[:, sequence[0]] * self.pi
            for i in range(1, N):
                v[i] = self.phi[:, sequence[i]] * np.max(v[i-1]*self.A.T, axis=1)
                v[i] /= np.mean(v[i])

            # Traceback
            indexes = np.zeros(shape=(N,), dtype=int)
            indexes[-1] = np.argmax(v[-1])
            for i in reversed(range(N-1)):
                indexes[i] = np.argmax((v[i]*self.A.T)[indexes[i+1]])
            prediction = "".join((indexes+asai).astype(str))
            predictions.append(prediction)
        return predictions

    def MCK(self, sequences):
        """ Marginalized Count Kernel """
        sequences = [np.array([self.base2int[s] for s in seq]) for seq in sequences]
        MCKs = np.zeros(shape=(0,self.M,self.K))
        for sequence in sequences:
            N = len(sequence)
            gamma, _ = self.Estep(sequence) # shape=(N,K)
            MCK = 1/N * np.eye(self.M)[sequence].T.dot(gamma) # (M,N)@(N,K)=(M,K)
            MCKs = np.r_[MCKs, MCK[None,:,:]]
        return MCKs

#=================================================================
# NOTE: Function to display parameters beautifully. Not necessary
    def params(self):
        length = self.indent + 7*self.K
        """print the model's parameters"""
        print("【Parameter】"); print("=" * length)
        print("*initial state"); self.initial_state(indent=True); print("-" * length)
        print("*transition probability matrix"); self.translation_prob(); print("-" * length)
        print("*emission probability"); self.emission_prob(indent=True)

    def initial_state(self, indent=False):
        z_format = ""
        for k in range(self.K):
            z_format += f"{f'z({k+1})':^7}"
        print(f"{' '*self.indent if indent else ''}{z_format}")

        pi_format = ""
        for k in range(self.K):
            pi_format += f"{self.pi[k]:.4f} "
        print(f"{' '*self.indent if indent else ''}{pi_format}")

    def translation_prob(self):
        print(f"{' '*self.indent}{'[States after transition]':^{7*self.K}}")
        z_format = ""
        for k in range(self.K):
            z_format += f"{f'z({k+1})':^7}"
        print(f"{' '*self.indent}{z_format}")

        for j in range(self.K):
            A_form = f"{f'z({j+1})':^{self.indent}}"
            for k in range(self.K):
                A_form += f"{self.A[j][k]:.4f} "
            print(A_form)

    def emission_prob(self, indent=False):
        width = self.indent-1 if indent else max([len(str(d)) for d in self.base2int.keys()])

        z_format  = ""
        for k in range(self.K):
            z_format += f"{f'z({k+1})':^7}"
        print(f"{' '*self.indent if indent else ' '*(width+1)}{z_format}")

        for j in range(self.M):
            phi_form = f"{self.int2base[j]:>{width}} "
            for k in range(self.K):
                phi_form += f"{self.phi[k][j]:.4f} "
            print(phi_form)
