# coding: utf-8
import json
import numpy as np
from fractions import Fraction

class Alignment():
    def __init__(self, match, mismatch, d, e, path):
        self.n=None
        self.m=None
        self.size=None
        self.DP=None
        self.TraceBack=None
        if path is not None:
            self.load_params(path)
        else:
            self.match=match
            self.mismatch=mismatch
            self.d=d
            self.e=e

    def set_params(self, path):
        with open(path) as params_file:
            params=json.load(params_file)
        for k in params.keys():
            params[k]=float(Fraction(params[k])) # Deal with Fraction (string format)
        self.__dict__.update(params)

    def params(self):
        key_title='parameter'; val_title="value"
        keys = [key for key in self.__dict__.keys() if key not in ["n","m","size","DP","TraceBack"]]
        vals = [self.__dict__[key] for key in keys]
        digit = max([len(val_title)] + [len(str(val)) for val in vals])
        width = max([len(key_title)] + [len(key) for key in keys])
        print(f"|{key_title:^{width}}|{val_title:^{digit}}|")
        print('-'*(width+digit+3))
        for i,(key,val) in enumerate(zip(keys,vals)): print(f"|{key:<{width}}|{val:>{digit}}|")

    def s(self,x,y):
        """ Calcurate the match/mismatch score s[xi,yj] """
        return self.match if x==y else self.mismatch

    def align(self,X,Y,width=60,memorize=False):
        self.n=len(X)+1; self.m=len(Y)+1; self.size=(len(X)+1)*(len(Y)+1)
        # Initialization
        DP = self._initialize_DPmatrix()
        T  = self._initialize_TraceBackPointer()
        # Recursion (DP)
        score,pointer,T = self._Recursion(DP,T,X,Y,memorize=memorize)
        # TraceBack
        Xidxes,Yidxes = self._TraceBack(T,pointer)
        alignedX,Xidxes = self._arangeSeq(X,Xidxes)
        alignedY,Yidxes = self._arangeSeq(Y,Yidxes)
        self._printAlignment(score,alignedX,alignedY,Xidxes,Yidxes,width=width)

    def _TraceBack(self,T,pointer):
        Xidxes=[]; Yidxes=[];
        while True:
            Xidxes.append((pointer%self.size)//self.m); Yidxes.append(pointer%self.m)
            pointer = T[pointer]
            if self.__direction__ == "forward"  and pointer==0: break
            if self.__direction__ == "backward" and (pointer+1)%self.size==0: break
        if self.__direction__ == "backward":
            Xidxes=Xidxes[::-1];Yidxes=Yidxes[::-1]
        if self.__method__ == "local" and not (Xidxes[-1]==1 and Yidxes[-1]==1):
            Xidxes=Xidxes[:-1]; Yidxes=Yidxes[:-1]
        return Xidxes,Yidxes

    def _arangeSeq(self, seq, idxes):
        if type(idxes) == type([]): idxes=np.array(idxes)
        masks = (np.roll(idxes,-1) == idxes) | (idxes==0)
        alignedSeq = "".join('-' if masks[i] else seq[idxes[i]-1] for i in reversed(range(len(idxes))))
        return alignedSeq, idxes-1

    def _printAlignment(self,score,alignedX,alignedY,Xidxes,Yidxes,width):
        print(f"\033[31m\033[07m {self.__name__} \033[0m\nAlignment score: \033[34m{score}\033[0m\n")
        self.params()
        print("="*(width+3))
        if self.__method__ == "local": print(f"Aligned positions: X[{min(Xidxes)},{max(Xidxes)}] Y[{min(Yidxes)},{max(Yidxes)}]")
        print("\n\n".join([f"X: {alignedX[i: i+width]}\nY: {alignedY[i: i+width]}" for i in range(0, len(alignedX), width)]))
        print("="*(width+3))

class NeedlemanWunshGotoh(Alignment):
    __name__ = "Needleman-Wunsh-Gotoh"
    __method__ = "global"
    __direction__ = "forward"

    def __init__(self, match=1, mismatch=-1, d=5, e=1, path=None):
        super().__init__(match,mismatch,d,e,path)

    def _initialize_TraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int) # TraceBack.
        for i in range(2,self.n):
            T[2*self.size+i*self.m] = 2*self.size+(i-1)*self.m # Y
        for j in range(2,self.m):
            T[1*self.size+j] = 1*self.size+j-1 # X
        return T

    def _initialize_DPmatrix(self):
        DP = np.zeros(shape=(3*self.size)) # DP matrix.
        for k in range(1,3): DP[k*self.size] = -np.inf
        for i in range(1,self.n):
            DP[0*self.size+i*self.m] = -np.inf # M
            DP[1*self.size+i*self.m] = -np.inf # X
            DP[2*self.size+i*self.m] = -self.d-(i-1)*self.e # Y
        for j in range(1,self.m):
            DP[0*self.size+j] = -np.inf # M
            DP[1*self.size+j] = -self.d-(j-1)*self.e # X
            DP[2*self.size+j] = -np.inf # Y
        return DP

    def _Recursion(self,DP,T,X,Y,memorize=False):
        for i in range(1,self.n):
            for j in range(1,self.m):
                candidates = [
                    [
                        DP[0*self.size+(i-1)*self.m+(j-1)]+self.s(X[i-1],Y[j-1]),
                        DP[1*self.size+(i-1)*self.m+(j-1)]+self.s(X[i-1],Y[j-1]),
                        DP[2*self.size+(i-1)*self.m+(j-1)]+self.s(X[i-1],Y[j-1]),
                    ],
                    [
                        DP[0*self.size+(i-1)*self.m+j]-self.d,
                        DP[1*self.size+(i-1)*self.m+j]-self.e,
                    ],
                    [
                        DP[0*self.size+i*self.m+(j-1)]-self.d,
                        DP[2*self.size+i*self.m+(j-1)]-self.e,
                    ],
                ]
                pointers = [
                    [k*self.size+(i-1)*self.m+(j-1) for k in (0,1,2)],
                    [k*self.size+(i-1)*self.m+j for k in (0,1)],
                    [k*self.size+i*self.m+(j-1) for k in (0,2)]
                ]
                for k in range(3):
                    DP[k*self.size+i*self.m+j] = np.max(candidates[k])
                    T[k*self.size+i*self.m+j]  = pointers[k][np.argmax(candidates[k])]
        scores = [DP[(k+1)*self.size-1] for k in range(3)]
        score = np.max(scores)
        pointer = (np.argmax(scores)+1)*self.size-1
        if memorize:
            self.DP=DP
            self.TraceBack = T
        return score,pointer,T

class SmithWaterman(Alignment):
    """ Local Alignment. """
    __name__ = "Smith-Waterman"
    __method__ = "local"
    __direction__ = "forward"

    def __init__(self, match=1, mismatch=-1, d=5, e=1, path=None):
        super().__init__(match,mismatch,d,e,path)

    def _initialize_TraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int) # TraceBack.
        # Below is Not necessary because it is initialized with 0
        for i in range(2,self.n):
            T[2*self.size+i*self.m] = 0 # Y
        for j in range(2,self.m):
            T[1*self.size+j] = 0 # X
        return T

    def _initialize_DPmatrix(self):
        DP = np.zeros(shape=(3*self.size)) # DP matrix.
        for k in range(3): DP[k*self.size] = -np.inf
        for i in range(1,self.n):
            DP[0*self.size+i*self.m] = -np.inf # M
            DP[1*self.size+i*self.m] = -np.inf # X
            DP[2*self.size+i*self.m] = 0 # Y
        for j in range(1,self.m):
            DP[0*self.size+j] = -np.inf # M
            DP[1*self.size+j] = 0 # X
            DP[2*self.size+j] = -np.inf # Y
        return DP

    def _Recursion(self,DP,T,X,Y,memorize=False):
        for i in range(1,self.n):
            for j in range(1,self.m):
                candidates = [
                    [
                        0,
                        DP[0*self.size+(i-1)*self.m+(j-1)]+self.s(X[i-1],Y[j-1]),
                        DP[1*self.size+(i-1)*self.m+(j-1)]+self.s(X[i-1],Y[j-1]),
                        DP[2*self.size+(i-1)*self.m+(j-1)]+self.s(X[i-1],Y[j-1]),
                    ],
                    [
                        DP[0*self.size+(i-1)*self.m+j]-self.d,
                        DP[1*self.size+(i-1)*self.m+j]-self.e,
                    ],
                    [
                        DP[0*self.size+i*self.m+(j-1)]-self.d,
                        DP[2*self.size+i*self.m+(j-1)]-self.e,
                    ],
                ]
                pointers = [
                    [0]+[k*self.size+(i-1)*self.m+(j-1) for k in (0,1,2)],
                    [k*self.size+(i-1)*self.m+j for k in (0,1)],
                    [k*self.size+i*self.m+(j-1) for k in (0,2)]
                ]
                for k in range(3):
                    DP[k*self.size+i*self.m+j] = np.max(candidates[k])
                    T[k*self.size+i*self.m+j]  = pointers[k][np.argmax(candidates[k])]
        score = np.max(DP)
        pointer = np.argmax(DP)
        if memorize:
            self.DP=DP
            self.TraceBack = T
        return score,pointer,T

class BackwardNeedlemanWunshGotoh(Alignment):
    __name__ = "Backward Needleman-Wunsh-Gotoh"
    __method__ = "global"
    __direction__ = "backward"

    def __init__(self, match=1, mismatch=-1, d=5, e=1, path=None):
        super().__init__(match,mismatch,d,e,path)

    def _initialize_TraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int) # TraceBack.
        for i in range(self.n-1):
            T[0*self.size+(i+1)*self.m-1] = 0*self.size+(i+2)*self.m-1 # bM
            T[2*self.size+(i+1)*self.m-1] = 2*self.size+(i+2)*self.m-1 # Y
        for j in range(self.m-1):
            T[1*self.size-self.m+j] = 1*self.size-self.m+(j+1) # bM
            T[2*self.size-self.m+j] = 2*self.size-self.m+(j+1) # X
        return T

    def _initialize_DPmatrix(self):
        DP = np.zeros(shape=(3*self.size)) # DP matrix.
        for i in range(1,self.n):
            DP[0*self.size+i*self.m-1] = -self.d-(self.n-i-1)*self.e # bM
            DP[1*self.size+i*self.m-1] = -np.inf # bX
            DP[2*self.size+i*self.m-1] = -(self.n-i)*self.e # bY
        for j in range(self.m-1):
            DP[1*self.size-self.m+j] = -self.d-(self.m-j-2)*self.e # bM
            DP[2*self.size-self.m+j] = -(self.m-j-1)*self.e # bX
            DP[3*self.size-self.m+j] = -np.inf # bY
        return DP

    def _Recursion(self,DP,T,X,Y,memorize=False):
        for i in reversed(range(self.n-1)):
            for j in reversed(range(self.m-1)):
                candidates = [
                    [
                        DP[0*self.size+(i+1)*self.m+(j+1)]+self.s(X[i],Y[j]),
                        DP[1*self.size+(i+1)*self.m+j]-self.d,
                        DP[2*self.size+i*self.m+(j+1)]-self.d,
                    ],
                    [
                        DP[0*self.size+(i+1)*self.m+(j+1)]+self.s(X[i],Y[j]),
                        DP[1*self.size+(i+1)*self.m+j]-self.e,
                    ],
                    [
                        DP[0*self.size+(i+1)*self.m+(j+1)]+self.s(X[i],Y[j]),
                        DP[2*self.size+i*self.m+(j+1)]-self.e,
                    ],
                ]
                pointers = [
                    [
                        0*self.size+(i+1)*self.m+(j+1),
                        1*self.size+(i+1)*self.m+j,
                        2*self.size+i*self.m+(j+1),
                    ],
                    [
                        0*self.size+(i+1)*self.m+(j+1),
                        1*self.size+(i+1)*self.m+j,
                    ],
                    [
                        0*self.size+(i+1)*self.m+(j+1),
                        2*self.size+i*self.m+(j+1),
                    ],
                ]
                for k in range(3):
                    DP[k*self.size+i*self.m+j] = np.max(candidates[k])
                    T[k*self.size+i*self.m+j]  = pointers[k][np.argmax(candidates[k])]
        score = DP[0]
        pointer = T[0]
        if memorize:
            self.DP=DP
            self.TraceBack = T
        return score,pointer,T

class PairHMM(Alignment):
    __name__ = "Pair HMM (Viterbi)"
    __method__ = "global"
    __direction__ = "forward"

    def __init__(self, path=None):
        super().__init__(path)

    def s(self,x,y):
        return self.px_e_y if x==y else self.px_ne_y

    def _memorize_seq_info(self,X,Y):
        self.n=len(X)+1; self.m=len(Y)+1; self.size=(len(X)+1)*(len(Y)+1)

    def _initialize_TraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int)
        for i in range(2,self.n): T[1*self.size+i*self.m+0] = 1*self.size+(i-1)*self.m+0
        for j in range(2,self.m): T[2*self.size+0*self.m+j] = 2*self.size+0*self.m+(j-1)
        return T

    def _initialize_DPmatrix(self):
        """ DP matrix for Viterbi or Forward algorithm. """
        F = np.zeros(shape=(3*self.size))
        F[0] = 1
        F[1*self.size+1*self.m+0] = self.delta*self.qx
        for i in range(2,self.n): F[1*self.size+i*self.m+0] = self.epsilon*self.qx*F[1*self.size+(i-1)*self.m+0]
        F[2*self.size+0*self.m+1] = self.delta*self.qy
        for j in range(2,self.m): F[2*self.size+0*self.m+j] = self.epsilon*self.qy*F[2*self.size+0*self.m+(j-1)]
        return F

    def _initialize_DPmatrix_Backward(self):
        """ DP matrix for Backward algorithm. """
        B = np.zeros(shape=(3*self.size))
        for k in range(3): B[(k+1)*self.size-1] = self.tau
        for i in reversed(range(1,self.n-1)):
            B[0*self.size+(i+1)*self.m-1] = self.delta*  self.qx*B[0*self.size+(i+2)*self.m-1]
            B[1*self.size+(i+1)*self.m-1] = self.epsilon*self.qx*B[0*self.size+(i+2)*self.m-1]
        for j in reversed(range(1,self.m-1)):
            B[0*self.size+(self.n-1)*self.m+j] = self.delta*  self.qx*B[0*self.size+(self.n-1)*self.m+(j+1)]
            B[2*self.size+(self.n-1)*self.m+j] = self.epsilon*self.qy*B[2*self.size+(self.n-1)*self.m+(j+1)]
        return B

    def _Recursion(self,DP,T,X,Y,memorize=False):
        """ Viterbi Algorithm """
        for i in range(1,self.n):
            for j in range(1,self.m):
                candidates = [
                    [
                        (1-2*self.delta-self.tau)*DP[0*self.size+(i-1)*self.m+(j-1)],
                        (1-self.epsilon-self.tau)*DP[1*self.size+(i-1)*self.m+(j-1)],
                        (1-self.epsilon-self.tau)*DP[2*self.size+(i-1)*self.m+(j-1)],
                    ],
                    [
                        self.delta*DP[0*self.size+(i-1)*self.m+j],
                        self.epsilon*DP[1*self.size+(i-1)*self.m+j],
                    ],
                    [
                        self.delta*DP[0*self.size+i*self.m+(j-1)],
                        self.epsilon*DP[2*self.size+i*self.m+(j-1)],
                    ],
                ]
                coefficients = [self.s(X[i-1],Y[j-1]), self.qx, self.qy]
                pointers = [
                    [k*self.size+(i-1)*self.m+(j-1) for k in (0,1,2)],
                    [k*self.size+(i-1)*self.m+j for k in (0,1)],
                    [k*self.size+i*self.m+(j-1) for k in (0,2)]
                ]
                for k in range(3):
                    DP[k*self.size+i*self.m+j] = coefficients[k] * np.max(candidates[k])
                    T[k*self.size+i*self.m+j] = pointers[k][np.argmax(candidates[k])]
        scores = [DP[(k+1)*self.size-1] for k in range(3)]
        score = self.tau*max(scores)
        pointer = (np.argmax(scores)+1)*self.size-1
        if memorize:
            self.DP=DP
            self.TraceBack = T
        return score,pointer,T

    def forward(self,X,Y):
        """
        Forward Algorithm.
        F[k,i,j] means the sum of all alignment('x1,..,xi' and 'y1,...,yj') probabilities
        under the condition that the state is k when xi and yj are considered.
        """
        self._memorize_seq_info(X,Y)
        F = self._initialize_DPmatrix()

        for i in range(1,self.n):
            for j in range(1,self.m):
                F[0*self.size+i*self.m+j] = self.s(X[i-1],Y[j-1]) * np.sum([
                    (1-2*self.delta-self.tau)*F[0*self.size+(i-1)*self.m+(j-1)],
                    (1-self.epsilon-self.tau)*F[1*self.size+(i-1)*self.m+(j-1)],
                    (1-self.epsilon-self.tau)*F[2*self.size+(i-1)*self.m+(j-1)],
                ])
                F[1*self.size+i*self.m+j] = self.qx * np.sum([
                    self.delta*F[0*self.size+(i-1)*self.m+j],
                    self.epsilon*F[1*self.size+(i-1)*self.m+j],
                ])
                F[2*self.size+i*self.m+j] = self.qy * np.sum([
                    self.delta*F[0*self.size+i*self.m+(j-1)],
                    self.epsilon*F[2*self.size+i*self.m+(j-1)],
                ])
        P = self.tau*np.sum([F[(k+1)*self.size-1] for k in range(3)])
        F = F.reshape(3,self.n,self.m)[0,1:,1:]
        return F, P

    def backward(self,X,Y):
        """
        Backward Algorithm.
        B[k,i,j] means the sum of all alignment('xi+1,..,xN' and 'yj+1,...,yM') probabilities
        under the condition that the state is k when xi and yj are considered.
        """
        self._memorize_seq_info(X,Y)
        B = self._initialize_DPmatrix_Backward()

        for i in reversed(range(self.n-1)):
            for j in reversed(range(self.m-1)):
                B[0*self.size+i*self.m+j] = np.sum([
                    (1-2*self.delta-self.tau)*self.s(X[i],Y[j])*B[0*self.size+(i+1)*self.m+(j+1)],
                    self.delta*self.qx*B[1*self.size+(i+1)*self.m+j],
                    self.delta*self.qy*B[1*self.size+i*self.m+(j+1)],
                ])
                B[1*self.size+i*self.m+j] = np.sum([
                    (1-self.epsilon-self.tau)*self.s(X[i],Y[j])*B[0*self.size+(i+1)*self.m+(j+1)],
                    self.epsilon*self.qx*B[1*self.size+(i+1)*self.m+j],
                ])
                B[2*self.size+i*self.m+j] = np.sum([
                    (1-self.epsilon-self.tau)*self.s(X[i],Y[j])*B[0*self.size+(i+1)*self.m+(j+1)],
                    self.epsilon*self.qy*B[1*self.size+(i+1)*self.m+j],
                ])
        B = B.reshape(3,self.n,self.m)[0,1:,1:]
        return B
