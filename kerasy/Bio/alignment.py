# coding: utf-8
import json
import numpy as np

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
    def load_params(self, path):
        with open(path) as params_file:
            params=json.load(params_file)
        self.match=params['match']
        self.mismatch=params['mismatch']
        self.d=params['d']
        self.e=params['e']
    def params(self):
        vals  = ['value',self.match,self.mismatch,self.d,self.e]
        names = ['parameter','match','mimatch','gap opening penalty','gap extension penalty']
        digit = max([len(str(val)) for val in vals])
        width = max([len(name) for name in names])
        for i,(name,val) in enumerate(zip(names,vals)):
            print(f"|{name:<{width}}|{val:>{digit}}|")
            if i==0: print('-'*(width+digit+3))

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
