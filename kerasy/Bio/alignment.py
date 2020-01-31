# coding: utf-8
import numpy as np
from scipy.special import logsumexp

from ..utils import Params
from ..utils import printAlignment
from ..utils import handleKeyError
from ..utils import flush_progress_bar
from ..utils import flatten_dual

class BaseAlignmentModel(Params):
    """Basement for Alignment models.
    ~~~~~~~ Variables.
    - nx             : the length of the sequence X) + 1. (For the margin of the DPmatrixmatrix.)
    - ny             : the length of the sequence Y) + 1. (For the margin of the DPmatrixmatrix.)
    - size           : nx × ny
    - DPmatrix       : the Matrix for Dynamic Programming. 1-dimentional array.
    - TraceBackMatrix: the Matrix for Trace Back. 1-dimentional array.
    ~~~~~~~ Methods
    - s          : Calcurate the match/mismatch score between base_x, and base_y
    - align      : Performe Alignment Program.
    - align_score: Performe Alignemnt Program. It is only for calcurating the score.
    - TraceBack  : Performe Trace Back for understanding which bases formed the base-pairs.
    - InitializeDPmatrix        : (Not Implemented) Initialize Dinamyc Programming Matrix.
    - InitializeTraceBackPointer: (Not Implemented) Initialize TraceBack Pointer Matrix.
    - DynamicProgramming        : (Not Implemented) Recursion of Dynamic Programming.
    """
    def __init__(self, **kwargs):
        self.nx=None # (the length of the sequence X) + 1. (For the margin of the DPmatrixmatrix.)
        self.ny=None # (the length of the sequence Y) + 1. (For the margin of the DPmatrixmatrix.)
        self.size=None # nx × ny
        self.DPmatrix=None # the Matrix for Dynamic Programming. 1-dimentional array.
        self.TraceBackMatrix=None
        self.__dict__.update(**kwargs)
        if len(kwargs)>0:
            self.format_params()

    def s(self,base_x,base_y):
        """ Calcurate the match/mismatch score s[xi,yj] """
        return self.match if base_x==base_y else self.mismatch

    def align(self,X,Y, width=60, only_score=False, display=True, verbose=1):
        self.nx=len(X)+1
        self.ny=len(Y)+1
        self.size=(len(X)+1)*(len(Y)+1)
        # Initialization
        self.InitializeDPmatrix()
        self.InitializeTraceBackPointer()
        score,pointer = self.DynamicProgramming(X,Y,verbose=verbose)
        if only_score:
            return score
        Xidxes,Yidxes = self.TraceBack(pointer)
        if display:
            printAlignment(sequences=[X,Y], indexes=[Xidxes,Yidxes], score=score, width=width, model=self.__class__.__name__)
        else:
            return (score,Xidxes,Yidxes)

    def align_score(self,X,Y):
        """ Calcurate only score. """
        score = self.align(X,Y,only_score=True)
        return score

    def InitializeDPmatrix(self):
        """ Initialize Dinamyc Programming Matrix. """
        raise NotImplementedError()

    def InitializeTraceBackPointer(self):
        """ Initialize TraceBack Pointer Matrix. """
        raise NotImplementedError()

    def DynamicProgramming(self, X, Y, verbose=1):
        """ Recursion of Dynamic Programming. """
        raise NotImplementedError()

    def TraceBack(self, pointer):
        handleKeyError(lst=["forward", "backward"], message="It indicates the direction of the Dynamic Programming.", direction=self.direction)
        handleKeyError(lst=["global", "local"], message="It indicates the range in which alignment is performed", method=self.method)
        Xidxes=[]; Yidxes=[];
        while True:
            Xidxes.append((pointer%self.size)//self.ny); Yidxes.append(pointer%self.ny)
            pointer = self.TraceBackMatrix[pointer]
            if self.direction == "forward"  and pointer==0: break
            if self.direction == "backward" and (pointer+1)==self.size: break
        if self.direction == "forward":
            # Because the traceback direction is backward.
            Xidxes = Xidxes[::-1];Yidxes=Yidxes[::-1]
        # Remove the initial header.
        if self.method == "local" and not (Xidxes[0]==0 and Yidxes[0]==0):
            Xidxes=Xidxes[1:]; Yidxes=Yidxes[1:]
        Xidxes = np.asarray(Xidxes, dtype=int)-1; Yidxes = np.asarray(Yidxes, dtype=int)-1
        return Xidxes,Yidxes

class NeedlemanWunshGotoh(BaseAlignmentModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disp_params = ["match", "mismatch", "gap_opening", "gap_extension"]
        self.direction = "forward"
        self.method = "global"

    def InitializeTraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int) # TraceBack.
        for i in range(2, self.nx):
            T[2*self.size+i*self.ny] = 2*self.size+(i-1)*self.ny # Y
        for j in range(2, self.ny):
            T[1*self.size+j] = 1*self.size+j-1 # X
        self.TraceBackMatrix = T

    def InitializeDPmatrix(self):
        DPmatrix = np.zeros(shape=(3*self.size)) # DPmatrix matrix.
        for k in range(1,3): DPmatrix[k*self.size] = -np.inf
        for i in range(1,self.nx):
            DPmatrix[0*self.size+i*self.ny] = -np.inf # M
            DPmatrix[1*self.size+i*self.ny] = -np.inf # X
            DPmatrix[2*self.size+i*self.ny] = -self.gap_opening-(i-1)*self.gap_extension # Y
        for j in range(1,self.ny):
            DPmatrix[0*self.size+j] = -np.inf # M
            DPmatrix[1*self.size+j] = -self.gap_opening-(j-1)*self.gap_extension # X
            DPmatrix[2*self.size+j] = -np.inf # Y
        self.DPmatrix = DPmatrix

    def DynamicProgramming(self,X,Y,verbose=1):
        max_iter = (self.nx-1)*(self.ny-1)
        for i in range(1,self.nx):
            for j in range(1,self.ny):
                candidates = [
                    [
                        self.DPmatrix[0*self.size+(i-1)*self.ny+(j-1)]+self.s(X[i-1],Y[j-1]),
                        self.DPmatrix[1*self.size+(i-1)*self.ny+(j-1)]+self.s(X[i-1],Y[j-1]),
                        self.DPmatrix[2*self.size+(i-1)*self.ny+(j-1)]+self.s(X[i-1],Y[j-1]),
                    ],
                    [
                        self.DPmatrix[0*self.size+(i-1)*self.ny+j]-self.gap_opening,
                        self.DPmatrix[1*self.size+(i-1)*self.ny+j]-self.gap_extension,
                    ],
                    [
                        self.DPmatrix[0*self.size+i*self.ny+(j-1)]-self.gap_opening,
                        self.DPmatrix[2*self.size+i*self.ny+(j-1)]-self.gap_extension,
                    ],
                ]
                pointers = [
                    [k*self.size+(i-1)*self.ny+(j-1) for k in (0,1,2)],
                    [k*self.size+(i-1)*self.ny+j for k in (0,1)],
                    [k*self.size+i*self.ny+(j-1) for k in (0,2)]
                ]
                for k in range(3):
                    self.DPmatrix[k*self.size+i*self.ny+j] = np.max(candidates[k])
                    self.TraceBackMatrix[k*self.size+i*self.ny+j] = pointers[k][np.argmax(candidates[k])]
                flush_progress_bar((i-1)*(self.ny-1)+(j-1), max_iter, barname="DP", metrics={"max alignment score": max(flatten_dual(candidates))}, verbose=verbose)
        if verbose>=1: print()
        scores = [self.DPmatrix[(k+1)*self.size-1] for k in range(3)]
        score = np.max(scores)
        pointer = (np.argmax(scores)+1)*self.size-1
        return score,pointer

class SmithWaterman(BaseAlignmentModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disp_params = ["match", "mismatch", "gap_opening", "gap_extension"]
        self.direction = "forward"
        self.method = "local"

    def InitializeTraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int) # TraceBack.
        # Below is Not necessary because it is initialized with 0
        for i in range(2,self.nx):
            T[2*self.size+i*self.ny] = 0 # Y
        for j in range(2,self.ny):
            T[1*self.size+j] = 0 # X
        self.TraceBackMatrix = T

    def InitializeDPmatrix(self):
        DPmatrix = np.zeros(shape=(3*self.size)) # DPmatrix matrix.
        for k in range(3): DPmatrix[k*self.size] = -np.inf
        for i in range(1,self.nx):
            DPmatrix[0*self.size+i*self.ny] = -np.inf # M
            DPmatrix[1*self.size+i*self.ny] = -np.inf # X
            DPmatrix[2*self.size+i*self.ny] = 0 # Y
        for j in range(1,self.ny):
            DPmatrix[0*self.size+j] = -np.inf # M
            DPmatrix[1*self.size+j] = 0 # X
            DPmatrix[2*self.size+j] = -np.inf # Y
        self.DPmatrix = DPmatrix

    def DynamicProgramming(self,X,Y, verbose=1):
        max_iter = (self.nx-1)*(self.ny-1)
        for i in range(1,self.nx):
            for j in range(1,self.ny):
                candidates = [
                    [
                        0,
                        self.DPmatrix[0*self.size+(i-1)*self.ny+(j-1)]+self.s(X[i-1],Y[j-1]),
                        self.DPmatrix[1*self.size+(i-1)*self.ny+(j-1)]+self.s(X[i-1],Y[j-1]),
                        self.DPmatrix[2*self.size+(i-1)*self.ny+(j-1)]+self.s(X[i-1],Y[j-1]),
                    ],
                    [
                        self.DPmatrix[0*self.size+(i-1)*self.ny+j]-self.gap_opening,
                        self.DPmatrix[1*self.size+(i-1)*self.ny+j]-self.gap_extension,
                    ],
                    [
                        self.DPmatrix[0*self.size+i*self.ny+(j-1)]-self.gap_opening,
                        self.DPmatrix[2*self.size+i*self.ny+(j-1)]-self.gap_extension,
                    ],
                ]
                pointers = [
                    [0]+[k*self.size+(i-1)*self.ny+(j-1) for k in (0,1,2)],
                    [k*self.size+(i-1)*self.ny+j for k in (0,1)],
                    [k*self.size+i*self.ny+(j-1) for k in (0,2)]
                ]
                for k in range(3):
                    self.DPmatrix[k*self.size+i*self.ny+j] = np.max(candidates[k])
                    self.TraceBackMatrix[k*self.size+i*self.ny+j] = pointers[k][np.argmax(candidates[k])]
                flush_progress_bar((i-1)*(self.ny-1)+(j-1), max_iter, barname="DP", metrics={"max alignment score": np.max(self.DPmatrix)}, verbose=verbose)
        if verbose>=1: print()
        score = np.max(self.DPmatrix)
        pointer = np.argmax(self.DPmatrix)
        return score,pointer

class BackwardNeedlemanWunshGotoh(BaseAlignmentModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disp_params = ["match", "mismatch", "gap_opening", "gap_extension"]
        self.direction = "backward"
        self.method = "global"

    def InitializeTraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int) # TraceBack.
        for i in range(self.nx-1):
            T[0*self.size+(i+1)*self.ny-1] = 0*self.size+(i+2)*self.ny-1 # bM
            T[2*self.size+(i+1)*self.ny-1] = 2*self.size+(i+2)*self.ny-1 # Y
        for j in range(self.ny-1):
            T[1*self.size-self.ny+j] = 1*self.size-self.ny+(j+1) # bM
            T[2*self.size-self.ny+j] = 2*self.size-self.ny+(j+1) # X
        self.TraceBackMatrix = T

    def InitializeDPmatrix(self):
        DPmatrix = np.zeros(shape=(3*self.size)) # DPmatrix matrix.
        for i in range(1,self.nx):
            DPmatrix[0*self.size+i*self.ny-1] = -self.gap_opening-(self.nx-i-1)*self.gap_extension # bM
            DPmatrix[1*self.size+i*self.ny-1] = -np.inf # bX
            DPmatrix[2*self.size+i*self.ny-1] = -(self.nx-i)*self.gap_extension # bY
        for j in range(self.ny-1):
            DPmatrix[1*self.size-self.ny+j] = -self.gap_opening-(self.ny-j-2)*self.gap_extension # bM
            DPmatrix[2*self.size-self.ny+j] = -(self.ny-j-1)*self.gap_extension # bX
            DPmatrix[3*self.size-self.ny+j] = -np.inf # bY
        self.DPmatrix = DPmatrix

    def DynamicProgramming(self,X,Y,verbose=1):
        max_iter = (self.nx-1)*(self.ny-1)
        for i in reversed(range(self.nx-1)):
            for j in reversed(range(self.ny-1)):
                candidates = [
                    [
                        self.DPmatrix[0*self.size+(i+1)*self.ny+(j+1)]+self.s(X[i],Y[j]),
                        self.DPmatrix[1*self.size+(i+1)*self.ny+j]-self.gap_opening,
                        self.DPmatrix[2*self.size+i*self.ny+(j+1)]-self.gap_opening,
                    ],
                    [
                        self.DPmatrix[0*self.size+(i+1)*self.ny+(j+1)]+self.s(X[i],Y[j]),
                        self.DPmatrix[1*self.size+(i+1)*self.ny+j]-self.gap_extension,
                    ],
                    [
                        self.DPmatrix[0*self.size+(i+1)*self.ny+(j+1)]+self.s(X[i],Y[j]),
                        self.DPmatrix[2*self.size+i*self.ny+(j+1)]-self.gap_extension,
                    ],
                ]
                pointers = [
                    [
                        0*self.size+(i+1)*self.ny+(j+1),
                        1*self.size+(i+1)*self.ny+j,
                        2*self.size+i*self.ny+(j+1),
                    ],
                    [
                        0*self.size+(i+1)*self.ny+(j+1),
                        1*self.size+(i+1)*self.ny+j,
                    ],
                    [
                        0*self.size+(i+1)*self.ny+(j+1),
                        2*self.size+i*self.ny+(j+1),
                    ],
                ]
                for k in range(3):
                    self.DPmatrix[k*self.size+i*self.ny+j] = np.max(candidates[k])
                    self.TraceBackMatrix[k*self.size+i*self.ny+j] = pointers[k][np.argmax(candidates[k])]
                flush_progress_bar(((self.nx-1)-i-1)*(self.ny-1)+(self.ny-1)-j-1, max_iter, barname="DP", metrics={"max alignment score": max(flatten_dual(candidates))}, verbose=verbose)
        if verbose>=1: print()
        score = self.DPmatrix[0]
        pointer = self.TraceBackMatrix[0]
        return score,pointer

class PairHMM(BaseAlignmentModel):
    """
    # NOTE:
    This model using logarithm so that the rounding does not occur due to the dynamic range.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disp_params = ["delta","epsilon","tau","px_e_y","px_ne_y","qx","qy"]
        self.direction = "forward"
        self.method = "global"

    def s(self,x,y):
        return self.px_e_y if x==y else self.px_ne_y

    def InitializeTraceBackPointer(self):
        T = np.zeros(shape=(3*self.size), dtype=int)
        for i in range(2,self.nx): T[1*self.size+i*self.ny+0] = 1*self.size+(i-1)*self.ny+0
        for j in range(2,self.ny): T[2*self.size+0*self.ny+j] = 2*self.size+0*self.ny+(j-1)
        self.TraceBackMatrix = T

    def InitializeDPmatrix(self):
        self.DPmatrix = self.InitializeDPmatrixForward()

    def InitializeDPmatrixForward(self):
        """ DPmatrix matrix for Viterbi or Forward algorithm. """
        F = np.zeros(shape=(3*self.size))

        for i in range(self.nx): F[i*self.ny+0] = -np.inf
        for j in range(self.ny): F[j] = -np.inf
        F[0] = 0

        F[1*self.size+1*self.ny+0] = np.log(self.delta*self.qx)
        for j in range(self.ny):
            F[1*self.size+0*self.ny+j] = -np.inf
        for i in range(2,self.nx):
            F[1*self.size+i*self.ny+0] = np.log(self.epsilon*self.qx) + F[1*self.size+(i-1)*self.ny+0]

        F[2*self.size+0*self.ny+1] = np.log(self.delta*self.qy)
        for i in range(self.nx):
            F[2*self.size+i*self.ny+0] = -np.inf
        for j in range(2,self.ny):
            F[2*self.size+0*self.ny+j] = np.log(self.epsilon*self.qy) + F[2*self.size+0*self.ny+(j-1)]
        return F

    def InitializeDPmatrixBackward(self):
        """ DPmatrix matrix for Backward algorithm. """
        B = np.zeros(shape=(3*self.size))
        for k in range(3): B[(k+1)*self.size-1] = np.log(self.tau)
        for i in reversed(range(1,self.nx-1)):
            B[0*self.size+(i+1)*self.ny-1] = np.log(self.delta*self.qx)   + B[0*self.size+(i+2)*self.ny-1]
            B[1*self.size+(i+1)*self.ny-1] = np.log(self.epsilon*self.qx) + B[0*self.size+(i+2)*self.ny-1]
        for j in reversed(range(1,self.ny-1)):
            B[0*self.size+(self.nx-1)*self.ny+j] = np.log(self.delta*self.qx)   + B[0*self.size+(self.nx-1)*self.ny+(j+1)]
            B[2*self.size+(self.nx-1)*self.ny+j] = np.log(self.epsilon*self.qy) + B[2*self.size+(self.nx-1)*self.ny+(j+1)]
        return B

    def DynamicProgramming(self,X,Y,verbose=1):
        """ Viterbi Algorithm """
        max_iter = (self.nx-1)*(self.ny-1)
        for i in range(1,self.nx):
            for j in range(1,self.ny):
                candidates = [
                    [
                        np.log(1-2*self.delta-self.tau) + self.DPmatrix[0*self.size+(i-1)*self.ny+(j-1)],
                        np.log(1-self.epsilon-self.tau) + self.DPmatrix[1*self.size+(i-1)*self.ny+(j-1)],
                        np.log(1-self.epsilon-self.tau) + self.DPmatrix[2*self.size+(i-1)*self.ny+(j-1)],
                    ],
                    [
                        np.log(self.delta)   + self.DPmatrix[0*self.size+(i-1)*self.ny+j],
                        np.log(self.epsilon) + self.DPmatrix[1*self.size+(i-1)*self.ny+j],
                    ],
                    [
                        np.log(self.delta)   + self.DPmatrix[0*self.size+i*self.ny+(j-1)],
                        np.log(self.epsilon) + self.DPmatrix[2*self.size+i*self.ny+(j-1)],
                    ],
                ]
                coefficients = [self.s(X[i-1],Y[j-1]), self.qx, self.qy]
                pointers = [
                    [k*self.size+(i-1)*self.ny+(j-1) for k in (0,1,2)],
                    [k*self.size+(i-1)*self.ny+j for k in (0,1)],
                    [k*self.size+i*self.ny+(j-1) for k in (0,2)]
                ]
                for k in range(3):
                    self.DPmatrix[k*self.size+i*self.ny+j] = np.log(coefficients[k]) + np.max(candidates[k])
                    self.TraceBackMatrix[k*self.size+i*self.ny+j] = pointers[k][np.argmax(candidates[k])]
                flush_progress_bar((i-1)*(self.ny-1)+(j-1), max_iter, barname="Viterbi", metrics={"max alignment score": max(flatten_dual(candidates))}, verbose=verbose)
        if verbose>=1: print()
        scores = [self.DPmatrix[(k+1)*self.size-1] for k in range(3)]
        score = np.log(self.tau)+max(scores)
        pointer = (np.argmax(scores)+1)*self.size-1
        return score, pointer

    def forward(self,X,Y,verbose=1):
        """
        Forward Algorithm.
        F[k,i,j] means the sum of all alignment('x1,..,xi' and 'y1,...,yj') probabilities
        under the condition that the state is k when xi and yj are considered.
        """
        self.nx=len(X)+1
        self.ny=len(Y)+1
        self.size=(len(X)+1)*(len(Y)+1)
        F = self.InitializeDPmatrixForward()
        max_iter = (self.nx-1)*(self.ny-1)
        for i in range(1,self.nx):
            for j in range(1,self.ny):
                F[0*self.size+i*self.ny+j] = np.log(self.s(X[i-1],Y[j-1])) + logsumexp([
                    np.log(1-2*self.delta-self.tau)+F[0*self.size+(i-1)*self.ny+(j-1)],
                    np.log(1-self.epsilon-self.tau)+F[1*self.size+(i-1)*self.ny+(j-1)],
                    np.log(1-self.epsilon-self.tau)+F[2*self.size+(i-1)*self.ny+(j-1)],
                ])
                F[1*self.size+i*self.ny+j] = np.log(self.qx) + logsumexp([
                    np.log(self.delta)+F[0*self.size+(i-1)*self.ny+j],
                    np.log(self.epsilon)+F[1*self.size+(i-1)*self.ny+j],
                ])
                F[2*self.size+i*self.ny+j] = np.log(self.qy) + logsumexp([
                    np.log(self.delta)+F[0*self.size+i*self.ny+(j-1)],
                    np.log(self.epsilon)+F[2*self.size+i*self.ny+(j-1)],
                ])
                max_score = max([F[k*self.size+i*self.ny+j] for k in range(3)])
                flush_progress_bar((i-1)*(self.ny-1)+(j-1), max_iter, barname="Forward", metrics={"max alignment score": max_score}, verbose=verbose)
        if verbose>=1: print()
        P = np.log(self.tau) + logsumexp([F[(k+1)*self.size-1] for k in range(3)])
        F = F.reshape(3,self.nx,self.ny)[0,1:,1:]
        return F, P

    def backward(self,X,Y,verbose=1):
        """
        Backward Algorithm.
        B[k,i,j] means the sum of all alignment('xi+1,..,xN' and 'yj+1,...,yM') probabilities
        under the condition that the state is k when xi and yj are considered.
        """
        self.nx=len(X)+1
        self.ny=len(Y)+1
        self.size=(len(X)+1)*(len(Y)+1)
        B = self.InitializeDPmatrixBackward()
        max_iter = (self.nx-1)*(self.ny-1)
        for i in reversed(range(self.nx-1)):
            for j in reversed(range(self.ny-1)):
                B[0*self.size+i*self.ny+j] = logsumexp([
                    np.log(1-2*self.delta-self.tau)+np.log(self.s(X[i],Y[j]))+B[0*self.size+(i+1)*self.ny+(j+1)],
                    np.log(self.delta*self.qx)+B[1*self.size+(i+1)*self.ny+j],
                    np.log(self.delta*self.qy)+B[1*self.size+i*self.ny+(j+1)],
                ])
                B[1*self.size+i*self.ny+j] = logsumexp([
                    np.log(1-self.epsilon-self.tau)+np.log(self.s(X[i],Y[j]))+B[0*self.size+(i+1)*self.ny+(j+1)],
                    np.log(self.epsilon*self.qx)+B[1*self.size+(i+1)*self.ny+j],
                ])
                B[2*self.size+i*self.ny+j] = logsumexp([
                    np.log(1-self.epsilon-self.tau)+np.log(self.s(X[i],Y[j]))+B[0*self.size+(i+1)*self.ny+(j+1)],
                    np.log(self.epsilon*self.qy)+B[1*self.size+(i+1)*self.ny+j],
                ])
                max_score = max([B[k*self.size+i*self.ny+j] for k in range(3)])
                flush_progress_bar(((self.nx-1)-i-1)*(self.ny-1)+(self.ny-1)-j-1, max_iter, barname="BackWard", metrics={"max alignment score": max_score}, verbose=verbose)
        if verbose>=1: print()
        B = B.reshape(3,self.nx,self.ny)[0,1:,1:]
        return B

    def score(self,X,Y,verbose=1):
        P_opt   = self.align(X,Y, only_score=True,verbose=verbose)
        F,P_all = self.forward(X,Y,verbose=verbose)
        print(f"- log(P(X,Y)) = {P_all:.3f}")
        print(f"- log(P(π*|x, y)) = {P_opt-P_all:.3f}")

    def align_ij(self, X, Y,verbose=1):
        """ Calcurate the Probability that bases i and j are aligned.
        Pij[i][j] = P(xi◇yj|x,y)
        """
        F,P_all = self.forward(X,Y,verbose=verbose)
        B = self.backward(X,Y,verbose=verbose)
        Pij = (F+B)-P_all
        return Pij
