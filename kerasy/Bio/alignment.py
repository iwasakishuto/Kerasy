# coding: utf-8
import json
import numpy as np

class NeedlemanWunshGotoh():
    """ Global Alignment. """
    def __init__(self, match=1, mismatch=-1, d=5, e=1, path=None):
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

    def s(self,x,y):
        """ Calcurate the match/mismatch score s[xi,yj] """
        return self.match if x==y else self.mismatch

    def align(self,X,Y,width=60):
        lenx=len(X)+1; leny=len(Y)+1
        size=lenx*leny
        M = np.zeros(shape=(size)) # DP matrix.
        T = np.zeros(shape=(size), dtype=int) # TraceBack.

        #=== Initialization ===
        for i in range(1,lenx):
            M[i*leny] = -self.d*i
            T[i*leny] = (i-1)*leny
        for j in range(1,leny):
            M[j] = -self.d*j
            T[j] = j-1

        #=== Recursion (DP) ===
        for i in range(1,lenx):
            for j in range(1,leny):
                candidates = [
                    M[leny*i+(j-1)]-self.d,
                    M[leny*(i-1)+j]-self.d,
                    M[leny*(i-1)+(j-1)]+self.s(X[i-1],Y[j-1]),
                ]
                pointers = [leny*i+(j-1), leny*(i-1)+j, leny*(i-1)+(j-1)]
                M[leny*i+j] = np.max(candidates)
                T[leny*i+j] = pointers[np.argmax(candidates)]

        #=== TraceBack ===
        pointer=T[-1]
        Xidxes=[lenx-1]; Yidxes=[leny-1];
        while True:
            Xidxes.append(pointer//leny); Yidxes.append(pointer%leny)
            pointer = T[pointer]
            if pointer==0:
                break
        Xidxes = np.array(Xidxes); Yidxes=np.array(Yidxes)
        alignedX = self._arangeSeq(X,Xidxes)
        alignedY = self._arangeSeq(Y,Yidxes)
        score = M[-1]
        print(f"Alignment score is {score}")
        print("="*(width+3))
        print("\n\n".join([f"X: {alignedX[i: i+width]}\nY: {alignedY[i: i+width]}" for i in range(0, len(alignedX), width)]))
        print("="*(width+3))

    def _arangeSeq(self, seq, idxes):
        masks = (np.roll(idxes,-1) == idxes) | (idxes==0)
        alignedSeq = "".join('-' if masks[i] else seq[idxes[i]-1] for i in reversed(range(len(idxes))))
        return alignedSeq

    def params(self):
        print(f"match                 {self.match}")
        print(f"mismatch              {self.mismatch}")
        print(f"gap opening penalty   {self.d}")
        print(f"gap extension penalty {self.e}")
