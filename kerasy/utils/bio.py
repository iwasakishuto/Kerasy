from __future__ import absolute_import
from ..utils.params import Params

import numpy as np

class BaseHandler(Params):
    __name__ = ""

    def _printAlignment(self, score, seq, pairs, width=60, xlabel="X", ylabel="Y"):
        label_length = max(len(xlabel),len(ylabel))
        print(f"\033[31m\033[07m {self.__name__} \033[0m\nScore: \033[34m{score}\033[0m\n")
        print("="*(width+label_length+2))
        print("\n\n".join([f"{xlabel:<{label_length}}: {seq[i: i+width]}\n{ylabel:<{label_length}}: {pairs[i: i+width]}" for i in range(0, len(seq), width)]))
        print("="*(width+label_length+2))

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

def alignStr(string, width=60):
    return "\n".join([string[i: i+width] for i in range(0, len(string), width)])

def readMonoSeq(path, header=True):
    with open(path, "r") as f:
        seqfile = f.readlines()
    sequence = ["".join(seqfile[header:]).replace('\n', '')]
    return sequence

def readMultiSeq(path, symbol='>'):
    with open(path, "r") as f:
        sequences=[]
        seq=""
        for line in f:
            if line[0]==symbol:
                if len(seq)==0: continue
                sequences.append(seq)
                seq=""
            else:
                seq+=line.rstrip('\n')
        sequences.append(seq)
    return sequences
