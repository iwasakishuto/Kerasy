from __future__ import absolute_import
from ..utils.params import Params

class BaseHandler(Params):
    __name__ = ""

    def _printAlignment(self, score, seq, pairs, width=60, xlabel="X", ylabel="Y"):
        label_length = max(len(xlabel),len(ylabel))
        print(f"\033[31m\033[07m {self.__name__} \033[0m\nScore: \033[34m{score}\033[0m\n")
        print("="*(width+label_length+2))
        print("\n\n".join([f"{xlabel:<{label_length}}: {seq[i: i+width]}\n{ylabel:<{label_length}}: {pairs[i: i+width]}" for i in range(0, len(seq), width)]))
        print("="*(width+label_length+2))
