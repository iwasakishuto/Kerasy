# coding: utf-8
import numpy as np
from ..utils import CategoricalEncoder
from ..utils import printAlignment
from ..utils import inverse_arr
from ..utils import priColor

from .suffix import SAIS_create, SAIS_recursion
from .suffix import BWT_create, reverseBWT, C_create, Occ_create
from .prefix import lcp, LCP_create

def checkString(string):
    if "$" in string:
        raise TypeError(
            "You do not need to add '$' at the end.",
            "If '$' is in `string`, please replace them."
        )

class StringSearch():
    """ Data Structure for searching particular pattern. """
    def build(self, string):
        checkString(string)
        self.encoder = CategoricalEncoder()
        self.string = string

        print("Building Suffix Array...")
        encoded_string = list(self.encoder.to_categorical(string, origin=1))
        self.SA  = SAIS(encoded_string)
        self.ISA = inverse_arr(self.SA)

        print("Building Auxiliary data structure for BWT...")
        self.BWT = BWT_create(string, self.SA)
        self.Occ = Occ_create(self.BWT, self.encoder.obj2cls)
        self.C   = C_create(self.BWT, self.encoder.obj2cls)

        print("Building Longest-Common-Prefix Array...")
        self.LCP = LCP_create(string, self.SA)

    def show(self):
        """ Print Suffix Array. """
        digit = max(len(str(self.SA[0])), 3)
        print(f"{'i':>{digit}} {'SA':>{digit}} {'LCP':>{digit}} Suffix")
        print("-"*(digit*3+9))
        for i,pos in enumerate(self.SA):
            print(f"{i:>{digit}} {pos:>{digit}} {self.LCP[i]:>{digit}} {self.string[pos:]+'$'}")

    def search(self, query):
        """ Scan through string looking for a match to the query.
        returning a all of the match indexes, or -1 if no match was found.
        """
        lb = 0;  ub=int(len(self.BWT)-1)
        q = ""
        for cha in reversed(query):
            if cha not in self.encoder.obj2cls.keys():
                return -1
            q  = cha+q
            lb = int(self.C[cha] + self.Occ[self.encoder.obj2cls[cha]-1][lb-1])
            ub = int(self.C[cha] + self.Occ[self.encoder.obj2cls[cha]-1][ub] - 1)
            if lb>ub:
                return -1
        return self.SA[lb:ub+1]

    def where(self, query, width=60, is_adjacent=False):
        len_query  = len(query)
        len_string = len(self.string)

        positions  = self.search(query)
        score = 0 if isinstance(positions, int) else len(positions)

        if len_query<3:
            is_adjacent = False
        else:
            post_end_pos=-1
            for pos in positions:
                if pos==post_end_pos:
                    is_adjacent = True
                    break
                post_end_pos = pos+len_query

        pos_info   = list(" " * len_string)
        if not isinstance(positions, int):
            for pos in positions:
                if is_adjacent:
                    mark = "<" + "-"*(len_query-2) + ">"
                else:
                    mark = "*" * len_query
                pos_info[pos:pos+len_query] = list(mark)
        pos_info = "".join(pos_info)

        printAlignment(
            sequences=[self.string, pos_info],
            indexes=[np.arange(len_string), np.arange(len_string)],
            score=score,
            scorename="Number of matches",
            add_info=f"Query: {query}",
            seqname=['S', ' '],
            model="Suffix Array",
            width=width,
            blank=" ",
        )
