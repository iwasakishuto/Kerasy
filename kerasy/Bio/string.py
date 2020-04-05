# coding: utf-8
import numpy as np
from ..utils import CategoricalEncoder
from ..utils import printAlignment
from ..utils import inverse_arr
from ..utils import priColor
from ..utils import fout_args

from .suffix import SAIS, checkString
from .suffix import BWT_create, reverseBWT, C_create, Occ_create
from .prefix import lcp, LCP_create
from .factorization import LPF_create, LZfactorization
from ..clib import c_tandem

alphabets = [chr(i) for i in range(ord("a"), ord("z")+1)]
ALPHABETS = [chr(i) for i in range(ord("A"), ord("Z")+1)]

class StringSearch():
    """ Data Structure for searching particular pattern. """
    def __init__(self, string=None, verbose=1):
        if string is not None:
            self.build(string, verbose=verbose)

    def build(self, string, verbose=1):
        checkString(string)
        self.encoder = CategoricalEncoder()
        self.string = string

        if verbose>0: print("Building Suffix Array...")
        encoded_string = list(self.encoder.to_categorical(string, origin=1))
        self.SA  = SAIS(encoded_string)
        self.ISA = inverse_arr(self.SA)

        if verbose>0: print("Building Auxiliary data structure for BWT...")
        self.BWT = BWT_create(string, self.SA)
        self.Occ = Occ_create(self.BWT, self.encoder.obj2cls)
        self.C   = C_create(self.BWT, self.encoder.obj2cls)

        if verbose>0: print("Building Longest Common Prefix array...")
        self.LCP = LCP_create(string, self.SA)

        if verbose>0: print("Building Longest Previous Factor array...")
        self.LPF = np.append(LPF_create(SA=self.SA[1:], LCP=self.LCP[:-1]), 0)

        if verbose>0: print("Building s-factorization...")
        self.LZ_factorization = LZfactorization(LPF=self.LPF[:-1], string=string)

    @property
    def s_factorization(self):
        return "|".join(self.LZ_factorization)

    def show(self, add_terminal=True):
        """ Print DB inner arrays. """
        if add_terminal:
            string = self.string + "$"
            SA  = self.SA
            LCP = self.LCP
            LPF = self.LPF
        else:
            string = self.string
            SA  = self.SA[1:]
            LCP = self.LCP[:-1]
            LPF = self.LPF[:-1]

        digit = max(len(str(len(string)-1)), 3)
        print_args = lambda *args: "".join([f"{str(arg):>{digit}} " for arg in args])
        title = print_args("i", "SA", "LCP", "LPF") + "Suffix"
        print(title)
        print("-"*len(title))
        for r,sa_r in enumerate(SA):
            print(print_args(r,sa_r,LCP[r],LPF[r]) + string[sa_r:])

    def save(self, path, sep="\t", add_terminal=True):
        if add_terminal:
            string = self.string + "$"
            SA  = self.SA
            LCP = self.LCP
            LPF = self.LPF
        else:
            string = self.string
            SA  = self.SA[1:]
            LCP = self.LCP[:-1]
            LPF = self.LPF[:-1]

        with open(path, mode="w") as f:
            f.write(fout_args("Rank r", "SA[r]", "LCP[r]", "LPF[SA[r]]", "Suffix", sep="\t"))
            for r,sa_r in enumerate(SA):
                f.write(fout_args(r, sa_r, LCP[r], LPF[sa_r], string[sa_r:], sep="\t"))

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

    def calc_tandem_score(self, query):
        len_query = len(query)
        positions = np.sort(self.search(query))
        span = np.diff(positions)
        not_repeat = np.nonzero(span != len_query)[0]
        len_repeat = np.diff(np.append(np.append(-1, not_repeat), len(span)))
        return len_query*np.max(len_repeat)

    def find_tandem(self):
        tandems = c_tandem._SAIS_Tandem(self.LZ_factorization + ["$"])
        scores = [self.calc_tandem_score(tandem) for tandem in tandems]
        return tandems[np.argmax(scores)]

    def where(self, query, width=60, is_adjacent=False):
        len_query  = len(query)
        len_string = len(self.string)

        positions  = np.sort(self.search(query))
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
