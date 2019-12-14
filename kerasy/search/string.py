# coding: utf-8
import numpy as np
from ..utils import CategoricalEncoder
from ..utils import printAlignment

def unzipBWT(BWT):
    

class StringSearch():
    def build(self, string):
        if "$" in string:
            raise TypeError(
                "You do not need to add '$' at the end.",
                "If '$' is used in `string`, please replace it."
            )
        self.encoder = CategoricalEncoder()
        self.string = string
        self.len = len(string)+1 # + len("$")=1

        print("Building Suffix Array...")
        self.SA = np.asarray(self.SAIS(
            list(self.encoder.to_categorical(string, origin=1)),
            isLMSsorted=False,
            LMS=None
        ), dtype=int)
        # Inverse Suffix Array.
        self.ISA = np.zeros(shape=self.len, dtype=int)
        for i in range(self.len): self.ISA[self.SA[i]] = i
        # Burrows-Wheeler Transform
        print("Building Auxiliary data structure for BWT...")
        self._build_BWT()
        self._build_C()
        self._build_Occ()
        # Longest-Common-Prefix
        print("Building Longest-Common-Prefix Array...")
        self._build_height()

    def SuffixArray(self):
        """ Print Suffix Array. """
        digit = max(len(str(self.len)), 2)
        print(f"{'i':>{digit}} {'SA':>{digit}} {'H':>{digit}} Suffix")
        print("-"*(digit*3+9))
        for i,pos in enumerate(self.SA):
            print(f"{i:>{digit}} {pos:>{digit}} {self.height[i]:>{digit}} {self.string[pos:]+'$'}")

    @staticmethod
    def SAIS(string, isLMSsorted=False, LMS=None):
        """Make suffix array recursively in linear order.
        @params string      : (list) A query strings
        @params isLMSsorted : (bool) Whether LMS is already sorted or not.
        @params LMS         : (list) If is not None, `LMS` has the correct ordered LMS.
        @return SA          : (list) Suffix Array
        """
        #=== Step0: Exception handling ===
        # If S's length is 1, we don't have to sort them.
        if len(string) == 1: return string

        # Add the last symbol("$")
        string = string + [0]
        l = len(string)

        #=== Step1: L/S type ===
        Types = ["S"   for _ in range(l)]
        isLMS = [False for _ in range(l)]
        SA_dict = {0: 1} # Initialize with "$"
        for i in reversed(range(l-1)):
            if (string[i] > string[i+1]) or (string[i]==string[i+1] and Types[i+1] == "L"):
                Types[i] = "L"
                if Types[i+1] == "S":
                    isLMS[i+1] = True
            if string[i] in SA_dict.keys():
                SA_dict[string[i]] += 1
            else:
                SA_dict[string[i]] = 1

        #=== Step2: LMS ===
        LMS = LMS if LMS is not None else [i for i,e in enumerate(isLMS) if e]
        LMSn = len(LMS)

        #=== Step3: Initialize Suffix Array ===
        SA = [None for i in range(l)]

        #=== Step4: Make Buckets ===
        pos = 0
        N_buckets = len(SA_dict.keys()) # The number of buckets.
        # It is easy to have 2 lists which hold the bucket's end position,
        # because the LMS position was overwritten by after step. (step.6)
        b_start = [0 for i in range(N_buckets)] # Memorize where the each buckets start from. (for step.5)
        b_end   = [0 for i in range(N_buckets)] # Memorize where the each buckets end with. (for step.6)
        for k in sorted(SA_dict.keys()):
            b_start[k] = pos
            SA_dict[k] += pos
            b_end[k] = pos = SA_dict[k]

        #=== Step5: Sort (Not strictly) LMS ===
        # We don't know the LMS's correct order, so position them as it is.
        for i_LMS in reversed(LMS):
            SA_dict[string[i_LMS]] -= 1
            SA[SA_dict[string[i_LMS]]] = i_LMS

        #=== Step6: Sort L-type ===
        # @param i == "Suffix Array's index (i) which is positioned correctly (or temporary)"
        # If string[i-1] is L-type, position i-1 at the appropriate bucket from the "front".
        # We will "INDUCE" the proper position by using "sorted" LMS.
        for i in SA:
            if i and Types[i-1] == "L":
                SA[b_start[string[i-1]]] = i-1
                b_start[string[i-1]] += 1

        #=== Step7: Sort S-type ===
        # @param i == "Suffix Array's index (i) which is positioned correctly (or temporary)"
        # If string[i-1] is S-type, position i-1 at the appropriate bucket from the "back".
        # We will "INDUCE" the proper position by using "sorted" Suffix Array.
        for i in reversed(SA):
            if i and Types[i-1] == "S":
                b_end[string[i-1]] -= 1
                SA[b_end[string[i-1]]] = i-1

        #=== Step8: Compare LMS substrings ===
        # If we ordered LMS correctly, now we colud make the correct Suffix Array,
        if isLMSsorted:
            return SA[1:]

        # but if not, we have to sort LMS recursively. (Use the encoded number.)
        # So, we will check whether we ordered LMS (at Step5) correctly or not.
        # The encode. It starts with 0, but we will remove the last symbol("$"), so encode starts from 1.
        enc = 0;
        prev = None # The last LMS's position
        pLMS = {}
        for i in SA:
            if isLMS[i]: # Think only about LMS.
                for j in range(l):
                    # TODO: I simply compare two LMS substrings, but maybe use the relationships
                    #       to compare the LMS substrings more efficiently.
                    if not prev or string[i+j]!=string[prev+j]:
                        # We can know the size relationship between 2 substrings.
                        enc += 1;
                        prev = i
                        break
                    elif j>0 and (isLMS[i+j] or isLMS[prev+j]):
                        break
                pLMS[i] = enc-1

        if enc < LMSn:
            # If we couldn't order LMS correctly, sort LMS recursively.
            sub_string = [pLMS[e] for e in LMS if e<l-1] # We have to remove last symbol "$".
            order = StringSearch.SAIS(sub_string, isLMSsorted=False, LMS=None) # Sort LMS recursively.
            LMS = [LMS[i] for i in order]
        else:
            LMS = [i for i in SA if isLMS[i]]
        return StringSearch.SAIS(string, isLMSsorted=True, LMS=LMS)

    def _build_BWT(self):
        """ characters just to the left of the suffixes in the suffix array """
        self.BWT = np.asarray(list(self.string+"$"))[self.SA-1]

    def _build_C(self):
        """ the total occurrence of a characters which is smaller than x in lexicographical order in BWT."""
        self.C = {"$": 0}
        tmp = 1
        for cha in self.encoder.obj2cls.keys():
            self.C[cha] = tmp
            tmp += np.count_nonzero(self.BWT==cha)

    def _build_Occ(self):
        """ the total occurrence of a character x up to i-th in the BWT. """
        num_characters = len(self.encoder.obj2cls.keys())
        self.Occ = np.zeros(shape=(self.len+1, num_characters), dtype=int)
        tmp = np.zeros(shape=num_characters, dtype=int)
        for i,cha in enumerate(self.BWT):
            if cha != "$": tmp[self.encoder.obj2cls[cha]-1] += 1
            self.Occ[i] = tmp
        # The reason why there are all $0$ array in the last is "Occ(x,-1) = 0"

    def search(self, query):
        """ Scan through string looking for a match to the query.
        returning a all of the match indexes, or -1 if no match was found.
        """
        lb = 0;  ub=int(len(self.BWT)-1)
        q = ""
        for cha in reversed(query):
            if cha not in self.encoder.obj2cls.keys(): return -1
            q  = cha+q
            lb = int(self.C[cha] + self.Occ[lb-1][self.encoder.obj2cls[cha]-1])
            ub = int(self.C[cha] + self.Occ[ub][self.encoder.obj2cls[cha]-1] - 1)
            if lb>ub:
                return -1
        return self.SA[lb:ub+1]

    def where(self, query, width=60):
        len_query  = len(query)
        len_string = len(self.string)

        positions  = self.search(query)
        score = 0 if isinstance(positions, int) else len(positions)
        pos_info   = list(" " * len_string)
        if not isinstance(positions, int):
            for pos in positions:
                pos_info[pos:pos+len_query] = list("*" * len_query)
        pos_info = "".join(pos_info)
        printAlignment(
            sequences=[self.string, pos_info],
            indexes=[np.arange(len_string), np.arange(len_string)],
            seqname=['S', ' '],
            width=width,
            score=score
        )

    def lcp(self,i,j):
        """ Longest-Common-Prefix """
        l=0
        while (l+max(i,j)<self.len-1) and (self.string[i+l]==self.string[j+l]):
            l+=1
        return l

    def _build_height(self):
        """ Longest-Common-Prefix in Suffix Array """
        self.height = np.zeros(shape=self.len, dtype=int)
        h=0
        for i in range(self.len):
            j = self.ISA[i]
            if j == self.len-1:
                self.height[j] = -1
                h = 0
                continue
            k = self.SA[j+1]
            h=h-1 + self.lcp(i+h-1, k+h-1) if h>0 else self.lcp(i, k)
            self.height[j] = h
