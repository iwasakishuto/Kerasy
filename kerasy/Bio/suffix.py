# coding: utf-8
import numpy as np
from ..utils import CategoricalEncoder

class SuffixArray():
    def build(self, string):
        self.encoder = CategoricalEncoder()
        self.string = string
        self.SA = self.SAIS(
            list(self.encoder.to_categorical(string, origin=1)),
            isLMSsorted=False,
            LMS=None
        )

    def summary(self):
        digit = len(str(self.SA[0]))
        for pos in self.SA:
            print(f"{pos:>{digit}}", self.string[pos:] + "$")


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
            order = SuffixArray.SAIS(sub_string, isLMSsorted=False, LMS=None) # Sort LMS recursively.
            LMS = [LMstring[i] for i in order]
        else:
            LMS = [i for i in SA if isLMS[i]]
        return SuffixArray.SAIS(string, isLMSsorted=True, LMS=LMS)
