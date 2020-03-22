# coding: utf-8
"""
This file contains the following algorithms:
- Suffix Array Induced Sorting (SA-IS)
    ref: https://ieeexplore.ieee.org/document/5582081
- Burrows Wheeler Transform (BWT)
- Reverse Burrows Wheeler Transform (reverseBWT)
- Auxiliary Data Structure (Useful for string search.)
    - C[x]     : the total occurrence of a characters which is
                 smaller than x in lexicographical order in BWT.
    - Occ(x,i) : the total occurrence of a character x up to i-th in the BWT.

[example] string = "abaabababbabbb"

     r SA Suffix           LCP
    ---------------------------
     0 14 $                 0
     1  2 aabababbabbb$     1
     2  0 abaabababbabbb$   3
     3  3 abababbabbb$      4
     4  5 ababbabbb$        2
     5  7 abbabbb$          3
     6 10 abbb$             0
     7 13 b$                1
     8  1 baabababbabbb$    2
     9  4 bababbabbb$       3
    10  6 babbabbb$         4
    11  9 babbb$            1
    12 12 bb$               2
    13  8 bbabbb$           2
    14 11 bbb$             -1
    ---------------------------
"""
import numpy as np

from ..utils import CategoricalEncoder
from ..utils import inverse_arr

def checkString(string):
    if "$" in string:
        raise TypeError(
            "You do not need to add '$' at the end.",
            "If '$' is in `string`, please replace them."
        )

################
# Suffix Array #
################

def SAIS(string):
    """ Suffix Array Induced Sorting Handler. """
    checkString(string)
    if isinstance(string, str):
        encoder = CategoricalEncoder()
        string = list(encoder.to_categorical(string, origin=1))
    SA = SAIS_recursion(string, isLMSsorted=False, LMS=None)
    return np.asarray(SA, dtype=int)

def SAIS_recursion(string, isLMSsorted=False, LMS=None):
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
        order = SAIS_recursion(sub_string, isLMSsorted=False, LMS=None) # Sort LMS recursively.
        LMS = [LMS[i] for i in order]
    else:
        LMS = [i for i in SA if isLMS[i]]
    return SAIS_recursion(string, isLMSsorted=True, LMS=LMS)

#############################
# Burrows-Wheeler-Transform #
#############################

def BWT_create(string, SA):
    """Make Burrows-Wheeler-Transform (BWT)
    @params string: Not include stop symbol '$'.
    @params SA    : Suffix Array.
    """
    checkString(string)
    bwt = np.asarray(list(string+"$"))[np.asarray(SA)-1]
    return bwt

def Occ_create(bwt, en_dict):
    """ Make Auxiliary data structure Occ(x, i)
    It memorizes the total occurrence of a character x up to i-th in the BWT.
    @params bwt    : (ndarray) Burrows-Wheeler-Transform.
    @params en_dict: (dict) encoder like `self.encoder.obj2cls`. It must be 1-orizin.
    """
    type_character = len(en_dict.keys())
    num_character  = len(bwt)
    # Initialization.
    Occ = np.zeros(shape=(type_character, num_character+1), dtype=int)
    tmp = np.zeros(shape=(type_character), dtype=int)
    for i,cha in enumerate(bwt):
        if cha != "$":
            tmp[en_dict[cha]-1] += 1 # -1 because 1-origin.
        Occ[:,i] = tmp
    # The reason why there are all $0$ array in the last is "Occ(x,-1) = 0"
    return Occ

def C_create(bwt, en_dict):
    """ Make Auxiliary data structure C(x)
    it memorizes the total occurrence of a characters which is smaller than x in lexicographical order in BWT.
    @params bwt    : (ndarray) Burrows-Wheeler-Transform.
    @params en_dict: (dict) encoder like `self.encoder.obj2cls`. It must be 1-orizin.
    """
    C = {"$": 0}
    tmp = 1
    for cha in en_dict.keys():
        C[cha] = tmp
        tmp+=np.count_nonzero(bwt==cha)
    return C

def reverseBWT(bwt):
    """ reverse Burrows Wheeler Transform
    @params bwt    : (str, arr, list) Burrows-Wheeler-Transform.
    @return string : Original String
    """
    encoder = CategoricalEncoder()
    encoder.to_categorical(bwt.replace("$", ""), origin=1)
    bwt = np.asarray(list(bwt))

    type_character = len(encoder.obj2cls.keys())
    num_character  = len(bwt)

    C = C_create(bwt, encoder.obj2cls)
    Occ = Occ_create(bwt, encoder.obj2cls)
    T = np.zeros(shape=num_character, dtype=int)
    for i in range(num_character):
        T[i] = 0 if bwt[i]=="$" else C[bwt[i]]+Occ[encoder.obj2cls[bwt[i]]-1][i]-1

    string = list(" " * num_character)
    bwt_idx = 0
    for i in reversed(range(-1, num_character-1)):
        string[i] = bwt[bwt_idx]
        bwt_idx = T[bwt_idx]
    return "".join(string)[:-1]
