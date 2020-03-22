# coding: utf-8
import numpy as np
from ..utils import CategoricalEncoder
from ..utils import printAlignment
from ..utils import inverse_arr
from ..utils import priColor

def checkString(string):
    if "$" in string:
        raise TypeError(
            "You do not need to add '$' at the end.",
            "If '$' is used in `string`, please replace it."
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
        self.BWT = mkBWT(string, self.SA)
        self.Occ = mkOcc(self.BWT, self.encoder.obj2cls)
        self.C   = mkC(self.BWT, self.encoder.obj2cls)

        print("Building Longest-Common-Prefix Array...")
        self.height = mkHeight(string, self.SA)

    def SuffixArray(self):
        """ Print Suffix Array. """
        digit = max(len(str(self.SA[0])), 2)
        print(f"{'i':>{digit}} {'SA':>{digit}} {'H':>{digit}} Suffix")
        print("-"*(digit*3+9))
        for i,pos in enumerate(self.SA):
            print(f"{i:>{digit}} {pos:>{digit}} {self.height[i]:>{digit}} {self.string[pos:]+'$'}")

    def search(self, query):
        """ Scan through string looking for a match to the query.
        returning a all of the match indexes, or -1 if no match was found.
        """
        lb = 0;  ub=int(len(self.BWT)-1)
        q = ""
        for cha in reversed(query):
            if cha not in self.encoder.obj2cls.keys(): return -1
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

def mkBWT(string, SA):
    """Make Burrows-Wheeler-Transform (BWT)
    @params string: Not include stop symbol '$'.
    @params SA    : Suffix Array.
    """
    checkString(string)
    bwt = np.asarray(list(string+"$"))[np.asarray(SA)-1]
    return bwt

def mkOcc(bwt, en_dict):
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

def mkC(bwt, en_dict):
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
    """ Reverse Original String only from BWT.
    @params bwt: (str, arr, list) Burrows-Wheeler-Transform.
    """
    encoder = CategoricalEncoder()
    encoder.to_categorical(bwt.replace("$", ""), origin=1)
    bwt = np.asarray(list(bwt))

    type_character = len(encoder.obj2cls.keys())
    num_character  = len(bwt)

    C = mkC(bwt, encoder.obj2cls)
    Occ = mkOcc(bwt, encoder.obj2cls)
    T = np.zeros(shape=num_character, dtype=int)
    for i in range(num_character):
        T[i] = 0 if bwt[i]=="$" else C[bwt[i]]+Occ[encoder.obj2cls[bwt[i]]-1][i]-1

    string = list(" " * num_character)
    bwt_idx = 0
    for i in reversed(range(-1, num_character-1)):
        string[i] = bwt[bwt_idx]
        bwt_idx = T[bwt_idx]
    return "".join(string)[:-1]

def lcp(string,i,j):
    """ Longest-Common-Prefix """
    l=0; n = len(string)
    while (l+max(i,j)<n) and (string[i+l]==string[j+l]): l+=1
    return l

def mkHeight(string, SA):
    """ Make Auxiliary data structure Height(i) which is for LCParray.
    @params string:
    @params SA    : Suffix Array.
    """
    checkString(string)
    length = len(SA)
    ISA = inverse_arr(SA)
    height = np.zeros(shape=length, dtype=int)
    h = 0
    for i in range(length):
        j = ISA[i]
        if j == length-1:
            height[j] = -1
            h = 0
            continue
        k = SA[j+1]
        h = h-1+lcp(string, i+h-1, k+h-1) if h>0 else lcp(string, i, k)
        height[j] = h
    return height

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

def simple_compression(string):
    """ Simple way to compress string. """
    symbol = ""; compressed_string = ""; n=1
    for cha in string:
        if cha==symbol: n+=1
        else:
            if n>1: compressed_string+=str(n)
            compressed_string+=cha
            symbol=cha
            n=1
    return compressed_string

def simple_decompression(compressed_string):
    """Decompression for `simple_compression(string)`"""
    string=""; i=0; len_string = len(compressed_string)
    while True:
        if compressed_string[i].isdigit():
            print(compressed_string[i])
            s = i
            while i<len_string and compressed_string[i].isdigit():
                i+=1
            n = int(compressed_string[s:i])
            string += string[-1]*(n-1)
        else:
            string+=compressed_string[i]
            i+=1
        if i==len_string:
            break
    return string

# TODE: "Move-to-front encoding" and "Run-length encoding".
