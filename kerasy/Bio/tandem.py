# coding: utf-8
import numpy as np

from ..utils import handleKeyError
from ..clib import c_tandem

TANDEM_FINDING_ALGORITHMS = ["SAIS", "DP"]

def _find_tandem_DP(sequence):
    n = len(sequence)
    DP = np.zeros(shape=(n+1, n+1), dtype=np.int32)
    max_val, tandem_pos_lists = c_tandem._DPrecursion_Tandem(DP, sequence)
    tandem_lists = [sequence[s:e] for s,e in tandem_pos_lists]
    return max_val, tandem_lists

def _find_tandem_SAIS(sequence):
    max_val = 0
    tandem_lists = [(0,len(sequence))]
    #TODO: Implementation
    # - O(n^2)   LCP + naive
    # - O(nlogn) LCP + divide-and-conquer method
    # - O(n)     LCP + factorization
    return max_val, tandem_lists

def find_tandem(sequence, method="SAIS"):
    handleKeyError(lst=TANDEM_FINDING_ALGORITHMS, method=method)
    max_val, tandem_lists = {
        "DP"   : _find_tandem_DP,
        "SAIS" : _find_tandem_SAIS
    }[method](sequence)
    return max_val, tandem_lists
