# coding: utf-8
import numpy as np

from ..utils import handleKeyError
from .string import StringSearch
from ..clib import c_tandem
from ..clib import c_prefix

TANDEM_FINDING_ALGORITHMS = ["SAIS", "DP"]

def _find_tandem_DP(sequence):
    n = len(sequence)
    DP = np.zeros(shape=(n+1, n+1), dtype=np.int32)
    max_val, tandem_pos_lists = c_tandem._DPrecursion_Tandem(DP, sequence)
    tandem_lists = [sequence[s:e] for s,e in tandem_pos_lists]
    return max_val, tandem_lists

def _find_tandem_SAIS(sequence):
    db = StringSearch(sequence, verbose=-1)
    tandem_lists = c_tandem._SAIS_Tandem(db.LZ_factorization + ["$"])
    scores = [db.calc_tandem_score(tandem) for tandem in tandem_lists]
    max_val = max(scores)
    tandem_lists = [tandem for tandem,score in zip(tandem_lists,scores) if score==max_val]
    return max_val, tandem_lists

def find_tandem(sequence, method="SAIS"):
    handleKeyError(lst=TANDEM_FINDING_ALGORITHMS, method=method)
    max_val, tandem_lists = {
        "DP"   : _find_tandem_DP,
        "SAIS" : _find_tandem_SAIS
    }[method](sequence)
    return max_val, tandem_lists
