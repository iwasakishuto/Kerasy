# coding: utf-8
import numpy as np
from kerasy.utils import generateSeq
from kerasy.Bio.maxsets import MSS

len_array = 1000

def get_test_data():
    rnd = np.random.RandomState(123)
    R_int   = rnd.randint(low=-10, high=100, size=len_array)
    R_float = rnd.uniform(low=-10, high=100, size=len_array)
    return R_int, R_float

def _test_maxsets(R, limits=(5,10)):
    model = MSS()
    for i,limit in enumerate(sorted(limits)):
        model.run(R, limit=limit, verbose=-1)
        score = model.score
        assert i==0 or prev_score >= score
        prev_score = score

def test_maxsets_int():
    R_int, _ = get_test_data()
    _test_maxsets(R_int, limits=(5,10))

def test_maxsets_float():
    _, R_float = get_test_data()
    _test_maxsets(R_float, limits=(5,10))
