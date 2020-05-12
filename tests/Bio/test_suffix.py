# coding: utf-8
import numpy as np
from kerasy.Bio.string import ALPHABETS
from kerasy.Bio.suffix import SAIS, BWT_create, reverseBWT

len_string = 100000

def get_test_data():
    rnd = np.random.RandomState(1)
    string = "".join(rnd.choice(ALPHABETS, len_string))
    return string

def test_SAIS():
    input = "abaabababbabbb"
    answer = np.asarray([14,  2,  0,  3,  5,  7, 10, 13,  1,  4,  6,  9, 12,  8, 11])
    SA = SAIS(input)
    assert np.all(SA == answer)

def test_burrows_wheeler_transform():
    string = get_test_data()
    SA = SAIS(string)
    bwt = BWT_create(string, SA)
    reversed_bwt = reverseBWT(bwt)
    assert reversed_bwt == string
