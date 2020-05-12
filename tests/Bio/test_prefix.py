# coding: utf-8
import numpy as np
from kerasy.utils import generateSeq
from kerasy.Bio.prefix import LCP_create, LP_create, LP_with_pattern_create

len_text = 100
len_pattern = 10

def get_test_data(len_sequences):
    sequence = generateSeq(size=len_sequences,
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123+len_sequences)
    sequence = "".join(sequence)
    return sequence

def test_LCP_create():
    # https://en.wikipedia.org/wiki/LCP_array
    input = "banana"
    answer = np.asarray([ 0,  1,  3,  0,  0,  2, -1])
    lcp_array = LCP_create(input)
    assert np.all(lcp_array == answer)

def test_LP_create():
    text = get_test_data(len_text)
    lp_array = LP_create(text)
    assert all([text[:lp] == text[i:i+lp] for i,lp in enumerate(lp_array)])

def test_LP_with_pattern_create():
    text = get_test_data(len_text)
    pattern = get_test_data(len_pattern)
    lp_with_pattern = LP_with_pattern_create(pattern, text)
    assert all([text[i:i+lp] == pattern[:lp] for i,lp in enumerate(lp_with_pattern)])
