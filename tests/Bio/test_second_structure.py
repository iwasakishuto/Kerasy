# coding: utf-8
import os
from kerasy.Bio.second_structure import Nussinov, Zuker
from kerasy.utils import generateSeq

len_sequences = 100

def get_test_data():
    sequence = generateSeq(size=len_sequences,
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123)
    sequence = "".join(sequence)
    return sequence

def test_nussinov():
    sequence = get_test_data()
    model = Nussinov()
    model.load_params()

    score, structure_info = model.predict(sequence, verbose=-1, ret_val=True)
    stack = []
    for i,symbol in enumerate(structure_info):
        if symbol == "(":
            stack.append(i)
        elif symbol == ")":
            assert model.is_bp(sequence[stack.pop(-1)], sequence[i])

def test_zuker():
    sequence = get_test_data()
    model = Zuker()
    model.load_params()

    score, structure_info = model.predict(sequence, verbose=-1, ret_val=True)
