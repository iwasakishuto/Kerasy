# coding: utf-8
import os
from kerasy.Bio.second_structure import Nussinov, Zuker

len_sequences = 100

def get_test_data():
    sequence = generateSeq(size=len_sequences,
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123)
    sequence = "".join(sequence)
    return sequence

def _test_secondary_structure(Model, path="secondary_structure.json", **kwargs):
    sequences = get_test_data()
    model = Model(**kwargs)
    model.load_params()

    score = model.predict(sequence, only_score=True, verbose=-1)
    model.save_params(path)

    model_ = Model(**kwargs)
    model_.load_params(path)
    score_ = model_.predict(sequence, only_score=True, verbose=-1)

    os.remove(path)

    assert score == score_

def test_nussinov():
    _test_secondary_structure(Nussinov)

def test_zuker():
    _test_secondary_structure(Zuker)
