# coding: utf-8
import os
from kerasy.utils import generateSeq
from kerasy.Bio.alignment import (NeedlemanWunshGotoh,
                                  SmithWaterman,
                                  BackwardNeedlemanWunshGotoh,
                                  PairHMM)

len_sequences = 30

def get_test_data():
    seqX, seqY = generateSeq(size=(2, len_sequences),
                             nucleic_acid='DNA',
                             weights=None,
                             seed=123)
    return seqX, seqY

def _test_alignment(Model, path="alignment.json", **kwargs):
    seqX, seqY = get_test_data()
    model = Model(**kwargs)
    model.load_params()

    score = model.align_score(seqX, seqY, verbose=-1)
    model.save_params(path)

    model_ = Model(**kwargs)
    model_.load_params(path)
    score_ = model_.align_score(seqX, seqY, verbose=-1)

    os.remove(path)

    assert score == score_

def test_needleman_wunsh_gotoh():
    _test_alignment(NeedlemanWunshGotoh)

def test_smith_waterman():
    _test_alignment(SmithWaterman)

def test_backward_needleman_wunsh_gotoh():
    _test_alignment(BackwardNeedlemanWunshGotoh)

def test_pair_hmm():
    _test_alignment(PairHMM)
