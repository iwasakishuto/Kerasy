# coding: utf-8
from kerasy.datasets import ncbi

def test_get_seq():
    # SARS coronavirus, complete genome
    sequence = ncbi.getSeq("NC_004718")

    assert sequence is not None
