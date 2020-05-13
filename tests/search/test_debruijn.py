# coding: utf-8
import os
from kerasy.search.debruijn import kmer_deBruijnGraph
from kerasy.utils import generateSeq

len_sequences = 50
num_sequecen = 100
k = 45

def get_test_data():
    sequences = generateSeq(size=(num_sequecen, len_sequences),
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123)
    sequences = list(map(lambda x:"".join(x), sequences))
    return sequences

def test_kmer_debruijn(path="deBruijnGraph.png"):
    reads = get_test_data()
    model = kmer_deBruijnGraph(k=k)
    model.build(reads)

    assert model.export_graphviz(path)
    os.remove(path)
