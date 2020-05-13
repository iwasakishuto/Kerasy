# coding: utf-8
from kerasy.search.trie import NaiveTrie, LOUDSTrieOffline
from kerasy.utils import generateSeq

len_sequences = 50
num_sequecen = 100

def get_test_data():
    sequences = generateSeq(size=(num_sequecen, len_sequences),
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123)
    sequences = list(map(lambda x:"".join(x), sequences))
    return sequences



def test_louds_trie():
    sequences = get_test_data()
    louds_trie = LOUDSTrieOffline(sequences)
    assert all([seq in louds_trie for seq in sequences])

def test_naive_trie():
    sequences = get_test_data()
    naive_trie = NaiveTrie(sequences)
    assert all([seq in naive_trie for seq in sequences])
