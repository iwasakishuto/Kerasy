# coding: utf-8
from kerasy.search.bloom import BloomFilter
from kerasy.utils import generateSeq

len_sequences = 50
num_sequecen = 1000

def get_test_data():
    reads = generateSeq(size=(num_sequecen, len_sequences),
                        nucleic_acid='DNA',
                        weights=None,
                        seed=123)
    reads = list(map(lambda x: "".join(x), reads))
    return reads

def test_bloom_filter():
    reads = get_test_data()
    bf = BloomFilter(num_sequecen, error_rate=0.001)
    for read in reads:
        bf.add(read)

    assert all([read in bf for read in reads])
