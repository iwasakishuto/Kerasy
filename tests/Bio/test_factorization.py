# coding: utf-8
from kerasy.utils import generateSeq
from kerasy.Bio.factorization import simple_compression, simple_decompression

len_sequences = 100

def get_test_data():
    sequence = generateSeq(size=len_sequences,
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123)
    sequence = "".join(sequence)
    return sequence

def test_simple_compression():
    sequence = get_test_data()
    compressed_string = simple_compression(sequence)
    assert len(sequence) >= len(compressed_string)

    decompressed_string = simple_decompression(compressed_string)
    assert decompressed_string == sequence
