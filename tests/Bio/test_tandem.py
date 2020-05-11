# coding: utf-8
from kerasy.Bio.tandem import find_tandem
from kerasy.utils import generateSeq

len_sequences = 1000

def get_test_data():
    sequence = generateSeq(size=len_sequences,
                           nucleic_acid='DNA',
                           weights=None,
                           seed=123)
    sequence = "".join(sequence)
    return sequence

def test_find_tandem():
    sequence = get_test_data()

    max_val_sais, tandem_lists_sais = find_tandem(sequence, method="SAIS")
    tandem_sais = tandem_lists_sais[0]

    max_val_dp, tandem_lists_dp = find_tandem(sequence, method="DP")
    tandem_dp = tandem_lists_dp[0]

    assert max_val_sais == max_val_dp
    assert any([tandem_dp[i:]+tandem_dp[:i] == tandem_sais for i in range(len(tandem_dp))])
