# coding: utf-8
import numpy as np
from kerasy.Bio.string import ALPHABETS, StringSearch

num_charactors = 10
len_string = 100000
len_query = 5

def get_test_data():
    rnd = np.random.RandomState(1)
    string = "".join(rnd.choice(ALPHABETS[:num_charactors], len_string))
    query  = "".join(rnd.choice(ALPHABETS[:num_charactors], len_query))
    return string, query

def test_string_search():
    string, query = get_test_data()
    db = StringSearch()
    db.build(string)
    positions = db.search(query)
    if isinstance(positions, np.ndarray):
        # positions != -1
        assert any([string[pos:pos+len_query] == query for pos in positions])
