# coding: utf-8
import os
import numpy as np
from kerasy.search.itemset import FrequentSet
from kerasy.search.itemset import create_one_hot

num_transactions = 100
num_data = 1000

def get_test_data():
    data_idx = np.arange(num_data)

    rnd = np.random.RandomState(123)
    retail = [rnd.choice(a=data_idx, size=size, replace=False) for size in rnd.randint(1, num_data/10, size=num_transactions)]

    database, idx2data = create_one_hot(retail)
    return database, idx2data


def test_all_itemset(path="itemset.png", threshold=10):
    database, idx2data = get_test_data()

    model = FrequentSet(threshold=threshold)
    model.fit(database, method="all")

    assert model.export_graphviz(path, class_names=idx2data)

    os.remove(path)

def test_closed_itemset(path="itemset.png", threshold=10):
    database, idx2data = get_test_data()

    model = FrequentSet(threshold=threshold)
    model.fit(database, method="closed")

    assert model.export_graphviz(path, class_names=idx2data)

    os.remove(path)
