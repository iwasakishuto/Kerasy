# coding: utf-8
import os
import numpy as np
from ..utils import flatten_dual
from ..utils import ItemsetTreeDOTexporter

def create_one_hot(data):
    """
    Create the one-hot binary matrix.
    @params data     : (list) Each element of data (data[i]) have variable length items.
    @return one_hot  : shape=(num_data,num_unique)
        one_hot[n][i]=1 means data[n] contains idx2data[i]
    @return idx2data : Dictionary from index to original data.
    """
    unique_data = sorted(list(set(flatten_dual(data))))
    num_unique = len(unique_data)
    num_data = len(data)
    data2idx = dict(zip(unique_data, range(num_unique)))
    one_hot = np.zeros(shape=(num_data,num_unique), dtype=np.int32)
    for i,row in enumerate(data):
        one_hot[i, np.asarray([data2idx[e] for e in row], dtype=int)]=1
    idx2data = dict(zip(range(num_unique), unique_data))
    return one_hot, idx2data

class Node():
    num_items = 0
    node_id = 0
    """ Node for Tree structure.
    @params database  :
    @params itemset   :
    @params num_items :
    @params freq      :
    @params tail      :
    @params threshold :
    """
    def __init__(self, database, freq, threshold, itemset, tail):
        self.itemset = itemset
        self.freq = freq # (=len(database))
        self.tail = tail
        self.sons = []
        self.deepen(database, threshold)

    def deepen(self, database, threshold):
        for i in range(self.tail+1, Node.num_items):
            next_itemset = self.itemset + [i]
            next_data = database[database[:,i]==1,:]
            freq = len(next_data)
            if freq >= threshold:
                son = Node(
                    database=next_data, freq=freq, threshold=threshold,
                    itemset=next_itemset, tail=i,
                )
                self.sons.append(son)

class FrequentSet():
    def __init__(self, threshold):
        self.tree = None
        self.threshold = threshold
        self.freq_sets = []

    def fit(self, database):
        """
        @param database: Binary Matrix. shape=(num_transactions, num_items)
        """
        num_transactions, Node.num_items = database.shape
        self.tree = Node(
            database=database, freq=num_transactions, threshold=self.threshold,
            itemset=[], tail=-1
        )
        self.freq_sets = []
        self._get_all_frequentset(self.tree)

    def _get_all_frequentset(self, node):
        self.freq_sets.append(node.itemset)
        for son in node.sons:
            self._get_all_frequentset(node=son)

    def export_graphviz(self, out_file=None, feature_names=None,
                        class_names=None, cmap="jet", filled=True,
                        rounded=True, precision=3):
        exporter = ItemsetTreeDOTexporter(
            cmap=cmap, feature_names=None, class_names=None,
            filled=filled, rounded=rounded, precision=precision
        )
        if out_file is not None:
            ext = os.path.splitext(os.path.basename(out_file))[-1]
            if ext==".png":
                return exporter.write_png(self.tree, path=out_file)
            elif ext==".dot":
                return exporter.export(self.tree, out_file=out_file)
        return exporter.export(self.tree, out_file=None)
