# coding: utf-8
import numpy as np

from ..utils import handleKeyError
from ..utils import flatten_dual
from ..utils import ItemsetTreeDOTexporter
from ..utils import DOTexporterHandler

ITEM_MINING_METHODS = ["all", "closed"]

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
    """ Node for Tree structure.
    @params database  :
    @params itemset   :
    @params num_items :
    @params freq      :
    @params tail      :
    @params threshold :
    """
    def __init__(self, itemset, freq, tail):
        self.itemset = itemset
        self.freq = freq # (=len(database))
        self.tail = tail
        self.children = []

    def _recurse_all(self, database, threshold, num_items):
        """ Find ALL closed Itemsets. """
        for i in range(self.tail+1, num_items):
            next_itemset = self.itemset + [i]
            next_data = database[database[:,i]==1,:]
            freq = len(next_data)
            if freq >= threshold:
                child = Node(itemset=next_itemset, freq=freq, tail=i)
                child._recurse_all(next_data, threshold, num_items)
                self.children.append(child)

    def _recurse_closed(self, database, threshold, num_items):
        """ Find ONLY closed Itemsets. """
        for i in range(self.tail+1, num_items):
            next_data = database[database[:,i]==1,:]
            freq = len(next_data)
            if freq >= threshold:
                add_itemset = i+np.where(np.all(next_data[:,i:], axis=0))[0]
                next_itemset = self.itemset + add_itemset.tolist()
                child = Node(itemset=next_itemset, freq=freq, tail=max(add_itemset))
                child._recurse_closed(next_data, threshold, num_items)
                self.children.append(child)

class FrequentSet():
    def __init__(self, threshold):
        self.root = None
        self.threshold = threshold
        self.freq_sets = []

    def fit(self, database, method="closed"):
        """
        @param database: Binary Matrix. shape=(num_transactions, num_items)
        """
        method = method.lower()
        handleKeyError(lst=ITEM_MINING_METHODS, method=method)
        num_transactions, num_items = database.shape
        self.root = Node(itemset=[], freq=num_transactions, tail=-1)
        self.root.__getattribute__({
            "all"    : "_recurse_all",
            "closed" : "_recurse_closed",
        }[method]).__call__(database, self.threshold, num_items)
        self.num_items = num_items
        self.all = self.get_itemsets(self.root)

    def get_itemsets(self, node):
        freq_sets = [node.itemset]
        for child in node.children:
            freq_sets.extend(self.get_itemsets(node=child))
        return freq_sets

    def export_graphviz(self, out_file=None, feature_names=None,
                        class_names=None, cmap="jet", filled=True,
                        rounded=True, precision=3):
        if class_names is None:
            class_names = np.arange(self.num_items)
        exporter = ItemsetTreeDOTexporter(
            cmap=cmap, class_names=class_names,
            filled=filled, rounded=rounded, precision=precision
        )
        return DOTexporterHandler(exporter, root=self.root, out_file=out_file)
