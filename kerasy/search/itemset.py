# coding: utf-8
import numpy as np

class Node():
    N = None
    M = None
    threshold = None

    def __init__(self, itemset, frq, tail, shape=None, threshold=None):
        self.itemset = itemset
        self.frq  = frq
        self.tail = tail
        self.sons = []

        if shape is not None:
            Node.N, Node.M = shape
            Node.threshold = threshold

    def deepen(self, data):
        for i in range(self.tail+1, Node.M):
            next_itemset = np.append(self.itemset,i) if self.itemset is not None else np.array([i])
            next_data = data[np.all(data[:,next_itemset]==1, axis=1)]
            frq = len(next_data)
            if frq >= Node.threshold:
                son = Node(itemset=next_itemset, frq=frq, tail=i)
                son.deepen(next_data)
                self.sons.append(son)

class FrequentSet():
    def __init__(self, threshold):
        self.root = None
        self.threshold = threshold
        self.all = []

    def fit(self, database):
        """
        @param database: Binary Matrix.(ndarray) shape=(N,M)
            - N: The number of transactions.
            - M: The number of items(elements).
        """
        self.root = Node(itemset=None, frq=len(database), tail=-1, shape=database.shape, threshold=self.threshold)
        self.root.deepen(data=database)
        self.all = []
        self.get_all_frequentset(self.root)

    def get_all_frequentset(self, node):
        self.all.append(node.itemset)
        for son in node.sons:
            self.get_all_frequentset(node=son)
