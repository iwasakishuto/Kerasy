#coding: utf-8

from collections import deque

from ..utils import handleTypeError
from ..utils import NaiveTrieDOTexporter
from ..utils import DOTexporterHandler

def words2LOUDSbit(words):
    handleTypeError(types=[list], words=words)
    trie = NaiveTrie(words)
    bit_array, labels = trie.to_bit()
    return bit_array, labels

def trie_mining(node, curt=""):
    for val, child in node.children.items():
        next = curt + val
        if child.end:
            yield next
        yield from trie_mining(node=child, curt=next)

class Node():
    """ Node for Tree structure. """
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.end = False

    def __iter__(self):
        return iter(self.children.values())

    def add(self, child):
        self.children.update({child.value: child})

class NaiveTrie():
    def __init__(self, words=None):
        self.root = Node("BOS")
        if words is not None:
            if isinstance(words, list):
                self.build(self.root, words.pop(0))
                for word in words:
                    self.add(word)
            elif isinstance(words, str):
                self.build(self.root, words)
            else:
                handleTypeError(types=[list, str], words=words)

    def __contains__(self, word):
        node = self.root
        for w in word:
            node = node.children.get(w)
            if node is None:
                return False
        return True

    def __iter__(self):
        return iter(trie_mining(self.root, curt=""))

    def build(self, node, word):
        for w in word:
            child = Node(w)
            node.add(child)
            node = child
        node.end = True

    def add(self, word):
        node = self.root
        for i,w in enumerate(word):
            child = node.children.get(w)
            if child is None:
                self.build(node, word[i:])
                break
            node = child
        node.end = True

    def to_bit(self):
        bit_array = "10"
        labels = [None]

        queue = deque()
        queue.append(self.root)

        while (len(queue) != 0):
            node = queue.popleft() # O(1)
            labels.append(node.value)

            bit_array += "1"*len(node.children) + "0"
            for child in node.children.values():
                queue.append(child)
        return bit_array, labels

    def export_graphviz(self, out_file=None, filled=True, rounded=True, precision=3):
        exporter = NaiveTrieDOTexporter(filled=filled, rounded=rounded, precision=precision)
        return DOTexporterHandler(exporter, root=self.root, out_file=out_file)

class LOUDSTrieOffline():
    """
    By creating a trie tree with a LOUDS data structure, the amount of memory is
    compressed considerably. If you are not familier with LOUDS,
    please check this blog: https://takeda25.hatenablog.jp/entry/20120409/1333971789
    ~~~~
    ex) words = ["a", "at", "an", "and"]
     * bit_array = 10101100100
     * labels    = [None, '<BOS>', 'a', 't', 'n', 'd']
    NOTE:
        - node_id: one-based indexing ex.) `node_id` of self.root is 1.
        - n-th   : one-based indexing
        - labels : one-based indexing. labels[1] = self.root.value = ""
    """
    def __init__(self, words):
        self.bit_array, self.labels = words2LOUDSbit(words)

    def get_nth_target_bit_idx(self, n, target_bit="0"):
        """Returns the location of the nth target bit.
        @params n         : (int) one-based indexing.
        @params target_bit: (str) "0" or "1"
        """
        for i,bit in enumerate(self.bit_array):
            if bit == target_bit:
                n -= 1
                if n==0:
                    return i
        return -1

    def get_select(self, target_bit):
        return lambda n: self.select(n, target_bit)

    def get_rank(self, target_bit):
        """
        Returns the number of target bits from head to the position.
        @params position  : (int) zero-based indexing
        @params target_bit: (str) 0 or 1
        """
        return lambda position: self.bit_array[:position+1].count(target_bit)

    def traverse_children(self, node_id, char):
        """
        If the character is found among the children of current_node,
        this method returns the node number of the child, None otherwise.
        """
        # Get the index of where "0" appears `node_id` times. Then, traverse the
        # children of the current node by scanning to the right from there. For
        # the sake of brevity, add 1 to get the index of first sibling of the children.
        index = self.get_nth_target_bit_idx(n=node_id, target_bit="0") + 1
        child_node_id = self.bit_array[:index].count("1")
        while self.bit_array[index] == "1":
            child_node_id += 1 # 'cause self.bit_array[index].count("1")==1
            if self.labels[child_node_id] == char:
                return child_node_id
            index += 1
        return None

    def search(self, query):
        """
        @params query : (str)
        @return index : query's index on the bit array.
        """
        node_id = 1 # This means the root node.
        for w in query:
            node_id = self.traverse_children(node_id, w)
            if node_id is None:
                return None
        return self.get_nth_target_bit_idx(n=node_id, target_bit="1")

class LOUDSTrieOnline():
    def __init__(self):
        raise NotImplementedError("Not Implemented.")
