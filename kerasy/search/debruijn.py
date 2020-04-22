# coding: utf-8
import os
import pydotplus

from ..utils import handleKeyError
from ..utils import kmer_create
from ..utils import flatten_dual
from ..utils.bio_utils import NUCLEIC_ACIDS_CREATOR

from .bloom import BloomFilter
from .trie import NaiveTrie

class kmer_deBruijnGraph():
    def __init__(self, k, nucleic_acid="DNA"):
        self.k = k
        self.nucleic_acid = nucleic_acid
        self.make_kmer_funcs(k, nucleic_acid)

    def make_kmer_funcs(self, k, nucleic_acid):
        handleKeyError(lst=list(NUCLEIC_ACIDS_CREATOR.keys()), nucleic_acid=nucleic_acid)

        bases = NUCLEIC_ACIDS_CREATOR.get(nucleic_acid)
        def kmer_right_extentions(kmer):
            return [kmer[1:] + nuc for nuc in bases]

        def kmer_left_extentions(kmer):
            return [nuc + kmer[:-1] for nuc in bases]

        def kmer_extentions(kmer):
            return kmer_left_extentions(kmer)+kmer_right_extentions(kmer)

        self.kmer_right_extentions = kmer_right_extentions
        self.kmer_left_extentions = kmer_left_extentions
        self.kmer_extentions = kmer_extentions

    def build(self, reads):
        self.num_reads = len(reads)
        self.ave_read_length = sum([len(read) for read in reads])/self.num_reads
        self.dot_data = "digraph DeBruijnGraph {\n "
        kmer_reads_list = [kmer_create(read, self.k) for read in reads]
        inits, bf, trie = self._memorize_kmer_reads(kmer_reads_list)
        cFP = self._construct_critical_FalsePositive(kmer_reads_list, bf, trie)
        for node in trie:
            self.dot_data += f"\t{node} [label=<{node}>] ;\n"
        self.dot_data += "}"

        self.inits = inits
        self.bf    = bf
        self.trie  = trie
        self.cFP   = cFP

    def _memorize_kmer_reads(self, kmer_reads_list):
        inits = set([kmer_reads[0] for kmer_reads in kmer_reads_list])
        bf = BloomFilter(capacity=len(flatten_dual(kmer_reads_list)))
        trie = NaiveTrie()
        # Memorize
        for kmer_reads in kmer_reads_list:
            for i,kmer in enumerate(kmer_reads):
                if i>0:
                    self.dot_data += f"\t{prev_kmer}->{kmer} ;\n"
                    if kmer in inits:
                        inits.remove(kmer)
                bf.add(kmer)
                trie.add(kmer)
                prev_kmer = kmer

        return (inits, bf, trie)

    def _construct_critical_FalsePositive(self, kmer_reads_list, bf, trie):
        cFP = []
        for kmer_reads in kmer_reads_list:
            for kmer in kmer_reads:
                for e in self.kmer_extentions(kmer):
                    if bf.has(e) and e not in trie:
                        cFP.append(e)
        return cFP

    def assemble(self, init, max_len=None):
        max_len = int(max_len or self.num_reads*self.ave_read_length/15)
        seqs = [init]
        result_trail = []
        k = self.k
        bf = self.bf
        cFP = self.cFP
        while len(seqs)>0:
            find_next = False
            seq = seqs.pop()
            if len(seq)>max_len:
                result_trail.append(seq)
            else:
                for e in self.kmer_right_extentions(seq[-k:]):
                    if e not in cFP and e in bf:
                        find_next = True
                        seqs.append(seq + e[-1])
                if not find_next:
                    result_trail.append(seq)
        return result_trail

    def export_graphviz(self, out_path):
        ext = os.path.splitext(os.path.basename(out_path))[-1]
        if ext==".png":
            graph = pydotplus.graph_from_dot_data(self.dot_data)
            graph.write_png(out_path, f='png', prog='dot')
        elif ext==".dot":
            with open(out_path, mode="w") as f:
                f.write(self.dot_data)
        else:
            handleKeyError(
                lst=[".png", ".dot"], out_path=ext,
                msg_="Extension type is not accepted"
            )
