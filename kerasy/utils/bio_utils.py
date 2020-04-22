# coding: utf-8
from __future__ import absolute_import

import numpy as np
from .generic_utils import priColor

NUCLEIC_ACIDS_CREATOR = {
    "DNA" : ["A","C","G","T"],
    "dna" : ["a","c","g","t"],
    "RNA" : ["A","C","G","U"],
    "rna" : ["a","c","g","u"],
}

def bpHandler(bp2id=None, nucleic_acid="DNA", WatsonCrick=True, Wobble=False):
    """ Make the function which checks 'whether 2 bases form a base pairs' or 'the energy of them'.
    @params bp2id       : (list) Describes the relationship between base pairs and the index.
    @params nucleic_acid: (str)  DNA: deoxyribonucleic acid, RNA: ribonucleic acid.
    @params WatsonCrick : (bool)
        - In canonical Watson–Crick base pairing in DNA,
            - adenine (A) forms a base pair with thymine (T) using 2 hydrogen bonds.
            - guanine (G) forms a base pair with cytosine (C) using 3 hydrogen bonds.
        - In canonical Watson–Crick base pairing in RNA,
            - thymine (T) is replaced by uracil (U).
    @params Wobble      : (bool)
        - A wobble base pair is a pairing between two nucleotides in RNA molecules that
          does not follow Watson-Crick base pair rules. The four main wobble base pairs are:
            - guanine (G) forms a base pair with uracil (U) using 2 hydrogen bonds.
            - hypoxanthine (I) forms a base pair with uracil (U) using 2 hydrogen bonds.
            - hypoxanthine (I) forms a base pair with adenine (A) using 2 hydrogen bonds.
            - hypoxanthine (I) forms a base pair with cytosine (C) using 2 hydrogen bonds.
            ※ hypoxanthine is the nucleobase of inosine.
    """
    if bp2id is not None:
        cond_expression = "".join([f'{i} if ("{b1}" in x and "{b2}" in x) else ' for i,(b1,b2) in enumerate(bp2id)]) + f"{len(bp2id)}"
        cond_branch = lambda x: eval(cond_expression)
    else:
        if nucleic_acid=="DNA":
            pair = (("A","T"),("G","C"))
        elif nucleic_acid=="RNA":
            pair = []
            if WatsonCrick:
                pair.extend([("A","U"),("G","C")])
            if Wobble:
                pair.append(("G","U"))
        cond_branch = lambda x: any([b1 in x and b2 in x for (b1,b2) in pair])
    def func(base_i, base_j):
        bases = base_i+base_j
        return cond_branch(bases)
    return func

def arangeFor_printAlignment(sequence, idxes, blank="-"):
    """Rewrite sequence information for `printAlignment` function.
    @params sequence: (str)
    @params idxes   : (ndarray) the information whether sequence[idx] aligned.
    """
    masks = np.logical_or((np.roll(idxes, 1) == idxes), idxes==-1)
    aligned_seq = "".join(blank if mask else sequence[idx] for mask,idx in zip(masks, idxes))
    return aligned_seq

def printAlignment(sequences, indexes, score, scorename="Alignment score", add_info="", seqname=None, model="", width=60, blank="-"):
    """print Alignment Result.
    @params sequences: (list) Raw sequences.
    @params idxes    : (list) indexes. 0-origin indexes. -1 means the 'deletion' at the header.
    @params seqname  : (list) If not specified, sequence name will be chr(65+i).
    """
    if not isinstance(sequences, list): sequences = list(sequences)
    if not isinstance(indexes, list): idxes = list(idxes)

    if seqname is None: seqname = [chr(65+i) for i in range(len(sequences))]
    if not isinstance(seqname, list): seqname = list(seqname)

    if len(sequences) != len(indexes):
        raise ValueError("`sequences` and `indexes` must be the same length, meaning that same number of sequences.")

    # Prepare for printing.
    digit = len(str(len(max(sequences, key=len))))
    aligned_seqs = [arangeFor_printAlignment(seq,idxes,blank=blank) for (seq, idxes) in zip(sequences, indexes)]
    aligned_length = len(aligned_seqs[0])

    if model: print(f"Model: {priColor.color(model, color='ACCENT')}")
    score = score if isinstance(score, int) else f"{score:.3f}"
    print(f"{scorename}: {priColor.color(score, color='RED')}\n{add_info}")
    print("=" * (9+2*digit+width))
    print(
        "\n\n".join([
            "\n".join([
                # Handling for idxes[0]==-1, and pos+width>aligned_length-1
                f"{seqname[i]}: [{max(0,idxes[pos]):>0{digit}}] {aligned_seq[pos: pos+width]} [{idxes[min(pos+width, aligned_length-1)]:>0{digit}}]" for i,(idxes,aligned_seq) in enumerate(zip(indexes, aligned_seqs))
            ]) for pos in range(0, aligned_length, width)
        ])
    )
    print("=" * (9+2*digit+width))

def read_fastseq(path, symbol='>'):
    with open(path, "r") as f:
        sequences=[]
        seq=""
        for line in f:
            if line[0]==symbol:
                if len(seq)==0: continue
                sequences.append(seq)
                seq=""
            else:
                seq+=line.rstrip('\n')
        sequences.append(seq)
    return sequences

def kmer_create(string,k):
    """ Disassemble to k-mer. """
    return [string[i:k+i] for i in range(len(string)-k+1)]
