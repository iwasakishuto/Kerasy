from __future__ import absolute_import

import numpy as np
from .generic_utils import priColor

def is_bp(self, base_i, base_j, nucleic_acid="DNA", WatsonCrick=True, Wobble=False):
    """ Check is the 2 bases form a base pair.
    @params base        : (str)  Single letter base.
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
    bases = base_i+base_j
    flag = False
    if WatsonCrick:
        WC_1 = ("A" in bases) and ("T" in bases) if nucleic_acid=="DNA" else ("A" in bases) and ("U" in bases)
        WC_2 = ("G" in bases) and ("C" in bases)
        flag = flag or (WC_1 or WC_2)
    if Wobble:
        Wob_1  = ("G" in bases) and ("U" in bases)
        flag = flag or Wob
    return flag

def arangeFor_printAlignment(sequence, idxes, blank="-"):
    """Rewrite sequence information for `printAlignment` function.
    @params sequence: (str)
    @params idxes   : (ndarray) the information whether sequence[idx] aligned.
    """
    masks = np.logical_or((np.roll(idxes, 1) == idxes), idxes==-1)
    aligned_seq = "".join(blank if mask else sequence[idx] for mask,idx in zip(masks, idxes))
    return aligned_seq

def printAlignment(sequences, indexes, score, model="", width=60, blank="-"):
    """print Alignment Result.
    @params sequences: (list) Raw sequences.
    @params idxes    : (list) indexes. 0-origin indexes. -1 means the 'deletion' at the header.
    """
    if not isinstance(sequences, list): aligned_seqs = list(aligned_seqs)
    if not isinstance(indexes, list): idxes = list(idxes)
    if len(sequences) != len(indexes):
        raise ValueError("`sequences` and `indexes` must be the same length, meaning that same number of sequences.")

    # Prepare for printing.
    digit = len(str(len(max(sequences, key=len))))
    aligned_seqs = [arangeFor_printAlignment(seq,idxes,blank=blank) for (seq, idxes) in zip(sequences, indexes)]
    aligned_length = len(aligned_seqs[0])

    if model: print(f"Model: {priColor.color(model, color='ACCENT')}")
    score = score if isinstance(score, int) else f"{score:.3f}"
    print(f"Alignment score: {priColor.color(score, color='RED')}\n")
    print("=" * (9+2*digit+width))
    print(
        "\n\n".join([
            "\n".join([
                # Handling for idxes[0]==-1, and pos+width>aligned_length-1
                f"{chr(65+i)}: [{max(0,idxes[pos]):>0{digit}}] {aligned_seq[pos: pos+width]} [{idxes[min(pos+width, aligned_length-1)]:>0{digit}}]" for i,(idxes,aligned_seq) in enumerate(zip(indexes, aligned_seqs))
            ]) for pos in range(0, aligned_length, width)
        ])
    )
    print("=" * (9+2*digit+width))

def readMonoSeq(path, header=True):
    with open(path, "r") as f:
        seqfile = f.readlines()
    sequence = ["".join(seqfile[header:]).replace('\n', '')]
    return sequence

def readMultiSeq(path, symbol='>'):
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
