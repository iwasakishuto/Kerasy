"""The Reference Sequence (RefSeq) database is an open access, annotated and curated collection
of publicly available nucleotide sequences (DNA, RNA) and their protein products.
This database is built by National Center for Biotechnology Information (NCBI),
and, unlike GenBank, provides only a single record for each natural biological molecule
(i.e. DNA, RNA or protein) for major organisms ranging from viruses to bacteria to eukaryotes.
"""

import re
import requests
from ..utils import priColor
from ..utils import urlDecorate

def getSeq(refSeq_num, asfasta=False, path=""):
    """get Sequence from NCBI database.
    @params refSeq_num: (str) ex.NM_000147
        - NM: mRNA
        - NR: ncRNA
        - NP: Protein
    @params asfasta   : (bool) If true, add header like .fasta file.
    @params path      : (str)  If you specify, .fasta file will be created.
    """
    try:
        # refSeq_num2geneID.
        url = f'https://www.ncbi.nlm.nih.gov/nuccore/?term={refSeq_num}'
        print(urlDecorate(url, addDate=True))
        ret = requests.get(url)
        gene_id = re.findall(r'<meta name="ncbi_uidlist" content="(.*?)" /><meta name="ncbi_filter"', ret.text)[0]

        # geneID2sequences.
        url = f'https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id={gene_id}'
        print(urlDecorate(url, addDate=True))
        ret = requests.get(url)

        #=== Arange data. ===
        # Definition.
        s = re.search(r'DEFINITION', ret.text).end()
        e = re.search(r'ACCESSION', ret.text).start()
        definition = re.sub(r'\s+', r' ', ret.text[s:e])
        # Sequences.
        joint = "\n" if asfasta else ""
        e = re.search(r"ORIGIN.*", ret.text).end()
        pattern = r"\d+\s([a-z\s]+\n)" if asfasta else r"\d+\s([a-z\s]+)\n"
        seq_info   = re.findall(r"\d+\s([a-z\s]+)\n", ret.text[e:])
        seq_prepro = [seq.replace(" ", "") for seq in seq_info]
        sequences  = joint.join(seq_prepro)
        sequences  = f">{refSeq_num} {definition}\n" + sequences if asfasta else sequences

        # If necessary, output to file.
        if path:
            with open(path, mode='w') as f:
                f.write(sequences)
            print(f"Saved to {priColor.color(path, color='RED')}.")
        return sequences
    except:
        print("There's something wrong with the connection. Please try again later.")
        return None
