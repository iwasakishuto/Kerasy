# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import flush_progress_bar
from ..utils import Table

class SCODE():
    """ ref: https://academic.oup.com/bioinformatics/article/33/15/2314/3100331
    SCODE: an efficient regulatory network inference algorithm
           from single-cell RNA-Seq during differentiation
    """
    def __init__(self, lamda=1e-2, b_min=-10, b_max=2, random_state=None):
        self.lamda = lamda
        self.b_min = b_min
        self.b_max = b_max
        self.seed = random_state

    def fit(self, X, t, dimension=4, max_iter=100, verbose=1):
        """
        @params X         : expressions. shape=(num_gene, num_cell)
        @params t         : Time when each cell was obtained. shape=(num_cell)
        @params dimension : How many dimensions to reduce.
        """
        rnd = np.random.RandomState(self.seed)
        num_gene, num_cell = X.shape
        b_min, b_max = (self.b_min, self.b_max)
        lamda = self.lamda
        # Initialization.
        b = np.random.uniform(low=b_min, high=b_max, size=dimension)

        for it in range(max_iter):
            d = rnd.randint(dimension)
            b[d] = rnd.uniform(low=b_min, high=b_max, size=1)
            Z = np.power(np.e, (np.expand_dims(t, axis=1) * np.expand_dims(b, axis=0)))
            # Solution of linear regression (X^{(e)}≃WZ^{(e)})
            W = np.dot(np.dot(X,Z), np.linalg.inv(np.dot(Z.T,Z) + np.eye(dimension)*lamda))
            # Residual sum of squares.
            RSS = np.linalg.norm(X-np.dot(W,Z.T))
            if it==0 or RSS < RSS_best:
                b_best = b
                W_best = W
                RSS_best = RSS
            else:
                b = b_best
            flush_progress_bar(it, max_iter, metrics={"RSS": RSS_best}, verbose=verbose)

        if verbose>0: print()
        B_best = np.diag(b_best)
        self.b = b_best
        self.B = B_best
        self.W = W_best
        self.A = np.dot(W_best, np.dot(B_best, np.dot(np.linalg.inv(np.dot(W_best.T, W_best)), W_best.T)))
        self.RSS = RSS_best

    def predict(self, time):
        Z = np.power(np.e, (np.expand_dims(time,axis=1)*np.expand_dims(self.b,axis=0)))
        return np.dot(self.W, Z.T)

    def strong_pairs(self, top=1, reverse=False):
        A = self.A
        # Arrange in order of strong correlation.
        ranks = np.argsort(np.abs(np.ravel(A)))[::-1]
        if reverse:
            ranks = ranks[::-1]
        regulated, regulate = np.unravel_index(ranks, A.shape)
        return (regulate[:top], regulated[:top])

    def top_pairs(self, reverse=False):
        reg, reged = self.strong_pairs(top=1, reverse=reverse)
        return (reg[0], reged[0])

    def show_corr(self, top=10, reverse=False, gene_name=None):
        """
        @param gene_name : Gene Names (ex.Transcription factors) shape=(num_gene)
        """
        A = self.A
        num_gene = A.shape[0]
        # Handle Naming.
        if gene_name is None:
            gene_name = [str(i) for i in range(num_gene)]
        else:
            if len(gene_name)!=num_gene:
                raise ValueError(f"`gene_name` must have the shape ({num_gene}), \
                                  but got ({len(gene_name)})")
        gene_name = np.asarray(gene_name)
        # Arrange in order of strong correlation.
        regulate,regulated = self.strong_pairs(top=top, reverse=reverse)

        table = Table()
        table.set_cols(colname="#", values=np.arange(top)+1, zero_padding=True)
        table.set_cols(colname="Regulate", values=gene_name[regulate])
        table.set_cols(colname="Regulated", values=gene_name[regulated])
        table.set_cols(colname="Correlation", values=A[regulated,regulate], fmt=".3f", sign="+")
        table.show()

    def plot_corr(self, ax=None, **heatmatp_args):
        if ax is not None:
            fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(self.A, ax=ax, **heatmatp_args)
        plt.show()
