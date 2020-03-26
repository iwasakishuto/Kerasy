# coding: utf-8
import numpy as np

from ..utils import flush_progress_bar

class SCODE():
    """ ref: https://academic.oup.com/bioinformatics/article/33/15/2314/3100331
    SCODE: an efficient regulatory network inference algorithm
           from single-cell RNA-Seq during differentiation
    """
    def __init__(self, lambda_, b_min=-10, b_max=2, random_state=None):
        self.lambda_ = lambda_
        self.b_min = b_min
        self.b_max = b_max
        self.seed = random_state

    def fit(self, expression, time, dimension=4, max_iter=100, verbose=1):
        """
        @params expression : shape=(num_gene, num_cell)
        @params time       : shape=(num_cell)
        @params dimension  : (int) How many dimensions to reduce.
        """
        rnd = np.random.RandomState(self.seed)
        num_gene, num_cell = expression.shape
        b = np.random.uniform(low=b_min, high=b_max, size=dimension)

        for it in range(max_iter):
            d = rnd.randint(dimension)
            b[d] = rnd.uniform(low=b_min, high=b_max, size=1)
            Z = np.power(np.e, (np.expand_dims(t, axis=1) * np.expand_dims(b, axis=0)))
            # Solution of linear regression (X^{(e)}≃WZ^{(e)})
            W = np.dot(np.dot(X,Z), np.linalg.inv(np.dot(Z.T,Z) + np.eye(D)*lamda))
            # Residual sum of squares.
            RSS = np.linalg.norm(X-np.dot(W,Z.T))
            if it==0 or RSS < RSS_best:
                b_best = b
                W_best = W
                RSS_best = a
            else:
                b = b_best
            flush_progress_bar(it, max_iter, metrics={"RSS": RSS}, verbose=verbose)

        if verbose>0: print()
        self.B = np.diag(b_best)
        self.W = W_best
