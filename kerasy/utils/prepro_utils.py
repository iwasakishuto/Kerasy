# coding: utf-8
import itertools
import numpy as np

class BaseTransformer():
    def __init__(self, _base_func_kwargs=[], **kwargs):
        """ Transformer's Base Class.
        @params _base_func_kwargs: (list) Keywards for `_base_function`.
        @params use_bias         : (bool) Whether x_transformed includes bias (all 1 array) or not.
        @params kwargs           : The pairs of 'Key' and 'Val', which are used in `_base_function`.
        """
        self.use_bias = kwargs.pop("use_bias", True)
        self._base_func_kwargs = _base_func_kwargs
        self.__dict__.update(**kwargs)
        # Preprocessing. Convert to List type to pass the `itertools`.
        for key in _base_func_kwargs:
            vals = self.__dict__[key]
            if isinstance(vals, np.ndarray):
                self.__dict__[key] = list(vals)
            elif not isinstance(vals, list):
                self.__dict__[key] = [vals]

    def _base_function(self, x, **kwargs):
        raise NotImplementedError()

    def transform(self, x):
        """
        @param  x           : (ndarray) shape=(N,D)
        @return x_tranformed: (ndarray) shape=(N,M)
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        N,D = x.shape

        base_func_values = [self.__dict__[k] for k in self._base_func_kwargs]
        base_func_values_combs = list(itertools.product(*base_func_values))

        x_transformed = np.asarray([self._base_function(x, **dict(zip(self._base_func_kwargs,valsets))).reshape(N,-1) for valsets in base_func_values_combs])
        x_transformed = np.hstack(x_transformed)

        if self.use_bias:
            x_transformed = np.c_[np.ones(shape=(N,1)), x_transformed]

        return x_transformed

class GaussianBaseTransformer(BaseTransformer):
    def __init__(self, **kwargs):
        """ x.shape = (N,D)
        @params kwargs:
            - mu      : (list) shape=(M_mu, D)
            - sigma   : (list) shape=(M_sigma, D)
            - use_bias: (bool) Whether x_transformed includes bias or not.
                               If True, x_transformed.shape[1] += 1.
        @return x_tranformed: shape=(N, M_mu*M_sigma)
        """
        super().__init__(_base_func_kwargs=["mu", "sigma"], **kwargs)

    def _base_function(self, x, mu, sigma):
        """
        @param x:  (ndarray) shape=(N,D)
        @param mu: (ndarray) shape=(D,)
        """
        return np.exp(-1/2 * np.sum(np.square(x-mu), axis=-1) / sigma**2)

class SigmoidBaseTransformer(BaseTransformer):
    def __init__(self, **kwargs):
        """ x.shape = (N,D)
        @params kwargs:
            - mu      : (list) shape=(M_mu, D)
            - sigma   : (list) shape=(M_sigma, D)
            - use_bias: (bool) Whether x_transformed includes bias or not.
                               If True, x_transformed.shape[1] += 1.
        @return x_tranformed: shape=(N, M_mu*M_sigma)
        """
        super().__init__(_base_func_kwargs=["mu", "sigma"], **kwargs)

    def _base_function(self, x, mu, sigma):
        """
        @param x:  (ndarray) shape=(N,D)
        @param mu: (ndarray) shape=(D,)
        """
        return 1 / (1 + np.exp(- (x-mu)/sigma ))

class PolynomialBaseTransformer(BaseTransformer):
    def __init__(self, **kwargs):
        """ x.shape = (N,D)
        @params kwargs:
            - exponent: (list) shape=(M_exponent) each element represents the exponent.
            - use_bias: (bool) Whether x_transformed includes bias or not.
                               If True, x_transformed.shape[1] += 1.
        @return x_tranformed: shape=(N, M)
        """
        super().__init__(_base_func_kwargs=["exponent"], **kwargs)

    def _base_function(self, x, exponent):
        """
        @param exponent: int
        """
        return x**exponent

class NoneBaseTransformer(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__(_base_func_kwargs=[], **kwargs)

    def _base_function(self, x):
        return x

BasisTransformerHandler = {
    'gaussian': GaussianBaseTransformer,
    'sigmoid': SigmoidBaseTransformer,
    'polynomial': PolynomialBaseTransformer,
    'none': NoneBaseTransformer,
}

def basis_transformer(basis_name, **basisargs):
    if basis_name not in BasisTransformerHandler:
        raise KeyError(f"Please select basis transformer from {list(BasisTransformerHandler.keys())}")
    return BasisTransformerHandler[basis_name](**basisargs)
