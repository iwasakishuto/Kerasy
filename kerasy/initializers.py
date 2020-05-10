# coding: utf-8
import re
import numpy as np
from scipy import stats
from abc import ABCMeta, abstractmethod

from .utils import mk_class_get
from .utils import handleKeyError
from .utils import handleRandomState

class KerasyAbstInitializer(metaclass=ABCMeta):
    def __init__(self):
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()

    @abstractmethod
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

class Zeros(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None):
        return np.zeros(shape=shape, dtype=dtype)

class Ones(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None):
        return np.ones(shape=shape, dtype=dtype)

class Constant(KerasyAbstInitializer):
    def __call__(self, shape, value=0, dtype=None):
        return np.full(shape=shape, fill_value=value, dtype=dtype)

class RandomNormal(KerasyAbstInitializer):
    def __call__(self, shape, mean=0, stddev=0.05, dtype=None, seed=None):
        rnd = handleRandomState(seed)
        return rnd.normal(size=shape, loc=mean, scale=stddev).astype(dtype)

class RandomUniform(KerasyAbstInitializer):
    def __call__(self, shape, minval=-0.05, maxval=0.05, dtype=None, seed=None):
        rnd = handleRandomState(seed)
        return rnd.uniform(size=shape, low=minval, high=maxval)

class TruncatedNormal(KerasyAbstInitializer):
    def __call__(self, shape, mean=0.0, stddev=0.05, dtype=None, seed=None):
        X = stats.truncnorm(
            (-stddev - mean) / stddev,
            (stddev  - mean) / stddev,
            loc=mean,
            scale=stddev,
        )
        return X.rvs(size=shape,random_state=seed).astype(dtype)

class VarianceScaling(KerasyAbstInitializer):
    def __call__(self, shape, scale=1.0, mode='fan_in', distribution='normal', dtype=None, seed=None):
        fan_in, fan_out = _compute_fans(shape)
        n = fan_in if mode=="fan_in" else fan_out if mode=="fan_out" else (fan_in+fan_out)/2
        scale /= max(1., n)
        if distribution=='normal':
            # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = np.sqrt(scale) / .87962566103423978
            return TruncatedNormal(shape=shape, mean=0.0, stddev=stddev, dtype=dtype, seed=seed)
        else:
            limit = np.sqrt(3 * scale)
            return RandomUniform(shape=shape, minval=-limit, maxval=limit, dtype=dtype, seed=seed)

class Orthogonal(KerasyAbstInitializer):
    def __call__(self, shape, gain=1.0, dtype=None, seed=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        rng = np.random
        if seed is not None:
            rng = np.random.RandomState(seed)
        a = rng.normal(loc=0.0, scale=1.0, size=flat_shape).astype(dtype)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # Pick the one with the correct shape.
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return gain * q[:shape[0], :shape[1]]

class Identity(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, gain=1.0):
        if len(shape) != 2 or shape[0]!=shape[1]:
            raise ValueError('Identity matrix initializer can only be used for 2D Square matrices.')
        return gain * np.eye(N=shape[0], dtype=dtype)

class GlorotNormal(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, seed=None):
        return VarianceScaling(
            shape=shape,
            scale=1.,
            mode='fan_avg',
            distribution='normal',
            dtype=dtype,
            seed=seed
        )

class GlorotUniform(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, seed=None):
        return VarianceScaling(
            shape=shape,
            scale=1.,
            mode='fan_avg',
            distribution='uniform',
            dtype=dtype,
            seed=seed
        )

class HeNormal(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, seed=None):
        return VarianceScaling(
            shape=shape,
            scale=2.,
            mode='fan_in',
            distribution='normal',
            dtype=dtype,
            seed=seed
        )

class LeCunNormal(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, seed=None):
        return VarianceScaling(
            shape=shape,
            scale=1.,
            mode='fan_in',
            distribution='normal',
            dtype=dtype,
            seed=seed
        )

class HeUniform(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, seed=None):
        return VarianceScaling(
            shape=shape,
            scale=2.,
            mode='fan_in',
            distribution='uniform',
            dtype=dtype,
            seed=seed
        )

class LeCunUniform(KerasyAbstInitializer):
    def __call__(self, shape, dtype=None, seed=None):
        return VarianceScaling(
            shape=shape,
            scale=1.,
            mode='fan_in',
            distribution='uniform',
            dtype=dtype,
            seed=seed
        )

def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.
    @param shape      : Integer shape tuple.
    @param data_format: Image data format to use for convolution kernels.
    @return fan_in    : size of the input shape.
    @return fan_out   : size of the output shape.
    """
    if len(shape) == 2:
        fan_in,fan_out = shape
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

KerasyInitializerFunctions = {
    'zeros'            : Zeros,
    'ones'             : Ones,
    'constant'         : Constant,
    'random_normal'    : RandomNormal,
    'random_uniform'   : RandomUniform,
    'truncated_normal' : TruncatedNormal,
    'variance_scaling' : VarianceScaling,
    'orthogonal'       : Orthogonal,
    'identity'         : Identity,
    'glorot_normal'    : GlorotNormal,
    'glorot_uniform'   : GlorotUniform,
    'he_normal'        : HeNormal,
    'lecun_normal'     : LeCunNormal,
    'he_uniform'       : HeUniform,
    'lecun_uniform'    : LeCunUniform,
}

get = mk_class_get(
    all_classes=KerasyInitializerFunctions,
    kerasy_abst_class=[KerasyAbstInitializer],
    genre="initializer"
)
