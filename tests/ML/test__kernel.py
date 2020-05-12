# coding: utf-8
import numpy as np
from kerasy.ML import _kernel

def get_test_data():
    rnd = np.random.RandomState(123)
    x,y = rnd.rand(2, 10)
    return x,y

def _test_kernel(kernel):
    x,y = get_test_data()
    kernel = _kernel.get(kernel)
    val = kernel(x,y)
    assert isinstance(val, int) or isinstance(val, float)

def test_linear():
    Linear = _kernel.get("linear", c=0)
    _test_kernel(Linear)

def test_polynomial():
    Polynomial = _kernel.get("polynomial", alpha=1, c=0, d=3)
    _test_kernel(Polynomial)

def test_gaussian():
    Gaussian = _kernel.get("gaussian", sigma=1)
    _test_kernel(Gaussian)

def test_exponential():
    Exponential = _kernel.get("exponential", sigma=0.1)
    _test_kernel(Exponential)

def test_laplacian():
    Laplacian = _kernel.get("laplacian", sigma=0.1)
    _test_kernel(Laplacian)

def test_sigmoid():
    Sigmoid = _kernel.get("sigmoid", alpha=1, c=0)
    _test_kernel(Sigmoid)

def test_rational_quadratic():
    Rational_quadratic = _kernel.get("rational_quadratic", c=1)
    _test_kernel(Rational_quadratic)

def test_multiquadric():
    Multiquadric = _kernel.get("multiquadric", c=1)
    _test_kernel(Multiquadric)

def test_inverse_multiquadric():
    Inverse_multiquadric = _kernel.get("inverse_multiquadric", c=1)
    _test_kernel(Inverse_multiquadric)

def test_log():
    Log = _kernel.get("log", d=3)
    _test_kernel(Log)
