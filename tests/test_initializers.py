# coding: utf-8
from kerasy.models import Sequential
from kerasy.layers import Input, Conv2D, Dense

square_matrix_size = (10,10)
kh,kw = (3,3)
F, OF = (5,8)

def _test_initialization_conv(kernel_initializer):
    model = Sequential()
    model.add(Input(input_shape=(20,20,F)))
    model.add(Conv2D(filters=OF, kernel_size=(kh,kw), kernel_initializer=kernel_initializer))
    model.compile(loss="mse", optimizer="adam")
    
    assert model.layers[-1].kernel.shape == (kh,kw,F, OF)

def _test_initialization_dense(kernel_initializer):
    units_in, units_out = square_matrix_size
    model = Sequential()
    model.add(Input(input_shape=(units_in)))
    model.add(Dense(units_out, kernel_initializer=kernel_initializer))
    model.compile(loss="mse", optimizer="adam")

    assert model.layers[-1].kernel.shape == square_matrix_size

def test_zeros():
    _test_initialization_conv(kernel_initializer="zeros")
    _test_initialization_dense(kernel_initializer="zeros")

def test_ones():
    _test_initialization_conv(kernel_initializer="ones")
    _test_initialization_dense(kernel_initializer="ones")

def test_constant():
    _test_initialization_conv(kernel_initializer="constant")
    _test_initialization_dense(kernel_initializer="constant")

def test_random_normal():
    _test_initialization_conv(kernel_initializer="random_normal")
    _test_initialization_dense(kernel_initializer="random_normal")

def test_random_uniform():
    _test_initialization_conv(kernel_initializer="random_uniform")
    _test_initialization_dense(kernel_initializer="random_uniform")

def test_truncated_normal():
    _test_initialization_conv(kernel_initializer="truncated_normal")
    _test_initialization_dense(kernel_initializer="truncated_normal")

def test_variance_scaling():
    _test_initialization_conv(kernel_initializer="variance_scaling")
    _test_initialization_dense(kernel_initializer="variance_scaling")

def test_orthogonal():
    _test_initialization_conv(kernel_initializer="orthogonal")
    _test_initialization_dense(kernel_initializer="orthogonal")

def test_identity():
    #_test_initialization_conv(kernel_initializer="identity")
    _test_initialization_dense(kernel_initializer="identity")

def test_glorot_normal():
    _test_initialization_conv(kernel_initializer="glorot_normal")
    _test_initialization_dense(kernel_initializer="glorot_normal")

def test_glorot_uniform():
    _test_initialization_conv(kernel_initializer="glorot_uniform")
    _test_initialization_dense(kernel_initializer="glorot_uniform")

def test_he_normal():
    _test_initialization_conv(kernel_initializer="he_normal")
    _test_initialization_dense(kernel_initializer="he_normal")

def test_lecun_normal():
    _test_initialization_conv(kernel_initializer="lecun_normal")
    _test_initialization_dense(kernel_initializer="lecun_normal")

def test_he_uniform():
    _test_initialization_conv(kernel_initializer="he_uniform")
    _test_initialization_dense(kernel_initializer="he_uniform")

def test_lecun_uniform():
    _test_initialization_conv(kernel_initializer="lecun_uniform")
    _test_initialization_dense(kernel_initializer="lecun_uniform")
