# coding: utf-8
import numpy as np
from kerasy.models import Sequential
from kerasy.layers import Input, Conv2D, Dense

conv_input_shape = (28,28,1)
dense_input_shape = (10,)

conv_input  = np.expand_dims(np.ones(shape=conv_input_shape), axis=0)
dense_input = np.expand_dims(np.ones(shape=dense_input_shape), axis=0)

def _test_activation_conv(activation):
    model = Sequential()
    model.add(Input(input_shape=conv_input_shape))
    model.add(Conv2D(3, activation=activation))
    model.compile(loss="mse", optimizer="adam")
    model.predict(conv_input)

def _test_activation_dense(activation):
    model = Sequential()
    model.add(Input(input_shape=(dense_input_shape)))
    model.add(Dense(3, activation=activation))
    model.compile(loss="mse", optimizer="adam")
    model.predict(dense_input)

def test_linear():
    _test_activation_conv(activation="linear")
    _test_activation_dense(activation="linear")

def test_softmax():
    _test_activation_conv(activation="softmax")
    _test_activation_dense(activation="softmax")

def test_tanh():
    _test_activation_conv(activation="tanh")
    _test_activation_dense(activation="tanh")

def test_relu():
    _test_activation_conv(activation="relu")
    _test_activation_dense(activation="relu")

def test_sigmoid():
    _test_activation_conv(activation="sigmoid")
    _test_activation_dense(activation="sigmoid")
