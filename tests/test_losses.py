# coding: utf-8
import numpy as np
from kerasy import losses

n_features = 5
y_shape_   = np.ones(shape=(5,))
y_shape_1  = np.ones(shape=(1, n_features))
y_shape_10 = np.ones(shape=(10, 5))

def loss_check(loss, y):
    val = loss.loss(y,y)
    assert isinstance(val, float) or isinstance(val, int)

    diff = loss.diff(y,y)
    assert diff.shape == y.shape

def _test_losses(identifier):
    loss = losses.get(identifier)
    loss_check(loss, y_shape_)
    loss_check(loss, y_shape_1)
    loss_check(loss, y_shape_10)

def test_mean_squared_error():
    _test_losses(identifier="mean_squared_error")

def test_mse():
    _test_losses(identifier="mse")

def test_categorical_crossentropy():
    _test_losses(identifier="categorical_crossentropy")

def test_softmax_categorical_crossentropy():
    _test_losses(identifier="softmax_categorical_crossentropy")
