# coding: utf-8
import numpy as np
from kerasy import metrics

n_features = 5
y_shape_   = np.ones(shape=(5,))
y_shape_1  = np.ones(shape=(1, n_features))
y_shape_10 = np.ones(shape=(10, 5))

def metrics_check(metric, y):
    loss = metric.loss(y,y)
    assert isinstance(loss, float) or isinstance(loss, int)

def _test_metrics(identifier):
    metric = metrics.get(identifier)
    metrics_check(metric, y_shape_)
    metrics_check(metric, y_shape_1)
    metrics_check(metric, y_shape_10)

def test_categorical_accuracy():
    _test_metrics(identifier="categorical_accuracy")

def test_mean_squared_error():
    _test_metrics(identifier="mean_squared_error")

def test_mse():
    _test_metrics(identifier="mse")

def test_categorical_crossentropy():
    _test_metrics(identifier="categorical_crossentropy")

def test_softmax_categorical_crossentropy():
    _test_metrics(identifier="softmax_categorical_crossentropy")
