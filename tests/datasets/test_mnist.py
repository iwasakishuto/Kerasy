# coding: utf-8
from kerasy.datasets import mnist

def test_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_train, *img_shape_train = x_train.shape
    num_test,  *img_shape_test  = x_test.shape

    assert img_shape_test == img_shape_train
    assert len(y_train) == num_train
    assert len(y_test)  == num_test
