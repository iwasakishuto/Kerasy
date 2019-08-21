# coding: utf-8
import numpy as np

class mean_squared_error():
    def loss(y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)
    def diff(y_true, y_pred):
        return y_pred-y_true

class categorical_crossentropy():
    def loss(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))
    def diff(y_true, y_pred):
        """ @param y_true:  """
        return y_pred*(y_true-y_pred)

LossHandler = {
    'mean_squared_error' : mean_squared_error,
    'categorical_crossentropy' : categorical_crossentropy,
}

def LossFunc(loss_func_name):
    return LossHandler[loss_func_name]
