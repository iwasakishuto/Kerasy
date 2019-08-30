# coding: utf-8
import numpy as np

class mean_squared_error():
    def loss(y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)
    def diff(y_true, y_pred):
        return y_pred-y_true

class categorical_crossentropy():
    def loss(y_true, y_pred):
        small_val = 0 #1e-7 # Avoiding np.log(y_pred)=inf if y_pred==0
        return -np.sum(y_true * np.log(y_pred+small_val))
    def diff(y_true, y_pred):
        small_val = 0 #1e-7
        return -y_true/(y_pred+small_val)

LossHandler = {
    'mean_squared_error' : mean_squared_error,
    'categorical_crossentropy' : categorical_crossentropy,
}

def LossFunc(loss_func_name):
    return LossHandler[loss_func_name]
