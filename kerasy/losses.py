# coding: utf-8

class mean_squared_error():
    def loss(y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)
    def diff(y_true, y_pred):
        return y_pred-y_true

LossHandler = {
    'mean_squared_error' : mean_squared_error,
}

def LossFunc(loss_func_name):
    return LossHandler[loss_func_name]
