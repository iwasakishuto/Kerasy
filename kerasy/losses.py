# coding: utf-8
import numpy as np
from abc import ABCMeta, abstractmethod
from .utils import mk_class_get

class KerasyAbstLoss(metaclass=ABCMeta):
    def __init__(self):
        self.name = self.__class__.__name__.lower()

    def __repr__(self):
        return f"{super().__repr__()}\n{self.__doc__}"

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def diff(self, y_true, y_pred):
        pass

class MeanSquaredError(KerasyAbstLoss):
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)
    def diff(self, y_true, y_pred):
        return y_pred-y_true

class CategoricalCrossentropy(KerasyAbstLoss):
    def loss(self, y_true, y_pred):
        small_val = 1e-7 # Avoiding np.log(y_pred)=inf if y_pred==0
        return -np.sum(y_true * np.log(y_pred+small_val))
    def diff(self, y_true, y_pred):
        small_val = 1e-7
        return -y_true/(y_pred+small_val)

class SoftmaxCategoricalCrossentropy(KerasyAbstLoss):
    def loss(self, y_true, y_pred):
        # Softmax Layer
        exps = np.exp(y_pred - np.max(y_pred))
        y_pred = exps/np.sum(exps)
        # Categorical CrossEntropy
        small_val = 1e-7 # Avoiding np.log(y_pred)=inf if y_pred==0
        return -np.sum(y_true * np.log(y_pred+small_val))
    def diff(self, y_true, y_pred):
        return y_pred - y_true

KerasyLossClasses = {
    'mean_squared_error' : MeanSquaredError,
    'mse'                : MeanSquaredError,
    'categorical_crossentropy' : CategoricalCrossentropy,
    'softmax_categorical_crossentropy' : SoftmaxCategoricalCrossentropy,
}

get = mk_class_get(
    all_classes=KerasyLossClasses,
    kerasy_abst_class=[KerasyAbstLoss],
    genre="loss"
)
