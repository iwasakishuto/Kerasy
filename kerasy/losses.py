# coding: utf-8
import re
import numpy as np
from abc import ABCMeta, abstractmethod
from .utils import mk_class_get
from .utils import format_spec_create

class KerasyAbstLoss(metaclass=ABCMeta):
    def __init__(self, aggr_type="sum", fmt=".3f", **format_codes):
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()
        # Aggregation method for training.
        self.aggr_method = {
            "sum" : lambda sum,n : sum,
            "ave" : lambda sum,n : sum/n
        }.get(aggr_type, lambda sum,n : sum)
        self.format_spec = format_spec_create(fmt=fmt, **format_codes)

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def diff(self, y_true, y_pred):
        pass

class MeanSquaredError(KerasyAbstLoss):
    def __init__(self):
        super().__init__(aggr_type="ave")

    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true), axis=-1)
    def diff(self, y_true, y_pred):
        return y_pred-y_true

class CategoricalCrossentropy(KerasyAbstLoss):
    """
    Categorical CrossEntropy Loss: np.sum(y_true * np.log(y_pred))
    Cross-entropy loss increases as the predicted probability diverges from the actual label.
    """
    def loss(self, y_true, y_pred):
        small_val = 1e-7 # Avoiding np.log(y_pred)=inf if y_pred==0
        return -np.sum(y_true * np.log(y_pred+small_val))
    def diff(self, y_true, y_pred):
        small_val = 1e-7
        return -y_true/(y_pred+small_val)

class SoftmaxCategoricalCrossentropy(KerasyAbstLoss):
    def loss(self, y_true, y_pred):
        # # Softmax Layer (model.activation)
        # exps = np.exp(y_pred - np.max(y_pred))
        # y_pred = exps/np.sum(exps)

        # Categorical CrossEntropy
        small_val = 1e-7 # Avoiding np.log(y_pred)=inf if y_pred==0
        return -np.sum(y_true * np.log(y_pred+small_val))
    def diff(self, y_true, y_pred):
        """
        Ref: https://iwasakishuto.github.io/Kerasy/doc/theme/img/ComputationalGraph/Softmax-with-Loss.png
               a -> Softmax -> y_pred \
        y_pred - y_true                | CrossEntropy -> loss
               ^               y_true /                    |
               |                                           |
               \-------------------------------------------/
        """
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
