# coding: utf-8
import numpy as np
from abc import ABCMeta, abstractmethod
from .utils import mk_class_get

from .losses import KerasyAbstLoss, KerasyLossClasses

class KerasyAbstMetrics(metaclass=ABCMeta):
    def __init__(self):
        self.name = self.__class__.__name__.lower()

    def __repr__(self):
        return f"{super().__repr__()}\n{self.__doc__}"

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

class Accuracy(KerasyAbstMetrics):
    def loss(self, y_true, y_pred):
        return np.mean(y_true==y_pred)

KerasyMetricsClasses = {
    'acc'      : Accuracy,
    'accuracy' :  Accuracy,
}
KerasyMetricsClasses.update(KerasyLossClasses)

get = mk_class_get(
    all_classes=KerasyMetricsClasses,
    kerasy_abst_class=[KerasyAbstMetrics, KerasyAbstLoss],
    genre="metrics"
)
