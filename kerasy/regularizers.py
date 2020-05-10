# coding: utf-8

import re
import numpy as np

from abc import ABCMeta, abstractmethod
from .utils import mk_class_get

class KerasyAbstRegularizer(metaclass=ABCMeta):
    def __init__(self, **format_codes):
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()

    @abstractmethod
    def loss(self, weight):
        pass

    @abstractmethod
    def diff(self, weight):
        pass

class none(KerasyAbstRegularizer):
    def loss(self, weight):
        return 0.

    def diff(self, weight):
        return 0.

class L1(KerasyAbstRegularizer):
    def __init__(self, lambda1=0.01):
        self.lambda1 = lambda1
        super().__init__()

    def loss(self, weight):
        return self.lambda1 * np.sum(np.abs(weight))

    def diff(self, weight):
        return self.lambda1 * np.where(weight>0, 1., -1.) * np.not_equal(weight, 0.0)

class L2(KerasyAbstRegularizer):
    def __init__(self, lambda2=0.01):
        self.lambda2 = lambda2
        super().__init__()

    def loss(self, weight):
        return self.lambda2 * np.sum(np.square(weight))

    def diff(self, weight):
        return 2 * self.lambda2 * weight

class L1L2(KerasyAbstRegularizer):
    def __init__(self, lambda1=0.01, lambda2=0.01):
        self.L1 = L1(lambda1)
        self.L2 = L2(lambda2)
        super().__init__()

    def loss(self, weight):
        return self.L1.loss(weight) * self.L2.loss(weight)

    def diff(self, weight):
        return self.L1.diff(weight) * self.L2.diff(weight)

KerasyRegularizerClasses = {
    'none'     : none,
    'l1'       : L1,
    'l2'       : L2,
    'l1l2'     : L1L2,
}

get = mk_class_get(
    all_classes=KerasyRegularizerClasses,
    kerasy_abst_class=[KerasyAbstRegularizer],
    genre="regularizer"
)
