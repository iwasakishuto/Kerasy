# coding: utf-8
import re
import numpy as np
from abc import ABCMeta, abstractmethod
from .utils import mk_class_get
from .utils import format_spec_create

from .losses import KerasyAbstLoss, KerasyLossClasses

class KerasyAbstMetrics(metaclass=ABCMeta):
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

class CategoricalAccuracy(KerasyAbstMetrics):
    """
    Categorical Accuracy:
    Calculates the mean accuracy rate across all predictions for multiclass classification problems.
    """
    def __init__(self):
        super().__init__(aggr_type="ave", fmt=".1%")

    def loss(self, y_true, y_pred):
        return np.mean(np.equal(
            y_true.argmax(axis=-1),
            y_pred.argmax(axis=-1)
        ))


KerasyMetricsClasses = {
    'categorical_accuracy': CategoricalAccuracy
}
KerasyMetricsClasses.update(KerasyLossClasses)

get = mk_class_get(
    all_classes=KerasyMetricsClasses,
    kerasy_abst_class=[KerasyAbstMetrics, KerasyAbstLoss],
    genre="metrics"
)
