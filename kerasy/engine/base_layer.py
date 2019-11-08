# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from ..utils.generic_utils import get_uid
from ..initializers import Initializer

class Layer():
    """Abstract base layer class."""
    def __init__(self, **kwargs):
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._losses = {}  # (name, delta)
        self._updates = {}
        prefix = self.__class__.__name__.lower()
        name = prefix + '_' + str(get_uid(prefix))
        self.trainable = kwargs.get('trainable', True)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        output_shape = input_shape
        self.output_shape = output_shape
        return output_shape

    def build(self, input_shape):
        output_shape = self.compute_output_shape(input_shape)
        return output_shape

    def add_weight(self, shape=None, name=None, dtype=None, initializer=None, regularizer=None, constraint=None, trainable=True):
        """
        @param  shape      : (tuple) The shape of the weight.
        @param  dtype      : (dtype) The dtype of the weight.
        @param  initializer: (string) An Initializer instance.
        @param  regularizer: (string) A Regularizer instance.
        @param  trainable  : (bool) A boolean, whether the weight should be trained via backprop or not.
        @return weight     : (ndarray) The created weights variable.
        """
        shape = () if shape is None else shape
        weight = initializer(shape=shape, dtype=dtype)
        if trainable:
            self._trainable_weights.append(name)
        else:
            self._non_trainable_weights.append(name)
        self._updates[name] = [weight]
        self._losses[name] = [np.zeros_like(weight)]
        return weight

    def update(self, optimizer):
        if self.trainable:
            self._trainable_weights += self._non_trainable_weights
            self._non_trainable_weights = []
        else:
            self._non_trainable_weights += self._trainable_weights
            self._trainable_weights = []

        for param in self._trainable_weights:
            self.__dict__[param] += optimizer.get_updates(self._losses[param], self._updates[param], self.name)
            self._updates[param].append(getattr(self, param))
            self._losses[param] = [np.zeros_like(weight)]
