# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from ..utils.generic_utils import get_uid

class Layer():
    """Abstract base layer class."""
    def __init__(self, **kwargs):
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._grads = {}  # (name, delta)
        self._updates = {}
        prefix = self.__class__.__name__.lower()
        self.name = prefix + '_' + str(get_uid(prefix))
        self.trainable = kwargs.get('trainable', True)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        output_shape = input_shape
        self.output_shape = output_shape
        return output_shape

    def build(self, input_shape):
        output_shape = self.compute_output_shape(input_shape)
        return output_shape

    def add_weight(self, shape=(), name=None, dtype=None, initializer=None, regularizer=None, constraint=None, trainable=True):
        """
        @param  shape      : (tuple) The shape of the weight.
        @param  dtype      : (dtype) The dtype of the weight.
        @param  initializer: (string) An Initializer instance.
        @param  regularizer: (string) A Regularizer instance.
        @param  trainable  : (bool) A boolean, whether the weight should be trained via backprop or not.
        @return weight     : (ndarray) The created weights variable.
        """
        weight = initializer(shape=shape, dtype=dtype)
        if trainable:
            self._trainable_weights.append(name)
        else:
            self._non_trainable_weights.append(name)
        self._updates[name] = np.expand_dims(weight, axis=0) # shape=(z,x,y)
        self._grads[name] = np.zeros_like(weight) # shape=(x,y)
        return weight

    def update(self, optimizer, batch_size):
        if self.trainable and len(self._non_trainable_weights)>0:
            self._trainable_weights += self._non_trainable_weights
            self._non_trainable_weights = []
        elif self.trainable == False and len(self._trainable_weights)>0:
            self._non_trainable_weights += self._trainable_weights
            self._trainable_weights = []

        for name in self._trainable_weights:
            weight = self.__dict__.get(name)
            regularizer = self.__dict__.get(f"{name}_regularizer")
            grad = self._grads[name]/batch_size + regularizer.diff(weight)
            new_weight = optimizer.get_updates(
                grad=grad,
                curt_param=weight,
                name=f"{self.name}_{name}"
            )
            self.__dict__[name] = new_weight # Update.
            # self._updates[name] = np.r_[self._updates[name], np.expand_dims(new_weight, axis=0)]
            self._grads[name]  = np.zeros_like(new_weight)

    def get_weights(self):
        return []

    def set_weights(self, weights):
        pass

    @property
    def weights(self):
        return self.get_weights()
