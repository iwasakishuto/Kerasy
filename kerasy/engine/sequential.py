# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

from ..losses import LossFunc
from ..optimizers import Optimizer
from ..layers.core import Input
from ..utils.training_utils import make_batches
from .base_layer import Layer

class Sequential():
    def __init__(self):
        self.layers = []

    def add(self, layer):
        """Adds a layer instance."""
        if not isinstance(layer, Layer):
            raise TypeError(f"The added layer must be an instance of class Layer. Found: {str(layer)}")
        self.layers.append(layer)

    def compile(self, optimizer, loss=None, metrics=None):
        """ Creates the layer weights.
        @param optimizer   : (String name of optimizer) or (Optimizer instance).
        @param loss        : (String name of loss function) or (Loss instance).
        @param metrics     : (List) Metrics to be evaluated by the model during training and testing.
        """
        self.optimizer = Optimizer(optimizer) if isinstance(optimizer, str) else optimizer
        self.loss = LossFunc(loss) if isinstance(loss, str) else loss
        self.metrics = metrics
        input_layer = self.layers[0]
        if not isinstance(input_layer, Input): raise ValueError(f"The initial layer should be Input instance, but {str(input_layer)}")
        output_shape = input_layer.input_shape
        for layer in self.layers:
            output_shape = layer.build(output_shape)

    def fit(self,
            x=None, y=None, batch_size=32, epochs=1, verbose=1, shuffle=True,
            validation_spilit=0, validation_data=None, validation_steps=None,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
        if kwargs: raise TypeError(f'Unrecognized keyword arguments: {str(kwargs)}')
        if (x is None) or (y is None): raise ValueError('Please specify the trainig data. (x,y)')
        # Prepare validation data.
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else: raise ValueError(f"When passing validation_data, it must contain 2 (x_val, y_val) or 3 (x_val, y_val, val_sample_weights) items. However, it contains {len(validation)} items.")

        num_train_samples = len(x)
        index_array = np.arange(num_train_samples)
        for epoch in range(initial_epoch, epochs):
            if shuffle: np.random.shuffle(index_array)
            batches = make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                for x_, y_ in zip(x[batch_ids], y[batch_ids]):
                    out = self.forward(x_)
                    self.backprop(y_, out)
                self.updates()
            print(f'[{epoch+1}/{epochs}]')

    def forward(self, input):
        out=input
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backprop(self, y_true, out):
        dEdXout = self.loss.diff(y_true, out)
        for layer in reversed(self.layers):
            dEdXout = layer.backprop(dEdXout)

    def predict(self, x_train):
        if np.ndim(x_train) == 1:
            return self.forward(x_train)
        else:
            return np.array([self.forward(x) for x in x_train])
