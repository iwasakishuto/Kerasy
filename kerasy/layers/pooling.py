# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

from ..engine.base_layer import Layer

class MaxPooling2D(Layer):
    """
    ex.) pool_size=(2,2)
    =======================
    [forward]
    0 1 2 0    \
    3 4 2 1  ---\  4 2
    0 0 1 3  ---/  4 3
    4 3 0 2    /
    =======================
    [backprop]
    0 0 b 0   /
    0 a b 0  /---  a b
    0 0 0 d  \---  c d
    c 0 0 0   \
    """
    def __init__(self, pool_size=(2, 2), **kwargs):
        self.mask = None
        self.pool_size = pool_size
        super().__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        self.H, self.W, self.F = input_shape
        self.input_shape = input_shape
        ph,pw = self.pool_size
        self.OH = self.H//ph
        self.OW = self.W//pw
        self.OF = self.F
        self.output_shape = (self.OH, self.OW, self.OF)
        return self.output_shape

    def _generator(self, image):
        """ Generator for training. """
        ph,pw = self.pool_size
        for i in range(self.H//ph):
            for j in range(self.W//pw):
                clipped_img = image[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                yield clipped_img, i, j

    def forward(self, input):
        ph,pw = self.pool_size
        out = np.zeros(self.output_shape) # output image shape.
        mask = np.zeros(self.input_shape)
        for clipped_input, i, j in self._generator(input):
            max_vals = np.amax(clipped_input, axis=(0, 1)) # shape=(F,)
            out[i,j,:] = max_vals
            mask[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = clipped_input==max_vals
        self.mask = mask
        return out

    def backprop(self, pooled_delta, lr=1e-3):
        """ Loss only flows to the pixel that takes the maximum value in pooling block. """
        delta = np.zeros(self.input_shape)
        ph,pw = self.pool_size
        for mask, i, j in self._generator(self.mask):
            delta[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :] = np.where(mask, pooled_delta[i,j,:], 0.)
        return delta
