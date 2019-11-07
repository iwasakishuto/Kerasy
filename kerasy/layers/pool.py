# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
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
        self.input = None
        self.pool_size = pool_size
        super().__init__(**kwargs)

    def compute_output_shap(self, input_shape):
        self.H, self.W, self.F = input_shape
        ph,pw = self.pool_size
        self.OH = self.H//ph
        self.OW = self.W//pw
        self.OF = self.F
        self.output_shape = (self.OH, self.OW, self.OF)

    def _generator(self, Xin):
        """ Generator for training. """
        ph,pw = self.pool_size
        for i in range(self.H//ph):
            for j in range(self.W//pw):
                clipedXin = Xin[i*ph:(i+1)*ph,j*pw:(j+1)*pw]
                yield clipedXin,i,j

    def forward(self, input):
        self.input = input # Memorize the input array. (not shape)
        ph,pw = self.pool_size
        out = np.zeros((self.H//ph, self.W//pw, self.F)) # output image shape.
        for crip_image, i, j in self._generator(input):
            out[i][j] = np.amax(crip_image, axis=(0, 1))
        return out

    def backprop(self, Pre_delta, lr=1e-3):
        """ Loss only flows to the pixel that takes the maximum value in pooling block. """
        Next_delta = np.zeros(self.input.shape)
        ph,pw = self.pool_size
        for crip_image, i, j in self._generator(self.input):
            Next_delta[i*ph:(i+1)*ph,j*pw:(j+1)*pw] = np.where(crip_image==np.max(crip_image), np.max(crip_image), 0)

        return Next_delta
