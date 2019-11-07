# coding: utf-8
from __future__ import absolute_import

import numpy as np

from ..activations import ActivationFunc
from ..initializers import Initializer
from ..losses import LossFunc
from ..engine.base_layer import Layer

class Input(Layer):
    def __init__(self, input_shape, **kwargs):
        self.input_shape=input_shape
        super().__init__(**kwargs)

class Flatten(Layer):
    def __init__(self):
        self.input=None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(list(input_shape)),)

    def forward(self, input):
        return input.flatten()

    def backprop(self, delta):
        return delta.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, units, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros'):
        """
        @param units             : (tuple) dimensionality of the (input space, output space).
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        self.output_shape=(units,)
        self.kernel_initializer = Initializer(kernel_initializer)
        self.bias_initializer   = Initializer(bias_initializer)
        self.h = ActivationFunc(activation)
        self.w = None
        self.z = None
        self.a = None
        self.trainable = True

    def build(self, input_shape):
        self.input_shape = input_shape
        self.w = np.c_[
            self.kernel_initializer(shape=(self.output_shape[0],self.input_shape[0])),
            self.bias_initializer(shape=(self.output_shape[0],1))
        ]

    def forward(self, input):
        """ @param input: shape=(Din,) """
        z_in = np.append(input,1)       # shape=(Din+1,)
        a = self.w.dot(z_in)            # (Dout,Din+1) @ (Din+1,) = (Dout,)
        z_out = self.h.forward(input=a) # shape=(Dout,)
        self.z = z_in
        self.a = a
        return z_out

    def backprop(self, dEdz_out):
        """ @param dEdz_out: shape=(Dout,) """
        dz_outda = self.h.diff(self.a)
        if len(dz_outda.shape) == 1:
            dEda = dz_outda*dEdz_out      # δ, shape=(Dout,)
        elif len(dz_outda.shape) == 2:
            dEda = np.sum(dz_outda*dEdz_out, axis=1) # δ, shape=(Dout,)
        else:
            raise ValueError("Couldn't understand the shape of dz_out/da. It's shape must be 1D or 2D.")
        dEdz_in = self.w.T.dot(dEda)        # (Din+1,Dout) @ (Dout,) = (Din+1,)
        if self.trainable:
            self.update(dEda)
        return dEdz_in[:-1]                 # shape=(Din,) term of bias is not propagated.

    def update(self, delta, ALPHA=0.01):
        """ @param delta: shape=(Dout,) """
        dw = np.outer(delta, self.z) # (Dout,) × (Din+1,) = (Dout,Din+1)
        self.w -= ALPHA*dw # update.
