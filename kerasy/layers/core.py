# coding: utf-8
from __future__ import absolute_import

import numpy as np

from ..activations import ActivationFunc
from ..initializers import Initializer
from ..losses import LossFunc

class Layers():
    NLayers=1

    def __init__(self, name, units):
        self.name=f"Layer{Layers.NLayers}.{name}"
        self.outdim=units
        self.w = None
        Layers.NLayers+=1

    def build(self, indim):
        self.indim = indim
        self.w = np.c_[
            self.kernel_initializer(shape=(self.outdim,self.indim)),
            self.bias_initializer(shape=(self.outdim,1))
        ]

class Input(Layers):
    def __init__(self, inputdim):
        super().__init__("inputs", inputdim)

class Dense(Layers):
    def __init__(self, units, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros'):
        """
        @param units             : (tuple) dimensionality of the (input space, output space).
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        super().__init__("dense", units)
        self.kernel_initializer = Initializer(kernel_initializer)
        self.bias_initializer   = Initializer(bias_initializer)
        self.h = ActivationFunc(activation)
        self.z = None
        self.a = None

    def forward(self, input):
        """ @param input: shape=(Din,) """
        z_in = np.append(input,1) # shape=(Din+1,)
        a = self.w.dot(z_in)      # (Dout,Din+1) @ (Din+1,) = (Dout,)
        z_out = self.h.forward(a) # shape=(Dout,)
        self.z = z_in
        self.a = a
        return z_out

    def backprop(self, dEdz_out):
        """ @param dEdz_out: shape=(Dout,) """
        dEda = self.h.diff(self.a)*dEdz_out # δ, shape=(Dout,)
        dEdz_in = self.w.T.dot(dEda)        # (Din+1,Dout) @ (Dout,) = (Din+1,)
        self.update(dEda)
        return dEdz_in[:-1]                 # shape=(Din,) term of bias is not propagated.

    def update(self, delta, ALPHA=0.01):
        """ @param delta: shape=(Dout,) """
        dw = np.outer(delta, self.z) # (Dout,) × (Din+1,) = (Dout,Din+1)
        self.w -= ALPHA*dw # update. w → w + ALPHA*dw
