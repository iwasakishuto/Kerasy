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

    def forward(self, input):
        return input

    def backprop(self, delta):
        return delta

class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(list(input_shape)),)
        return self.output_shape

    def forward(self, input):
        return input.flatten()

    def backprop(self, delta):
        return delta.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, units, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros', **kwargs):
        """
        @param units             : (tuple) dimensionality of the (input space, output space).
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        self.output_shape=(units,)
        self.kernel_initializer = Initializer(kernel_initializer)
        self.kernel_regularizer = None
        self.kernel_constraint  = None
        self.bias_initializer   = Initializer(bias_initializer)
        self.bias_regularizer   = None
        self.bias_constraint    = None
        self.h = ActivationFunc(activation)
        self.use_bias = True
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return self.output_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        output_shape = self.compute_output_shape(input_shape)
        self.kernel  = self.add_weight(shape=(self.output_shape + input_shape),
                                       name="kernel",
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       constraint =self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_shape[0],1),
                                        name="bias",
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint =self.bias_constraint)
        else:
            self.bias = np.empty(shape=(self.output_shape[0],1))
        return output_shape

    def forward(self, input):
        """ @param input: shape=(Din,) """
        Xin  = np.append(input,1) if self.use_bias else input # shape=(Din+1,)
        a    = np.c_[self.kernel, self.bias].dot(Xin) # (Dout,Din+1) @ (Din+1,) = (Dout,)
        Xout = self.h.forward(input=a) # shape=(Dout,)
        self.a = a
        self.Xin = Xin # shape=(Din+1,) or shape=(Din,)
        return Xout

    def backprop(self, dEdXout):
        """ @param dEdXout: shape=(Dout,) """
        dXoutda = self.h.diff(self.a) # shape=(Dout,)
        dEda = dEdXout.dot(dXoutda) if len(dXoutda.shape)==2 else dEdXout * dXoutda # shape=(Dout,)
        dEdXin = np.c_[self.kernel, self.bias].T.dot(dEda) # (Din+1,Dout) @ (Dout,) = (Din+1,)
        self.memorize_delta(dEda)
        dEdXin = dEdXin[:-1] if self.use_bias else dEdXin
        return dEdXin # shape=(Din,) delta of bias is not propagated.

    def memorize_delta(self, dEda):
        dEdw = np.outer(dEda, self.Xin) # (Dout, Din+1)
        if self.use_bias:
            self._losses['kernel'] += dEdw[:,:-1] # shape=(Dout, Din)
            self._losses['bias'] += dEdw[:,-1:] # shape=(Dout, 1)
        else:
            self._losses['kernel'] += dEdw
