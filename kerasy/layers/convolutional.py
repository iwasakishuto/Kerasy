# coding: utf-8
from __future__ import absolute_import

import numpy as np

from .. import activations
from .. import initializers
from .. import regularizers
from ..engine.base_layer import Layer

from ..utils import handleKeyError
from ..utils import set_weight
from ..clib import c_deep

class Conv2D(Layer):
    def __init__(self, filters, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu',
                 kernel_initializer='random_normal', kernel_regularizer='none',
                 bias_initializer='zeros', bias_regularizer='none',
                 data_format='channels_last', **kwargs):
        """
        @param filters           : (int) the dimensionality of the output space.
        @param kernel_size       : (int,int) height and width of each kernel. kernel-shape=(*kernel_size, F, OF)
        @param strides           : (int,int) height and width of stride step.
        @param padding           : (str) "valid" or "same". Padding method.
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        handleKeyError(lst=["same", "valid"], padding=padding)
        self.OF = filters # Output filters.
        self.kh, self.kw = kernel_size # kernel size.
        self.sh, self.sw = strides
        self.padding = padding
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = None
        self.bias_initializer   = initializers.get(bias_initializer)
        self.bias_regularizer   = regularizers.get(bias_regularizer)
        self.bias_constraint    = None
        self.use_bias = True
        super().__init__(**kwargs)

    @property
    def input_shape(self):
        return (self.H, self.W, self.F)
    @property
    def output_shape(self):
        return (self.OH, self.OW, self.OF)

    @property
    def kernel_size(self):
        return (self.kh, self.kw)

    @property
    def strides(self):
        return (self.sh, self.sw)

    @property
    def padding_size(self):
        return (self.ph, self.pw)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"The input shape of {self.name} must be 3-dimension (height, width, channel). However it is {len(input_shape)}-dimension.")
        self.H, self.W, self.F = input_shape
        if self.padding=="same":
            self.OH = self.H
            self.OW = self.W
            self.ph = ((self.sh-1)*self.H+self.kh-self.sh)//2
            self.pw = ((self.sw-1)*self.W+self.kw-self.sw)//2
            self.padded_input_shape = (self.H+2*self.ph, self.W+2*self.pw, self.F)
            self.padding_input = self._padding_input_same_with_zero
        elif self.padding=="valid":
            self.OH = (self.H-self.kh)//self.sh+1
            self.OW = (self.W-self.kw)//self.sw+1
            self.ph = 0
            self.pw = 0
            self.padded_input_shape = (self.sh*self.OH+self.kh-1, self.sw*self.OW+self.kw-1, self.F)
            self.padding_input = self._padding_input_valid
        return self.output_shape

    def build(self, input_shape):
        """ @params input_shape: (H,W,F) of input image. """
        output_shape = self.compute_output_shape(input_shape)
        self.kernel  = self.add_weight(
            shape=(self.kh, self.kw, self.F, self.OF),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint =self.kernel_constraint,
            trainable  =self.trainable
        )
        self.bias = self.add_weight(
            shape=(self.OF,),
            name="bias",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint =self.bias_constraint,
            trainable  =self.trainable
        )
        return output_shape

    def _padding_input_same_with_zero(self, input):
        Xin = np.zeros(shape=self.padded_input_shape)
        Xin[self.ph:self.H+self.ph,self.pw:self.W+self.pw,:] = input
        return Xin

    def _padding_input_valid(self, input):
        Xin = input[:self.sh*self.OH+self.kh-1, :self.sw*self.OW+self.kw-1, :]
        return Xin

    def forward(self, input):
        """ @param input: (ndarray) 3-D array. shape=(H,W,F) """
        self.Xin = self.padding_input(input)
        a   = np.empty(shape=self.output_shape)
        c_deep.Conv2D_forward(
            a, self.Xin, self.kernel,
            self.OH, self.OW,
            *self.strides, *self.kernel_size,
        )
        if self.use_bias:
            a += self.bias # (OH,OW,OF) + (OF,) = (OH,OW,OF)
        self.a = a
        Xout = self.activation.forward(a)
        return Xout

    def backprop(self, dEdXout):
        dEda = dEdXout*self.activation.diff(self.a) # Xout=h(a) → dE/da = dE/dXout*h'(a)
        dEdXin = np.empty(shape=self.padded_input_shape)   # shape=(H+2ph,W+2pw,F)
        dEdw = np.empty_like(self.kernel) # shape=(kh, kw, F, OF)
        c_deep.Conv2D_backprop(
            dEda, dEdXin, dEdw, self.Xin, self.kernel,
            *self.padded_input_shape, *self.output_shape, *self.strides, *self.kernel_size,
            trainable=self.trainable
        )
        if self.trainable:
            self._grads['kernel'] += dEdw
            self._grads['bias'] += np.sum(dEda, axis=(0,1))
        return dEdXin[self.ph:self.H+self.ph,self.pw:self.W+self.pw,:]

    def get_weights(self):
        return [self.kernel, self.bias]

    def set_weights(self, weights):
        if len(weights)==1:
            kernel = weights[0]
        elif len(weights)==2:
            kernel, bias = weights
            set_weight(self.bias, bias)
        else:
            raise ValueError(f"Conv2D has 2 weights, but len(weights)={len(weights)}")
        set_weight(self.kernel, kernel)

    # old version 1 : https://github.com/iwasakishuto/Kerasy/blob/c6a896834be7703e0454ba44ffcd8a66e5de197c/kerasy/layers/convolutional.py#L94
    # old version 2 : https://github.com/iwasakishuto/Kerasy/blob/ff8aab0d3f32a5d5cfb28f38deecfc8c561184b3/kerasy/layers/convolutional.py#L99
