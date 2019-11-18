# coding: utf-8
from __future__ import absolute_import

import numpy as np

from ..activations import ActivationFunc
from ..initializers import Initializer
from ..losses import LossFunc
from ..engine.base_layer import Layer

class Conv2D(Layer):
    def __init__(self, filters, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu',
                 kernel_initializer='random_normal', bias_initializer='zeros', data_format='channels_last', **kwargs):
        """
        @param filters           : (int) the dimensionality of the output space.
        @param kernel_size       : (int,int) height and width of each kernel. kernel-shape=(*kernel_size, F, OF)
        @param strides           : (int,int) height and width of stride step.
        @param padding           : (str) "valid" or "same". Padding method.
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        self.OF = filters # Output filters.
        self.kh, self.kw = kernel_size # kernel size.
        self.sh, self.sw = strides
        if padding not in ["same", "valid"]: raise ValueError("padding must be 'same' or 'valid'. Please chose one of them.")
        self.padding = padding
        self.h = ActivationFunc(activation)
        self.kernel_initializer = Initializer(kernel_initializer)
        self.kernel_regularizer = None
        self.kernel_constraint  = None
        self.bias_initializer   = Initializer(bias_initializer)
        self.bias_regularizer   = None
        self.bias_constraint    = None
        self.use_bias = True
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3: raise ValueError(f"The input shape of {self.name} must be 3-dimension (height, width, channel). However it is {len(input_shape)}-dimension.")
        self.H, self.W, self.F = input_shape
        if self.padding=="same":
            self.OH = self.H
            self.OW = self.W
            self.ph = ((self.sh-1)*self.H+self.kh-self.sh)//2
            self.pw = ((self.sw-1)*self.W+self.kw-self.sw)//2
        elif self.padding=="valid":
            self.OH = (self.H-self.kh)//self.sh+1
            self.OW = (self.W-self.kw)//self.sw+1
            self.ph = 0
            self.pw = 0
        self.output_shape = (self.OH, self.OW, self.OF)
        return self.output_shape

    def build(self, input_shape):
        """ @params input_shape: (H,W,F) of input image. """
        output_shape = self.compute_output_shape(input_shape)
        self.kernel  = self.add_weight(shape=(self.kh, self.kw, self.F, self.OF),
                                       name="kernel",
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       constraint =self.kernel_constraint,
                                       trainable  =self.trainable)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.OF,),
                                        name="bias",
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint =self.bias_constraint,
                                        trainable  =self.trainable)
        else:
            self.bias = None
        return output_shape

    def _paddInput(self, input):
        """ @param input: (ndarray) must be a 3-D array. shape=(H,W,F) """
        #=== padding ===
        if self.padding =="same":
            Xin = np.zeros(shape=(self.H+2*self.ph,self.W+2*self.pw,self.F))
            Xin[self.ph:self.H+self.ph,self.pw:self.W+self.pw,:] = input
        elif self.padding == "valid":
            Xin = input[:self.sh*self.OH+self.kh-1, :self.sw*self.OW+self.kw-1, :]
        else:
            TypeError(f"Can't understand 'padding=f{self.padding}'")
        self.Xin = Xin # Memorize
        return Xin

    def _generator(self, Xin):
        #=== generator ===
        for i in range(self.OH//self.sh):
            for j in range(self.OW//self.sw):
                clipedXin = Xin[self.sh*i:(self.sh*i+self.kh), self.sw*j:(self.sw*j+self.kw), :]
                yield clipedXin,i,j

    def forward(self, input):
        """ @param input: (ndarray) 3-D array. shape=(H,W,F) """
        Xin  = self._paddInput(input)
        a    = np.zeros(shape=(self.OH, self.OW, self.OF))
        for c in range(self.OF):
            for clip_image,i,j in self._generator(Xin):
                """ 'self.kernel[:,:,:,c]' and 'clip_image' shapes equal in (kh,kw,F) """
                a[i,j,c] = np.sum(clip_image*self.kernel[:,:,:,c])
        a     += self.bias # (OH,OW,OF) + (OF,) = (OH,OW,OF)
        self.a = a # Memorize. (output layer. shape=(OH,OW,OF))
        Xout = self.h.forward(a)
        return Xout

    def _backprop_mask(self,i,j,m,n):
        return ((i%self.sh+m < self.kh) and (j%self.sw+n < self.kw)) and ((self.OH > (i-m)//self.sh >= 0) and (self.OW > (j-n)//self.sw >= 0))

    def backprop(self, dEdXout, lr=1e-3):
        """
        @param  delta_times_w: shape=(OH,OW,OF)
        @param  self.kernel  : shape=(self.kh, self.kw, self.F, self.OF)
        @return delta_times_w: shape=(H,W,F)
        """
        dEda = dEdXout*self.h.diff(self.a) # Xout=h(a) → dE/da = dE/dXout*h'(a)
        dEdXin = np.zeros_like(self.Xin)   # shape=(H+2ph,W+2pw,F)
        for c in range(self.F):
            for i in range(self.H+2*self.ph):
                for j in range(self.W+2*self.pw):
                    dEdXin[i,j,c] = np.sum([ dEda[(i-m)//self.sh,(j-n)//self.sw,:] * self.kernel[i%self.sh+m,j%self.sw+n,c,:] for m in range(0,self.kh,self.sh) for n in range(0,self.kw,self.sw) if self._backprop_mask(i,j,m,n)])
        if self.trainable: self.memorize_delta(dEda)
        return dEdXin[self.ph:-self.ph,self.pw:-self.pw,:]

    def memorize_delta(self, dEda):
        dEdw = np.zeros(shape=self.kernel.shape) # shape=(kh, kw, F, OF)
        for m in range(self.kh):
            for n in range(self.kw):
                for c in range(self.F):
                    for c_ in range(self.OF):
                        # dEdw_{m,n,c,c'} = ΣiΣj(dEda * dadw) = ΣiΣj(dEda_{i,j,c'}*Xin_{i+m,j+n,c})
                        dEdw[m,n,c,c_] = np.sum(dEda[:,:,c_] * self.Xin[m:m+self.OH:self.sh, n:n+self.OW:self.sw, c])
        self._losses['kernel'] += dEdw
        self._losses['bias'] += np.sum(dEda, axis=(0,1))
