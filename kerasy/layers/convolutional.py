# coding: utf-8
from __future__ import absolute_import

import numpy as np

from ..activations import ActivationFunc
from ..initializers import Initializer
from ..losses import LossFunc
from ..layers.core import Input, Dense

class Input():
    def __init__(self, input_shape):
        self.output_shape=input_shape

class Conv2D():
    def __init__(self, filters, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu',
                 kernel_initializer='random_normal', bias_initializer='zeros', data_format='channels_last'):
        """
        @param filters           : (int) the dimensionality of the output space.
        @param kernel_size       : (int,int)
        @param strides           : (int,int)
        @param padding           : (str) "valid" or "same".
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        self.OF = filters # Output filters.
        self.kh, self.kw = kernel_size # kernel size.
        self.sh, self.sw = strides
        self.padding = padding
        self.h = ActivationFunc(activation)
        self.kernel_initializer = Initializer(kernel_initializer)
        self.bias_initializer   = Initializer(bias_initializer)

    def build(self, input_shape):
        """ @params input_shape: (H,W,F) of input image. """
        self.H, self.W, self.F = input_shape
        try:
            if self.padding=="same":
                self.OH=self.H; self.OW=self.W
                self.ph = ((self.sh-1)*self.H+self.kh-self.sh)//2
                self.pw = ((self.sw-1)*self.W+self.kw-self.sw)//2
            elif self.padding=="valid":
                self.OH=self.H-self.kh+1; self.OW=self.W-self.kw+1
                while self.OH%self.sh != 0:
                    self.OH-=1
                while self.OW%self.sw != 0:
                    self.OW-=1
                self.ph, self.pw = (0,0)
        except:
            print("Can't understand 'padding=f{self.padding}'. Please chose 'same' or 'valid'.")

        self.kernel = self.kernel_initializer(shape=(self.kh, self.kw, self.F, self.OF))
        self.bias   = self.bias_initializer(shape=(self.OF,))
        self.output_shape=(self.OH, self.OW, self.OF)

    def _generator(self, input):
        """ @param input: (ndarray) must be a 3-D array. (Height, Width, Channel) """
        #=== padding ===
        try:
            if self.padding =="same":
                z_in = np.zeros(shape=(self.H+2*self.ph,self.W+2*self.pw))
                z_in[self.ph:self.H+self.ph,self.pw:self.W+self.pw,:] = input
            elif self.padding == "valid":
                z_in = input[:self.OH,:self.OW,:]
        except:
            print("Can't understand 'padding=f{self.padding}'")

        for i in range(0,self.H-self.kh+1,self.sh):
            for j in range(0,self.W-self.kw+1,self.sw):
                clip_image = input[i:(i+self.kh), j:(j+self.kw), :]
                yield clip_image,i,j

    def forward(self, z_in):
        """ @param z_in: (ndarray) 3-D array. """
        self.z = z_in # Memorize. (input layer. shape=(H,W,F))
        z_out = np.zeros(shape=(self.OH, self.OW, self.OF))
        for f in range(self.OF):
            for clip_image,i,j in self._generator(z_in):
                """ 'self.kernel[:,:,:,f]' and 'clip_image' shapes equal in (kh,kw,F) """
                z_out[i][j][f] = np.sum(clip_image*self.kernel[:,:,:,f])
        a = z_out + self.bias # (OH,OW,OF) + (OF,) = (OH,OW,OF)
        self.a = a # Memorize. (output layer. shape=(OH,OW,OF))
        return a
        """
        [Memo: Broadcasting]
        ====================
            x = np.zeros(shape=(2,2,3)) # shape=(2,2,3)
            y = np.array([1,2,3])       # shape=(3,)
            z = x + y                   # shape=(2, 2, 3)
            print(z)
            >>>[[[1. 2. 3.]
                 [1. 2. 3.]]

                [[1. 2. 3.]
                 [1. 2. 3.]]]
        """

    def backprop(self, delta_times_w, lr=1e-3):
        """ @param delta_times_w: (ndarray) 3-D array. shape=(OH,OW,OF) """
        delta = self.h.diff(self.a) * delta_times_w # shape=(OH,OW,OF)
        delta_times_w = np.zeros(shape=(self.H,self.W,self.F))

        delta_padd = np.zeros(shape=(self.H+self.kh-1, self.W+self.kw-1, OF))
        """
        [Aim] ex.) i=1,j=2,M=3,N=3
        =========================
        δ[1-0][2-0], δ[1-0][2-1], δ[1-0][2-2]        δ[1][2], δ[1][1], δ[1][0]
        δ[1-1][2-0], δ[1-1][2-1], δ[1-1][2-2]  ===>  δ[0][2], δ[0][1], δ[0][0]
        δ[1-2][2-0], δ[1-2][2-1], δ[1-2][2-2]              0,       0,      0
        """
        delta_padd[self.kh-1:self.kh-1+self.OH, self.kw-1:self.kw-1+self.OW, :] = delta[:self.H, :self.W, :]
        for i in range(self.kh-1,self.kh-1+self.H):
            for j in range(self.kw-1,self.kw-1+self.W):
                for f in range(self.F):
                    delta_times_w[i][j][f] = np.sum(np.flip(delta_padd[i:i+self.kh-1,j:j+self.kw-1,:])*self.w[:,:,f,:])

        self.update(delta)
        return delta_times_w

    def update(self, delta, ALPHA=0.01):
        """ @param delta: shape=(OH,OW,OF) """
        # Kernels
        z_padd = np.zeros(shape=(self.H+self.kh-1, self.W+self.kw-1, F))
        z_padd[:self.H,:self.W,:] = self.z

        dEdw = np.zeros(shape=self.kernel.shape) # shape=(kh, kw, F, OF)
        for m in range(self.kh):
            for n in range(self.kw):
                dEdw[m][n][f] = np.sum(np.expand_dims(z_padd[m:m+self.H,n:n+self.W,:], axis=3)*np.expand_dims(delta[i,j,:], axis=2))
        self.kernel -= ALPHA*dEdw

        # bias
        self.bias -= ALPHA*np.sum(delta, axis=(1,2))


class Sequential():
    def __init__(self, layer=None):
        self.layers = []
        self.epochs = 0
        if layer is not None: self.add(layer)
        self.loss = None
        self.config = None

    def is_valid(self):
        """ Check whether this is correct Sequential instance. """
        return isinstance(self.layers[0],Input)

    def add(self, layer):
        """ Adds a layer instance. """
        self.layers.append(layer)

    def compile(self, loss, input_shape=None):
        """ Creates the layer weights. """
        if not self.is_valid:
            print(
                "Please specify the Input shape like:\n\
                model = Sequential()\n\
                model.add(Input(input_shape=(28,28,1))\n\
                ："
            )
        else:
            input_shape = self.layers[0].output_shape
            for l in self.layers[1:]:
                l.build(input_shape=input_shape)
                input_shape=l.output_shape
            self.loss = LossFunc(loss)

    def forward(self, input):
        if self.is_valid:
            out=input
        for l in self.layers[1:]:
            out=l.forward(out)
        return out

    def backprop(self, y_true, out):
        delta_times_w = self.loss.diff(y_true, out)
        if self.is_valid:
            for l in reversed(self.layers):
                for l in self.layers[1:]:
                    delta_times_w = l.backprop(delta_times_w)

class Flatten():
    def __init__(self):
        self.input=None
        
    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(list(input_shape)),)
        
    def forward(self, input):
        return input.flatten()

    def backprop(delta):
        return delta.reshape(self.input_shape)

class MaxPooling2D():
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
    def __init__(self, pool_size=(2, 2)):
        self.input = None
        self.pool_size = pool_size
        
    def build(self, input_shape):
        self.H, self.W, self.F = input_shape
        ph,pw = self.pool_size
        self.OH = self.H//ph
        self.OW = self.W//pw
        self.OF = self.F
        self.output_shape=(self.OH, self.OW, self.OF)

    def _generator(self, image):
        """ Generator for training. """
        h,w,_ = image.shape
        ph,pw = self.pool_size
        for i in range(self.H//ph):
            for j in range(self.W//pw):
                crip_image = image[i*ph:(i+1)*ph,j*pw:(j+1)*pw]
                yield crip_image, i, j

    def forward(self, input):
        self.input = input # Memorize the input array. (not shape)
        h,w,f = image.shape
        ph,pw = self.pool_size
        out = np.zeros((h//ph, w//pw, f)) # output image shape.
        for crip_image, i, j in self._generator(image):
            out[i][j] = np.amax(crip_image, axis=(0, 1))
        return out

    def backprop(self, Pre_delta, lr=1e-3):
        """ Loss only flows to the pixel that takes the maximum value in pooling block. """
        Next_delta = np.zeros(self.input.shape)
        for crip_image, i, j in self._generator(self.input):
            Next_delta[i*ph:(i+1)*ph,j*pw:(j+1)*pw] = np.where(crip_image==np.max(crip_image), np.max(crip_image), 0)

        return Next_delta

class Dense():
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
    
    def build(self, input_shape):
        self.input_shape = input_shape
        self.w = np.c_[
            self.kernel_initializer(shape=(self.output_shape[0],self.input_shape[0])),
            self.bias_initializer(shape=(self.output_shape[0],1))
        ]

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
