# coding: utf-8
from __future__ import absolute_import

import numpy as np

from .. import activations
from .. import initializers
from .. import regularizers
from ..engine.base_layer import Layer

from ..utils import flush_progress_bar
from ..utils import handleRandomState
from ..utils import set_weight

class Input(Layer):
    def __init__(self, input_shape, **kwargs):
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape=self.output_shape=input_shape
        super().__init__(**kwargs)
        self.trainable = False

    def forward(self, input):
        return input

    def backprop(self, delta):
        return delta

class Flatten(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape), )
        return self.output_shape

    def forward(self, input):
        # return input.flatten()
        return np.ravel(input)

    def backprop(self, delta):
        return delta.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, units, activation='linear',
                 kernel_initializer='random_normal', kernel_regularizer='none',
                 bias_initializer='zeros', bias_regularizer='none', **kwargs):
        """
        @param units             : (tuple) dimensionality of the (input space, output space).
        @param activation        : (str) Activation function to use.
        @param kernel_initializer: (str) Initializer for the `kernel` weights matrix.
        @param bias_initializer  : (str) Initializer for the bias vector.
        """
        self.output_shape=(units,)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = None
        self.bias_initializer   = initializers.get(bias_initializer)
        self.bias_regularizer   = regularizers.get(bias_regularizer)
        self.bias_constraint    = None
        self.activation = activations.get(activation)
        self.use_bias = True
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return self.output_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        output_shape = self.compute_output_shape(input_shape)
        self.kernel  = self.add_weight(
            shape=(self.output_shape + input_shape),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint =self.kernel_constraint
        )
        self.bias = self.add_weight(
            shape=(self.output_shape[0],1),
            name="bias",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint =self.bias_constraint
        )
        return output_shape

    def forward(self, input):
        """ @param input: shape=(Din,) """
        if self.use_bias:
            Xin = np.append(input,1) # shape=(Din+1,)
            a = np.c_[self.kernel, self.bias].dot(Xin) # (Dout,Din+1) @ (Din+1,) = (Dout,)
        else:
            Xin = input # shape=(Din,)
            a = self.kernel.dot(Xin) # (Dout,Din) @ (Din,) = (Dout,)
        Xout = self.activation.forward(input=a) # shape=(Dout,)
        self.a = a
        self.Xin = Xin # shape=(Din+1,) or shape=(Din,)
        return Xout

    def backprop(self, dEdXout):
        """ @param dEdXout: shape=(Dout,) """
        # dXoutda = self.activation.diff(self.a) # shape=(Dout,)
        # dEda = dEdXout.dot(dXoutda) if dXoutda.ndim==2 else dEdXout * dXoutda # shape=(Dout,)
        dEda = self.activation.diff(self.a) * dEdXout
        if self.trainable:
            self.memorize_delta(dEda)

        if self.use_bias:
            dEdXin = np.c_[self.kernel, self.bias].T.dot(dEda) # (Din+1,Dout) @ (Dout,) = (Din+1,)
            dEdXin = dEdXin[:-1] # delta of bias is not propagated.
        else:
            dEdXin = self.kernel.T.dot(dEda)
        return dEdXin # shape=(Din,)

    def memorize_delta(self, dEda):
        dEdw = np.outer(dEda, self.Xin)
        if self.use_bias:
            self._grads['kernel'] += dEdw[:,:-1] # shape=(Dout, Din)
            self._grads['bias'] += dEdw[:,-1:] # shape=(Dout, 1)
        else:
            self._grads['kernel'] += dEdw

    def get_weights(self):
        return [self.kernel, self.bias]

    def set_weights(self, weights):
        if len(weights)==1:
            kernel = weights[0]
        elif len(weights)==2:
            kernel, bias = weights
            set_weight(self.bias, bias)
        else:
            raise ValueError(f"Dense has 2 weights, but len(weights)={len(weights)}")
        set_weight(self.kernel, kernel)

class Dropout(Layer):
    def __init__(self, keep_prob, random_state=None, **kwargs):
        self.keep_prob = min(1., max(0., keep_prob))
        super().__init__(**kwargs)
        self.trainable = False
        self.rnd = handleRandomState(random_state)

    def compute_output_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        return input_shape

    def forward(self, input):
        self.mask = self.rnd.uniform(low=0., high=1., size=input.shape)<=self.keep_prob
        return input * self.mask

    def backprop(self, delta):
        return delta * self.mask
