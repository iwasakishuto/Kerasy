# coding: utf-8
from __future__ import absolute_import

import numpy as np
from collections import defaultdict

def clip_norm(grads,c,n):
    """Ref: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/optimizers.py#L21
    Clip the gradient `g` if the L2 norm `n` exceeds `c`.
    =====================================================
    @param  g: (ndarray) Gradient.
    @param  c: (float)   Gradients will be clipped when their L2 norm exceeds this value.
    @param  n: (ndarray) Actual norm of `g`
    @return g: (ndarray) Gradient clipped if required.
    """
    if c<=0: return g
    g = c/n*g if n>=c else g
    return g

class Optimizer():
    """Abstract optimizer base class.
    ALL Kerasy optimizer support the following keyword arguments:
    @param clipnorm : (float >= 0) Gradients will be clipped when their L2 norm exceeds this value.
    @param clipvalue: (float >= 0) Gradients will be clipped when their absolute value exceeds this value.
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {"clipnorm", "clipvalue"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError(f'Unexpecte keyword argument passed to optimizer: {str(k)}')

        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []
        self.optparams = defaultdict(lambda: defaultdict(list))

    def get_updates(self, grads, past_params, name):
        """ This is the parent class of all optimizer, not an actual optimizer. """
        raise NotImplementedError

    def get_gradients(self, grads):
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = np.sqrt(sum([np.sum(np.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [np.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

class SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    @param learning_rate: (float) Learning rate.
    @param momentum     : (float) Parameter that accelerates SGD in the relevant direction and dampens oscillations.
    @param nesterov     : (bool)  Whether to apply Nesterov momentum.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate) # dict.pop(key,d): If key is not found, d is returned
        self.initial_decay = kwargs.pop('decay', 0.0)
        super().__init__(**kwargs)
        self.iterations = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = self.initial_decay
        self.nesterov = nesterov

    def get_updates(self, grads, past_params, name):
        """
        @grads      : (ndarray) shape=(x,y)
        @past_params: (ndarray) shape=(z,x,y)
        @name       : (str) layer name + param name.
        """
        grads = self.get_gradients(grads) # Cliped.
        # Increase processing speed by storing in local variables.
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay*self.iterations))
        if len(past_params) <= 1:
            dParam = np.zeros_like(grads)
            Last = past_params[-1]
        else:
            preLast, Last = past_params[-2:]
            dParam = Last-preLast
        velocity = self.momentum*dParam - self.learning_rate*grads
        if self.nesterov:
            new_param = Last + self.momentum*velocity - self.lr*grads
        else:
            new_param = Last + velocity
        return new_param
