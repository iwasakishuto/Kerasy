# coding: utf-8
from __future__ import absolute_import

import re
import numpy as np
from collections import deque, defaultdict
from abc import ABCMeta, abstractmethod
from .utils import mk_class_get

def clip_norm(g, c, n):
    """Ref: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/optimizers.py#L21
    Clip the gradient `g` if the L2 norm `n` exceeds `c`.
    =====================================================
    @param  g: (ndarray) Gradient.
    @param  c: (float)   Gradients will be clipped when their L2 norm exceeds this value.
    @param  n: (ndarray) Actual norm of `g`
    @return g: (ndarray) Gradient clipped if required.
    """
    if c<=0 or n<c:
        return g
    else:
        return c/n*g

class KerasyAbstOptimizer(metaclass=ABCMeta):
    """Abstract optimizer base class.
    ALL Kerasy optimizer support the following keyword arguments:
    @param clipnorm : (float >= 0) Gradients will be clipped when their L2 norm exceeds this value.
    @param clipvalue: (float >= 0) Gradients will be clipped when their absolute value exceeds this value.
    """
    def __init__(self, **kwargs):
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()
        allowed_kwargs = {"clipnorm", "clipvalue"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError(f'Unexpecte keyword argument passed to optimizer: {str(k)}')

        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []
        self.optparams = defaultdict(lambda: defaultdict(list))

    @abstractmethod
    def get_updates(self, grad, curt_param, name):
        """ This is the parent class of all optimizer, not an actual optimizer. """
        pass

    def get_gradient(self, grad):
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            l2norm = np.linalg.norm(grad)
            grad = clip_norm(grad, self.clipnorm, l2norm)
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grad = np.clip(grad, -self.clipvalue, self.clipvalue)
        return grad

class GradientDescent(KerasyAbstOptimizer):
    def __init__(self, eta=0.01, **kwargs):
        self.eta = eta
        super().__init__(**kwargs)

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.
        return curt_param - self.eta*grad

class SGD(KerasyAbstOptimizer):
    """ (Momentum SGD) Stochastic gradient descent optimizer.
    @param learning_rate: (float) Learning rate.
    @param momentum     : (float) Parameter that accelerates SGD in the relevant direction and dampens oscillations.
    @param nesterov     : (bool)  Whether to apply Nesterov momentum.
    """
    def __init__(self, learning_rate=0.01, momentum=0., nesterov=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate) # dict.pop(key,d): If key is not found, d is returned
        self.initial_decay = kwargs.pop('decay', 0.0)
        super().__init__(**kwargs)
        self.iterations = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = self.initial_decay
        self.nesterov = nesterov
        self.moments = defaultdict()

    def get_updates(self, grad, curt_param, name):
        """
        @grad      : (ndarray) shape=(x,y)
        @curt_param: (ndarray) shape=(z,x,y)
        @name      : (str) layer name + param name.
        """
        grad = self.get_gradient(grad) # Cliped.
        self.iterations += 1

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        m = self.moments.get(name, np.zeros_like(curt_param))
        v = self.momentum*m - lr*grad
        self.moments[name] = v

        if self.nesterov:
            new_param = curt_param + self.momentum*v - lr*grad
        else:
            new_param = curt_param + v

        return new_param

KerasyOptimizerClasses = {
    'sgd': SGD,
    'gra': GradientDescent,
}

get = mk_class_get(
    all_classes=KerasyOptimizerClasses,
    kerasy_abst_class=[KerasyAbstOptimizer],
    genre="optimizer"
)
