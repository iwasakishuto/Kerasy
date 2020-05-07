# coding: utf-8
import re
import numpy as np
from abc import ABCMeta, abstractmethod
from .utils import mk_class_get

class KerasyAbstActivation(metaclass=ABCMeta):
    def __init__(self):
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def diff(self, delta):
        pass

class Linear(KerasyAbstActivation):
    """ Do nothing. """
    def forward(self, input):
        return input

    def diff(self, delta):
        return 1.

class Softmax(KerasyAbstActivation):
    """ Softmax function, also known as 'softargmax' or 'normalized exponential function' """
    def __init__(self):
        self.exps = None
        self.S = None
        super().__init__()

    def forward(self, input):
        """ @param input shape=(class,) """
        self.exps = np.exp(input - np.max(input)) # For avoiding overflowing.
        self.S = np.sum(self.exps)
        return self.exps/self.S

    def diff(self, delta):
        """
        @param  delta : shape=(class,)
        @return delta : shape=(class,)
        =================================
        diff = ( np.sum((-t/y)*exp*(-1/S**2)) + (-t/y)*(1/S) )*exp
        """
        return (-np.sum(delta*self.exps)/(self.S**2) + delta/self.S)*self.exps

class Tanh(KerasyAbstActivation):
    def forward(self, input):
        return np.tanh(input)

    def diff(self, delta):
        return 1-np.tanh(delta)**2

class Relu(KerasyAbstActivation):
    def forward(self, input):
        self.mask = input>0
        return np.where(self.mask, input, 0.)

    def diff(self, delta):
        return np.where(self.mask, 1., 0.)

class Sigmoid(KerasyAbstActivation):
    def __init__(self):
        self.out = None
        super().__init__()

    def forward(self, input):
        out = 1/(1+np.exp(input))
        self.out = out # Memorize!!
        return out

    def diff(self, delta):
        return delta * (1-self.out) * self.out

KerasyActivationClasses = {
    'linear' : Linear,
    'softmax': Softmax,
    'tanh'   : Tanh,
    'relu'   : Relu,
    'sigmoid': Sigmoid,
}

get = mk_class_get(
    all_classes=KerasyActivationClasses,
    kerasy_abst_class=[KerasyAbstActivation],
    genre="activation"
)
