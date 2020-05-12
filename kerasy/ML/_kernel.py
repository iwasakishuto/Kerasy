# coding: utf-8
import re
import numpy as np
from scipy import stats
from abc import ABCMeta, abstractmethod

from ..utils import mk_class_get

class KerasyAbstKernel(metaclass=ABCMeta):
    def __init__(self):
        self.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()

    @abstractmethod
    def __call__(self, x, x_prime):
        raise NotImplementedError

class Linear(KerasyAbstKernel):
    def __init__(self, c=0):
        self.c = c
        super().__init__()

    def __call__(self, x, x_prime):
        return x.T.dot(x_prime) + self.c

class Polynomial(KerasyAbstKernel):
    def __init__(self, alpha=1, c=0, d=3):
        self.alpha = alpha
        self.c = c
        self.d = d
        super().__init__()

    def __call__(self, x, x_prime):
        return (self.alpha*x.T.dot(x_prime) + self.c)**self.d

class Gaussian(KerasyAbstKernel):
    def __init__(self, sigma=1):
        self.sigma = sigma
        super().__init__()

    def __call__(self, x, x_prime):
        return np.exp(-sum((x-x_prime)**2)/(2*self.sigma**2))

class Exponential(KerasyAbstKernel):
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        super().__init__()

    def __call__(self, x, x_prime):
        return np.exp(-sum(abs(x-x_prime))/(2*self.sigma**2))

class Laplacian(KerasyAbstKernel):
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        super().__init__()

    def __call__(self, x, x_prime):
        return np.exp(-sum(abs(x-x_prime))/self.sigma)

class HyperbolicTangent(KerasyAbstKernel):
    def __init__(self, alpha=1, c=0):
        self.alpha = alpha
        self.c = c
        super().__init__()

    def __call__(self, x, x_prime):
        return np.tanh(self.alpha*x.T.dot(x_prime) + self.c)

class RationalQuadratic(KerasyAbstKernel):
    def __init__(self, c=1):
        self.c = c
        super().__init__()

    def __call__(self, x, x_prime):
        return 1 - sum((x-x_prime)**2)/(sum((x-x_prime)**2)+self.c)

class Multiquadric(KerasyAbstKernel):
    def __init__(self, c=1):
        self.c = c

    def __call__(self, x, x_prime):
        return np.sqrt(sum((x-x_prime)**2) + self.c)

class InverseMultiquadric(KerasyAbstKernel):
    def __init__(self, c=1):
        self.multiquadric = Multiquadric(c=c)

    def __call__(self, x, x_prime):
        return 1/self.multiquadric(x,x_prime)

class Log(KerasyAbstKernel):
    def __init__(self, d=3):
        self.d = d

    def __call__(self, x, x_prime):
        return -np.log(sum(abs(x-x_prime)**self.d) + 1)

all = KerasyKernelFunctions = {
    "linear"               : Linear,
    "polynomial"           : Polynomial,
    "gaussian"             : Gaussian,
    "exponential"          : Exponential,
    "laplacian"            : Laplacian,
    "sigmoid"              : HyperbolicTangent,
    "rational_quadratic"   : RationalQuadratic,
    "multiquadric"         : Multiquadric,
    "inverse_multiquadric" : InverseMultiquadric,
    "log"                  : Log,
}

get = mk_class_get(
    all_classes=KerasyKernelFunctions,
    kerasy_abst_class=[KerasyAbstKernel],
    genre="kernel"
)
