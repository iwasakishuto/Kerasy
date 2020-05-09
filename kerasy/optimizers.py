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

        self.initial_decay = kwargs.pop('decay', 0.0)
        self.decay = self.initial_decay
        self.epsilon = kwargs.pop('eosukib', 1e-7)

        allowed_kwargs = {"clipnorm", "clipvalue"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError(f'Unexpecte keyword argument passed to optimizer: {str(k)}')

        self.updates = []
        self.weights = []
        self.iterations = 0

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
    def __init__(self, learning_rate=0.01, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        new_param = curt_param - lr*grad

        return new_param

class SGD(KerasyAbstOptimizer):
    """ (Momentum SGD) Stochastic gradient descent optimizer.
    ~~~
    @param learning_rate: (float) Learning rate.
    @param momentum     : (float) Accelerates SGD in the relevant direction and dampens oscillations.
    @param nesterov     : (bool)  Whether to apply Nesterov momentum.
    """
    def __init__(self, learning_rate=0.01, momentum=0., nesterov=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        v = self.velocities.get(name, np.zeros_like(curt_param))
        v = self.momentum*v - lr*grad
        self.velocities[name] = v

        if self.nesterov:
            new_param = curt_param + self.momentum*v - lr*grad
        else:
            new_param = curt_param + v

        return new_param

class RMSprop(KerasyAbstOptimizer):
    """ RMSprop optimizer.
    * Maintain a moving (discounted) average of the square of gradients.
    * Divide the gradient by the root of this average.
    ~~~
    @param learning_rate: (float) Learning rate.
    @param rho          : (float) Discounting factor for the history/coming gradient.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.accumulators = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        # update accumulator
        a = self.accumulators.get(name, np.zeros_like(curt_param))
        new_a = self.rho*a + (1.-self.rho) * np.square(grad)
        self.accumulators[name] = new_a

        new_param = curt_param - lr*grad/(np.sqrt(new_a) + self.epsilon)
        return new_param

class Adagrad(KerasyAbstOptimizer):
    """ Adagrad optimizer.
    Adagrad is an optimizer with parameter-specific learning rates, which are
    adapted relative to how frequently a parameter gets updated during training.
    The more updates a parameter receives, the smaller the updates.
    ~~~
    @param learning_rate: (float) Learning rate.
    """
    def __init__(self, learning_rate=0.01, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.accumulators = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        # update accumulator
        a = self.accumulators.get(name, np.zeros_like(curt_param))
        new_a = a + np.square(grad)
        self.accumulators[name] = new_a

        new_param = curt_param - lr*grad/(np.sqrt(new_a) + self.epsilon)
        return new_param

class Adadelta(KerasyAbstOptimizer):
    """ Adadelta optimizer.
    Adadelta optimization is a stochastic gradient descent method that is based
    on adaptive learning rate per dimension to address two drawbacks:
    * The continual decay of learning rates throughout training
    * The need for a manually selected global learning rate
    ~~~
    @param learning_rate: (float) Learning rate.
    @param rho          : (float) The decay rate.
    """
    def __init__(self, learning_rate=1.0, rho=0.95, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.rho = rho
        self.accumulators = defaultdict()
        self.delta_accumulators = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        # update accumulator
        a = self.accumulators.get(name, np.zeros_like(curt_param))
        d_a = self.delta_accumulators.get(name, np.zeros_like(curt_param))
        new_a = self.rho * a + (1. - self.rho) * np.square(grad)
        self.accumulators[name] = new_a

        # use the new accumulator and the *old* delta_accumulator
        update = grad * np.sqrt(d_a + self.epsilon) / np.sqrt(new_a + self.epsilon)
        new_param = curt_param - lr * update

        # update delta_accumulator
        new_d_a = self.rho * d_a + (1 - self.rho) * np.square(update)
        self.delta_accumulators[name] = new_d_a

        return new_param

class Adam(KerasyAbstOptimizer):
    """ Adam optimizer. (RMSprop with momentum)
    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.
    ~~~
    @param learning_rate: (float) Learning rate.
    @param beta_1       : (float) The exponential decay rate for the 1st moment estimates.
    @param beta_2       : (float) The exponential decay rate for the 2nd moment estimates.
    @param amsgrad      : (bool)  Whether to apply AMSGrad variant of this algorithm
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.moments_1 = defaultdict()
        self.moments_2 = defaultdict()
        if amsgrad:
            self.vhats = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        t = self.iterations + 1.
        lr_t = lr * ( np.sqrt(1.-pow(self.beta_2, t)) / (1.-pow(self.beta_1, t)))

        m_1 = self.moments_1.get(name, np.zeros_like(curt_param))
        m_2 = self.moments_2.get(name, np.zeros_like(curt_param))
        # update the moments.
        m_1 = self.beta_1*m_1 + (1.-self.beta_1)*grad
        m_2 = self.beta_2*m_2 + (1.-self.beta_2)*np.square(grad)
        # Memorize.
        self.moments_1[name] = m_1
        self.moments_2[name] = m_2

        if self.amsgrad:
            vhat = self.vhats.get(name, np.zeros_like(curt_param))
            vhat_t = np.maximum(vhat, m_2)
            new_param = curt_param - lr_t * m_1 / (np.sqrt(vhat_t) + self.epsilon)
            self.vhats[name] = vhat_t
        else:
            new_param = curt_param - lr_t * m_1 / (np.sqrt(m_2) + self.epsilon)

        return new_param

class Adamax(KerasyAbstOptimizer):
    """ Adamax optimizer.
    Adamax optimize is a variant of Adam based on the infinity norm. Adamax is
    sometimes superior to adam, specially in models with embeddings.
    ~~~
    @param learning_rate: (float) Learning rate.
    @param beta_1       : (float) The exponential decay rate for the 1st moment estimates.
    @param beta_2       : (float) The exponential decay rate for the exponentially weighted infinity norm.
    """
    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.moments_1 = defaultdict()
        self.inf_norms = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        # Increase processing speed by storing in local variables.
        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay*self.iterations))

        t = self.iterations + 1.
        lr_t = lr / (1. - pow(self.beta_1, t))

        m_1 = self.moments_1.get(name, np.zeros_like(curt_param))
        i_n = self.inf_norms.get(name, np.zeros_like(curt_param))
        # update the moments.
        m_1 = self.beta_1*m_1 + (1.-self.beta_1)*grad
        i_n = np.maximum(self.beta_2*i_n, np.abs(grad))
        # Memorize.
        self.moments_1[name] = m_1
        self.inf_norms[name] = i_n

        new_param = curt_param - lr_t * m_1 / (i_n + self.epsilon)

        return new_param

class Nadam(KerasyAbstOptimizer):
    """ Nesterov Adam optimizer. (Adam with Nesterov momentum.)
    ~~~
    @param learning_rate: (float) Learning rate.
    @param beta_1       : (float) The exponential decay rate for the 1st moment estimates.
    @param beta_2       : (float) The exponential decay rate for the exponentially weighted infinity norm.
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_schedule = 1.
        self.moments_1 = defaultdict()
        self.inf_norms = defaultdict()

    def get_updates(self, grad, curt_param, name):
        grad = self.get_gradient(grad) # Cliped.

        t = self.iterations + 1.
        momentum_cache_t   = self.beta_1 * (1. - 0.5*(pow(0.96,     t*self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5*(pow(0.96, (t+1)*self.schedule_decay)))
        m_schedule_new  = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1

        m_1 = self.moments_1.get(name, np.zeros_like(curt_param))
        i_n = self.inf_norms.get(name, np.zeros_like(curt_param))

        g_prime = grad / (1. - m_schedule_new)
        m_1 = self.beta_1 * m_1 + (1.-self.beta_1) * grad
        m_1_prime = m_1 / (1. - m_schedule_next)
        m_1_bar = (1.-momentum_cache_t) * g_prime + (momentum_cache_t_1 * m_1_prime)
        i_n = self.beta_2 * i_n + (1. - self.beta_2) * np.square(grad)
        i_n_prime = i_n / (1. - pow(self.beta_2, t))

        # Memorize.
        self.moments_1[name] = m_1
        self.inf_norms[name] = i_n

        new_param = curt_param - self.learning_rate * m_1_bar / (np.sqrt(i_n_prime) + self.epsilon)

        return new_param

KerasyOptimizerClasses = {
    'gra'      : GradientDescent,
    'sgd'      : SGD,
    'rmsprop'  : RMSprop,
    'adagrad'  : Adagrad,
    'adadelta' : Adadelta,
    'adam'     : Adam,
    'adamax'   : Adamax,
    'nadam'    : Nadam,
}

get = mk_class_get(
    all_classes=KerasyOptimizerClasses,
    kerasy_abst_class=[KerasyAbstOptimizer],
    genre="optimizer"
)
