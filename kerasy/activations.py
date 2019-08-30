# coding: utf-8
import numpy as np

class Linear():
    def forward(input):
        return input

    def diff(delta):
        return delta

class Softmax():        
    def forward(input):
        """ @param input shape=(class,) """
        exps = np.exp(input - np.max(input)) # For avoiding overflowing.
        return exps/np.sum(exps)
    
    def diff(delta):
        """
        @param  : shape=(class,)
        @return : shape=(class,class)
        =================================
        ∂z_i    / z_i(1-z_i)  --- if i==k
        ---- = |  
        ∂a_k    \ -z_iz_k     --- if i!=k
        """
        exps = np.exp(delta - np.max(delta)) # For avoiding overflowing.
        y = exps/np.sum(exps)
        return np.diag(y) - np.outer(y,y)

class Tanh():
    def forward(input):
        return np.tanh(input)

    def diff(delta):
        return 1-np.tanh(delta)**2

class Relu():
    def forward(input):
        return np.where(input>0, input, 0)

    def diff(delta):
        return np.where(delta>0, 1, 0)

class Sigmoid():
    def __init__(self):
        self.out = None

    def forward(self, input):
        out = 1/(1+np.exp(input))
        self.out = out # Memorize!!
        return out

    def diff(self, delta):
        return delta * (1-self.out) * self.out

ActivationHandler = {
    'linear' : Linear,
    'softmax': Softmax,
    'tanh'   : Tanh,
    'relu'   : Relu,
    'sigmoid': Sigmoid(),
}

def ActivationFunc(activation_func_name):
    return ActivationHandler[activation_func_name]
