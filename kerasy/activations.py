# coding: utf-8
import numpy as np

class Linear():
    def forward(input):
        return input

    def diff(delta):
        return delta

class Softmax():
    def forward(input):
        exps = np.exp(input - np.max(input)) # For avoiding overflowing.
        return exps/np.sum(exps)

    def diff(delta):
        return 0

class Tanh():
    def forward(input):
        return np.tanh(input)

    def diff(delta):
        return 1-np.tanh(delta)**2

class Relu():
    def forward(input):
        return np.where(input>0, input, 0)
    
    def diff(delta):
        return np.where(input>0, 1, 0)
    
ActivationHandler = {
    'linear' : Linear,
    'softmax': Softmax,
    'tanh'   : Tanh,
    'relu'   : Relu,
}

def ActivationFunc(activation_func_name):
    return ActivationHandler[activation_func_name]
