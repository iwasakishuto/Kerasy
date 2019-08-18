# coding: utf-8
import numpy as np

def Zeros(shape, dtype=None):
    return np.zeros(shape=shape, dtype=dtype)

def Ones(shape, dtype=None):
    return np.ones(shape=shape, dtype=dtype)

def RandomNormal(shape):
    return np.random.normal(loc=0, scale=0.05, size=shape)

def RandomUniform(shape):
    return np.random.uniform(low=-0.05,high=0.05,size=shape)

InitializeHandler = {
    'zeros': Zeros,
    'ones': Ones,
    'random_normal': RandomNormal,
    'random_normal': RandomUniform,
}

def Initializer(initializer_name):
    return InitializeHandler[initializer_name]
