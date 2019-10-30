#coding: utf-8
import numpy as np

def generateSin(size, xmin=-1, xmax=1, seed=None):
    x = np.random.RandomState(seed).uniform(low=xmin,high=xmax,size=size)
    y = np.sin(2*np.pi*x)
    return (x,y)

def generateGausian(size, x=None, loc=0, scale=0.1, seed=None):
    x = x if x is not None else np.linspace(-1,1,size)
    y = np.random.RandomState(seed).normal(loc=loc,scale=scale,size=size)
    return (x,y)
