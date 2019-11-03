#coding: utf-8
import numpy as np

def generateX(size, xmin=-1, xmax=1, seed=None):
    x = np.linspace(xmin, xmax, size+2)[1:-1]
    x += np.random.normal(0, scale=(xmax-xmin)/(4*size), size=size)
    x = np.where(x<xmin, xmin, x)
    x = np.where(x>xmax, xmax, x)
    return x

def generateSin(size, xmin=-1, xmax=1, seed=None):
    x = generateX(size=size, xmin=xmin, xmax=xmax, seed=seed)
    y = np.sin(2*np.pi*x)
    return (x,y)

def generateGausian(size, x=None, loc=0, scale=0.1, seed=None):
    x = x if x is not None else np.linspace(-1,1,size)
    y = np.random.RandomState(seed).normal(loc=loc,scale=scale,size=size)
    return (x,y)

def generateMultivariateNormal(cls, N, dim=2, low=0, high=1, scale=3e-2, seed=None, same=True):
    means = np.random.RandomState(seed).uniform(low,high,(cls,dim))
    covs = np.eye(dim)*scale
    if same:
        Ns = [N//cls for i in range(cls)]
    else:
        idx = np.random.RandomState(seed).randint(low=0, high=cls, size=N)
        Ns = [np.count_nonzero(idx==k) for k in range(cls)]

    data = np.concatenate([np.random.RandomState(seed+k).multivariate_normal(mean=means[k], cov=covs, size=Ns[k]) for k in range(cls)])
    return (data, means)
