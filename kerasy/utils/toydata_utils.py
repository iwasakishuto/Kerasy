#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def generateX(size, xmin=-1, xmax=1, seed=None):
    x = np.linspace(xmin, xmax, size+2)[1:-1]
    x += np.random.normal(0, scale=(xmax-xmin)/(4*size), size=size)
    x = np.where(x<xmin, xmin, x)
    x = np.where(x>xmax, xmax, x)
    return x

def generateSin(size, xmin=-1, xmax=1, noise_scale=0.1, seed=None):
    x     = generateX(size=size, xmin=xmin, xmax=xmax, seed=seed)
    noise = np.random.RandomState(seed).normal(loc=0,scale=noise_scale,size=size)
    y     = np.sin(2*np.pi*x) + noise
    return (x,y)

def generateGausian(size, x=None, loc=0, scale=0.1, seed=None):
    x = x if x is not None else np.linspace(-1,1,size)
    y = np.random.RandomState(seed).normal(loc=loc,scale=scale,size=size)
    return (x,y)

def generateMultivariateNormal(cls, N, dim=2, low=0, high=1, scale=3e-2, seed=None, same=True):
    rnd = np.random.RandomState(seed)
    means = rnd.uniform(low,high,(cls,dim))
    covs = np.eye(dim)*scale
    if same:
        Ns = [N//cls for i in range(cls)]
    else:
        idx = rnd.randint(low=0, high=cls, size=N)
        Ns = [np.count_nonzero(idx==k) for k in range(cls)]
    data = np.concatenate([rnd.multivariate_normal(mean=means[k], cov=covs, size=Ns[k]) for k in range(cls)])
    clsses = np.concatenate([[k for _ in range(n)] for k,n in enumerate(Ns)])
    return (data, clsses)

def generateWholeCakes(cls, N, r_low=0, r_high=5, seed=None, same=True, plot=False, add_noise=True, noise_scale=5, figsize=(6,6), return_ax=False):
    """ Generate cls-class, 2-dimensional data. """
    rnd = np.random.RandomState(seed)
    if same:
        Ns = [N//cls for i in range(cls)]
    else:
        idx = rnd.randint(low=0, high=cls, size=N)
        Ns = [np.count_nonzero(idx==k) for k in range(cls)]
    Xs = []; clsses = []
    if plot: fig, ax = plt.subplots(figsize=figsize)
    for k,n in enumerate(Ns):
        theta = rnd.uniform(k*2/cls, (k+1)*2/cls, size=n)*np.pi
        if add_noise:
            theta_noise = rnd.uniform(-2/(cls*noise_scale), 2/(cls*noise_scale), size=n)*np.pi
            theta += theta_noise
        r = rnd.uniform(r_low,r_high, size=n)
        Xs.append(np.c_[
            r*np.cos(theta), # x-axis.
            r*np.sin(theta)  # y-axis
        ])
        clsses.append([k for _ in range(n)])
        if plot: ax.scatter(r*np.cos(theta), r*np.sin(theta), color=cm.jet(k/cls), label=f"class {k}: n={n}")
    if plot: ax.set_xlabel("$x$", fontsize=14), ax.set_ylabel("$y$", fontsize=14), ax.set_title("Generated data", fontsize=14)
    if return_ax:
        return ax, np.concatenate(Xs), np.concatenate(clsses)
    return np.concatenate(Xs), np.concatenate(clsses)

def generateWhirlpool(N, xmin=0, xmax=5, seed=None, plot=False, figsize=(6,6)):
    """ Generate 2-class 2-dimensional data. """
    a = np.linspace(xmin, xmax*np.pi, num=N//2)
    x = np.concatenate([
        np.stack([        a*np.cos(a),         a*np.sin(a)], axis=1),
        np.stack([(a+np.pi)*np.cos(a), (a+np.pi)*np.sin(a)], axis=1)
    ])
    x += np.random.RandomState(seed).uniform(size=x.shape)
    y  = np.repeat(np.arange(2), N//2)
    if plot:
        plt.figure(figsize=figsize)
        plt.scatter(x[:,0], x[:,1], c=y, cmap="jet")
        plt.xlabel("$x$", fontsize=14), plt.ylabel("$y$", fontsize=14), plt.title("Generated data", fontsize=14)
    return x, y
