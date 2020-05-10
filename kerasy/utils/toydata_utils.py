# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .generic_utils import handleKeyError
from .generic_utils import handleRandomState

DNA_BASE_TYPES=["A","T","G","C"]
RNA_BASE_TYPES=["A","U","G","C"]

def generateX(size, xmin=-1, xmax=1, seed=None):
    rnd = handleRandomState(seed)
    x = np.linspace(xmin, xmax, size+2)[1:-1]
    x += rnd.normal(0, scale=(xmax-xmin)/(4*size), size=size)
    return np.clip(a=x, a_min=xmin, a_max=xmax)

def generateSin(size, xmin=-1, xmax=1, noise_scale=0.1, seed=None):
    rnd = handleRandomState(seed)
    x     = generateX(size=size, xmin=xmin, xmax=xmax, seed=rnd)
    noise = rnd.normal(loc=0,scale=noise_scale,size=size)
    y     = np.sin(2*np.pi*x) + noise
    return (x,y)

def generateGausian(size, x=None, loc=0, scale=0.1, seed=None):
    rnd = handleRandomState(seed)
    x = x if x is not None else np.linspace(-1,1,size)
    y = rnd.normal(loc=loc,scale=scale,size=size)
    return (x,y)

def generateMultivariateNormal(cls, N, dim=2, low=0, high=1, scale=3e-2, seed=None, same=True):
    rnd = handleRandomState(seed)
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
    rnd = handleRandomState(seed)
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
    rnd = handleRandomState(seed)
    a = np.linspace(xmin, xmax*np.pi, num=N//2)
    x = np.concatenate([
        np.stack([        a*np.cos(a),         a*np.sin(a)], axis=1),
        np.stack([(a+np.pi)*np.cos(a), (a+np.pi)*np.sin(a)], axis=1)
    ])
    x += rnd.uniform(size=x.shape)
    y  = np.repeat(np.arange(2), N//2)
    if plot:
        plt.figure(figsize=figsize)
        plt.scatter(x[:,0], x[:,1], c=y, cmap="jet")
        plt.xlabel("$x$", fontsize=14), plt.ylabel("$y$", fontsize=14), plt.title("Generated data", fontsize=14)
    return x, y

def generateSeq(size, nucleic_acid="DNA", weights=None, seed=None):
    """Generate Sample Sequences.
    @params size        : sequence size.
    @params nucleic_acid: DNA or RNA
    @params weights     : weights for each base. it must sum to 1.
    @params seed        : for the porpose of Reproducibility.
    """
    rnd = handleRandomState(seed)
    if nucleic_acid == "DNA":
        bases = DNA_BASE_TYPES
    elif nucleic_acid == "RNA":
        bases = RNA_BASE_TYPES
    else:
        handleKeyError(["DNA","RNA"], nucleic_acid=nucleic_acid)
    sequences = rnd.choice(bases, p=weights, size=size)
    return sequences

def generateSeq_embedded_Motif(Np, Nn, length, motif, nuc=DNA_BASE_TYPES, seed=None):
    """ Generate dataset for DeepBind.
    @params Np     : (Positive) Number of sequence which has motif.
    @params Nn     : (Negative) Number of sequence which doesn't have motif.
    @params length : (int) The length of sequence. (Fixed length)
                     (tuple) The range of length of sequence. (Varying length)
    @params motif  : shape=(len_motif, num_base_types)
    @params nuc    : Types of Base.
    """
    rnd = handleRandomState(seed)
    num_total = Np+Nn

    if isinstance(length, int):
        len_fix = True
    else:
        len_min = min(length)
        len_max = max(length)
        len_fix = False

    len_motif, num_base_types = motif.shape
    if num_base_types != len(nuc):
        raise ValueError(f"The number of bases in the `motif` is different from \
                            the number of bases in the `nuc`.\
                            ({num_base_types} != {len(nuc)})")
    if (len_fix and len_motif > length) or (not len_fix and len_motif > len_min):
        raise ValueError(f"The length of motif should be smaller than \
                            the length of each sequence. \
                            (got {len_motif} > {length if len_fix else len_min})")

    y = [1] * Np + [0] * Nn
    x = [None] * num_total

    for i in range(num_total):
        if len_fix:
            len_seq = length
        else:
            len_seq = int(rnd.uniform(low=len_min, high=len_max, size=1))
        seq = rnd.choice(nuc, size=len_seq) # generate Random Sequence.
        if i < Np: # If positive, embed motif at random position.
            j = rnd.randint(len_seq - len_motif + 1)
            for k in range(len_motif):
                # embed motif based on respective probabilities.
                seq[j + k] = rnd.choice(nuc, p=motif[k])
        x[i] = "".join(seq)

# Reference: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/test_utils.py#L27
def generate_test_data(num_train=1000, num_test=500,
                       input_shape=(10,), output_shape=(2,),
                       classification=True, num_classes=2,
                       random_state=None):
    rnd = handleRandomState(random_state)
    samples = num_train + num_test

    if classification:
        y = rnd.randint(0, num_classes, size=(samples,))
        X = np.zeros((samples,) + input_shape, dtype=np.float64)
        for i in range(samples):
            X[i] = rnd.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = rnd.random((samples,))
        X = np.zeros((samples,) + input_shape, dtype=np.float64)
        y = np.zeros((samples,) + output_shape, dtype=np.float64)
        for i in range(samples):
            X[i] = rnd.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = rnd.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (X[:num_train], y[:num_train]), (X[num_train:], y[num_train:])
    return (x, y)
