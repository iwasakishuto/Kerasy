# coding: utf-8
import numpy as np
from scipy.sparse import issparse

from .generic_utils import handleKeyError
from .np_utils import CategoricalEncoder

def norm_vectors(*args, axis=-1, squared=True):
    if squared:
        return tuple([np.sum(np.square(arg), axis=axis) for arg in args])
    else:
        return tuple([np.sqrt(np.sum(np.square(arg), axis=axis)) for arg in args])

def normalize_vectors(*args, axis=-1):
    return tuple([arg/np.expand_dims(np.linalg.norm(arg, axis=axis), axis=axis) for arg in args])

def mean_squared_error(y_true, y_pred, sample_weight=None):
    return np.average((y_true - y_pred)**2, axis=0, weights=sample_weight).astype(float)

def root_mean_squared_error(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

def silhouette_samples(X, labels, metric='euclidean'):
    encoder = CategoricalEncoder()
    labels = encoder.to_categorical(labels)
    all_dists = pairwise_euclidean_distances(X)
    cluster_mean_dists = np.asarray([np.mean(all_dists[labels==label], axis=0) for label in np.unique(labels)]).T
    silhouette_scores = np.zeros(shape=(len(X)))
    for i,(dists,cls) in enumerate(zip(cluster_mean_dists,labels)):
        intra_dists = dists[cls]
        extra_dists = np.delete(dists, cls).min()
        silhouette_scores[i] = (extra_dists-intra_dists)/max(intra_dists,extra_dists)

    return silhouette_scores

def silhouette_score(X, labels, metric='euclidean', **kwargs):
    return np.mean(silhouette_samples(X, labels, metric=metric, **kwargs))

# Paired Distances
def check_paired_array(X,Y):
    """ Arrange the two array shapes.
    @param  X: shape=(Nx,Dx)
    @param  Y: shape=(Ny,Dy)
    @return X: shape=(N,D)
    @return Y: shape=(N,D)
    """
    if X.ndim==1: X=X.reshape(1,-1)
    if Y.ndim==1: Y=Y.reshape(1,-1)

    Nx,Dx = X.shape; Ny,Dy = Y.shape

    if Dx != Dy:
        raise ValueError(f"X.shape[1], and Y.shape[1] should be the same shape, but X.shape[1]={Dx} while Y.shape[1]={Dy}")
    if Nx!=Ny:
        if Nx==1:   X=np.tile(X, [Ny, 1])
        elif Ny==1: Y=np.repeat(Y, [Nx, 1])
        else:
            raise ValueError(f"X.shape={X.shape}, and Y.shape={Y.shape}, so we couldn't understand how to pair of 2 arrays.")
    return X,Y

def paired_distances(X,Y,metric="euclidean",**kwargs):
    """Compute the paired distances between X and Y.
    @params X,Y      : shape=(N,D) should be the same.
    @param metric    : string or callable.
    @return distances: shape=(N,)
    """
    if metric in PAIRED_DISTANCES:
        func = PAIRED_DISTANCES[metric]
        return func(X,Y)
    elif callable(metric):
        X,Y = check_paired_array(X,Y)
        distances = np.asarray([metric(x,y) for x,y in zip(X,Y)])
        return distances
    else:
        handleKeyError(list(PAIRED_DISTANCES.keys()), metric=metric)

def paired_euclidean_distances(X,Y,squared=False):
    X,Y = check_paired_array(X,Y)
    if squared:
        return np.sum(np.square(X-Y), axis=1)
    else:
        return np.sqrt(np.sum(np.square(X-Y), axis=1))

def paired_manhattan_distances(X,Y):
    X,Y = check_paired_array(X,Y)
    return np.sum(np.abs(X,Y), axis=1)

def paired_cosine_distances(X,Y):
    """ ||X-Y||^2 = ||X||^2 + ||Y||^2 - 2||X||･||Y||cos(theta) """
    X,Y = check_paired_arrays(X,Y)
    X,Y = normalize_vectors(X,Y)
    return .5 * paired_euclidean_distances(X-Y, squared=True)

PAIRED_DISTANCES = {
    'cosine': paired_cosine_distances,
    'euclidean': paired_euclidean_distances,
    'l2': paired_euclidean_distances,
    'l1': paired_manhattan_distances,
    'manhattan': paired_manhattan_distances,
    'cityblock': paired_manhattan_distances,
}

def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype

# Pairwise Distances
def check_pairwise_array(X,Y=None):
    """ Arrange for the pairwise distance.
    @param X: shape=(N,D)
    """
    if Y is None:
        Y = X
    X, Y, dtype_float = _return_float_dtype(X, Y)
    X = np.array(X, dtype=dtype_float)
    Y = np.array(Y, dtype=dtype_float)

    if X.ndim==1: X=X.reshape(1,-1)
    if Y.ndim==1: Y=Y.reshape(1,-1)
    Nx,Dx = X.shape; Ny,Dy = Y.shape
    if Dx != Dy:
        raise ValueError(f"X.shape={X.shape}, and Y.shape={Y.shape}, so we couldn't understand how to pairwise of 2 arrays.")
    return X,Y

def pairwise_distances(X,Y=None,metric="euclidean",**kwargs):
    """Compute the pair-wise distances between X and Y (X).
    @params X        : shape=(Nx,D)
    @params Y        : shape=(Ny,D)
    @param metric    : string or callable.
    @return distances: shape=(Nx,Ny)
    """
    if metric in PAIRWISE_DISTANCES:
        func = PAIRWISE_DISTANCES[metric]
        return func(X,Y,**kwargs)
    elif metric in PAIRED_DISTANCES:
        func = PAIRWISE_DISTANCES[metric]
        distances = np.asarray([func(x,Y) for x in X])
        return distances
    elif callable(metric):
        X,Y = check_pairwise_array(X,Y)
        distances = np.asarray([[metric(x,y) for y in Y] for x in X])
        return distances
    else:
        handleKeyError(list(PAIRWISE_DISTANCES.keys()), metric=metric)

def pairwise_euclidean_distances(X,Y=None,squared=False):
    """Calculate euclid distance
    @param X: (ndarray) shape=(Nx,D)
    @param Y: (ndarray) shape=(Ny,D)
    @param squared   : (bool) whether distances will be squared or not.
    @return distances: Dij means the distance between Xi and Xj.
    """
    X,Y = check_pairwise_array(X,Y)
    Xnorm,Ynorm = norm_vectors(X,Y,axis=-1,squared=True)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
    distances = np.expand_dims(Xnorm, axis=1) + np.expand_dims(Ynorm, axis=0) - 2*X.dot(Y.T)
    distances = np.maximum(distances, 0.0) # for numerical errors.
    if Y is X:
        np.fill_diagonal(a=distances, val=0.)
    if not squared:
        mask = np.equal(distances, 0.) # for avoid the gradient of sqrt is infinite,
        distances += mask*1e-16 # add a small epsilon where distances == 0.0
        distances = np.sqrt(distances, out=distances)
        distances = distances * np.logical_not(mask) # Correct the epsilon added

    return distances

PAIRWISE_DISTANCES = {
    'euclidean': paired_euclidean_distances,
    'l2': paired_euclidean_distances,
}
