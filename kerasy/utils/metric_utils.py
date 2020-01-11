"""Metric-related utilities."""

import numpy as np

def mean_squared_error(y_true, y_pred, sample_weight=None):
    return np.average((y_true - y_pred)**2, axis=0, weights=sample_weight).astype(float)

def root_mean_squared_error(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

def euclid_distances(x,y):
    """
    @param x,y shape=(N,D)
    @return shape=(N,)
    """
    return np.sqrt(np.sum(np.square(x-y), axis=1))

def pairwise_euclid_distances(X, squared=False):
    """Calculate euclid distance
    @param X         : (ndarray) shape=(N,D)
                        - N = Number of Data.
                        - D = Number of features.
    @param squared   : (bool) whether distances will be squared or not.
    @return distances: Dij means the distance between Xi and Xj.
    """
    dot_product = np.dot(X, X.T) # dot product between all X
    square_norm = np.diag(dot_product) # extract only diagonal elements so that get respective norms.

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    distances = np.expand_dims(square_norm, axis=0) - 2.0 * dot_product + np.expand_dims(square_norm, axis=1)
    distances = np.maximum(distances, 0.0) # for numerical errors.

    if not squared:
        mask = np.equal(distances, 0) # for avoid the gradient of sqrt is infinite,
        distances += mask*1e-16 # add a small epsilon where distances == 0.0
        distances = np.sqrt(distances)
        distances = distances * np.logical_not(mask) # Correct the epsilon added

    return distances
