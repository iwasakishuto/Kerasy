# coding: utf-8
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

from .generic_utils import handleKeyError

class CategoricalEncoder():
    def __init__(self):
        self.obj2cls = {}
        self.cls2obj = {}
        self.dtype = None

    def _mk_obj2cls_dict(self, obj, origin=0):
        if len(self.obj2cls) == 0:
            arr = np.asarray(list(obj))
            unique = np.unique(arr)
            self.obj2cls = dict(zip(unique, range(origin, origin+len(unique))))
            self.cls2obj = dict(zip(range(origin, origin+len(unique)), unique))
            self.dtype = arr.dtype
        else:
            print("Dictionaly for Encoder is already made.")

    def to_categorical(self, obj, origin=0):
        self._mk_obj2cls_dict(obj, origin=origin)
        return np.asarray([self.obj2cls[e] for e in obj], dtype=int)

    def to_onehot(self, obj, num_classes=None, dtype='float32', origin=0):
        n = len(obj)
        self._mk_obj2cls_dict(obj, origin=origin)
        num_classes = num_classes if num_classes is not None else len(self.obj2cls)
        categorical = np.zeros(shape=(n, num_classes), dtype=int)
        categorical[np.arange(n), np.asarray([self.obj2cls[e] for e in obj], dtype=int)] = 1
        return categorical

    def reverse(self, cls):
        return np.asarray([self.cls2obj[c] for c in cls], dtype=self.dtype)

def findLowerUpper(data, margin=1e-2, N=None):
    """ data.shape=(N,D) """
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    margins = (maxs-mins)*margin
    lbs = mins-margins; ubs = maxs+margins
    if N is None:
        return (lbs, ubs)
    else:
        return np.concatenate([[np.linspace(lb,ub,N) for lb,ub in zip(lbs, ubs)]])

def inverse_arr(arr):
    """
    arr[k]=v → arr_inv[k]=v
    """
    arr_inv = np.zeros_like(arr)
    for i in range(len(arr)):
        arr_inv[arr[i]] = i
    return arr_inv

def _check_sample_weight(sample_weight, X, dtype=np.float64):
    """ Validate sample weights.
    @params sample_weight : ndarray shape=(n_samples,) or None
    @params X             : Input data. shape=(n_samples, *n_features)
    @params dtype         : sample weight's data type.
    @return sample_weight : shape=(n_samples,)
    """
    n_samples = X.shape[0]

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=dtype)
    else:
        sample_weight = np.asarray(sample_weight, dtype=dtype)
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        if sample_weight.shape != (n_samples,):
            raise ValueError(f"sample_weight.shape == {sample_weight.shape}, expected {(n_samples,)}!")
    return sample_weight

def normalize(arr, axis=None):
    """ Inplace Method
    Normalizes the input array so that it sums to 1 along with `axis`.
    @params arr : Non-normalized Input array.
    @params axis: Dimension along which normalization is performed.
    """
    if axis is None:
        arr /= arr.sum()
    else:
        arr /= np.expand_dims(arr.sum(axis), axis)

def standardize(arr, axis=None):
    """ Inplace Method
    Standarzes the input array so that it sums to 1 along with `axis`.
    @params arr : Non-standarzed Input array.
    @params axis: Dimension along which normalization is performed.
    """
    max = arr.max(axis)
    min = arr.min(axis)
    if axis is None:
        arr -= min
        arr /= (max-min)
    else:
        arr -= np.expand_dims(min, axis=axis)
        arr /= np.expand_dims((max-min), axis=axis)

def log_normalize(arr, axis=None):
    """ Inplace Method
    Normalizes the input array so that ``sum(exp(a)) == 1``.
    @params arr : Non-normalized Input array.
    @params axis: Dimension along which normalization is performed.
    """
    with np.errstate(under="ignore"):
        arr_lse = logsumexp(arr, axis, keepdims=True)
    arr -= arr_lse

def log_mask_zero(arr):
    """Computes the log of input probabilities masking divide by zero in log."""
    arr = np.asarray(arr)
    with np.errstate(divide="ignore"):
        return np.log(arr)

# ref: https://github.com/hmmlearn/hmmlearn/blob/0e9274cb138427919c13ef79f11a7358c4e2b4a9/lib/hmmlearn/stats.py#L5
def log_multivariate_normal_density(X, means, covars, covariance_type='diag', *args, **kwargs):
    """ Compute the log probability under a multivariate Gaussian distribution.

    (PRML 2.43)
    N_i(x_n|mu_i,Sigma_i) = 1/(2*pi)^{D/2} * 1/|Sigma_i|^{1/2} * exp(-1/2*(x_n-mu_i)^T*Sigma_i^{-1}*(x_n-mu_i))

    | covariance_type | covars.shape                         |   **kwargs      |
    ----------------------------------------------------------------------------
    | 'spherical'     | (n_gaussian, n_features)             |                 |
    | 'tied'          | (n_features, n_features)             |                 |
    | 'diag'          | (n_gaussian, n_features)             |                 |
    | 'full'          | (n_gaussian, n_features, n_features) | min_covar=1.e-7 |

    @params X               : Input data. shape=(n_samples, n_features)
    @params menas           : mean vectors for n_gaussian Gaussian. shape=(n_gaussian, n_features)
    @params covars          : covariance for for n_gaussian Gaussians. The shape depends on `covariance_type`.
    @params covariance_type : Type of the covariance.
    @params *args           : No means.
    @params **kwargs        : No means.
    @return log_probs       : Log probabilities of each data point in X. shape=(n_samples, n_gaussian)
    """
    return {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full,
    }[covariance_type](X, means, covars, *args, **kwargs)

def _log_multivariate_normal_density_spherical(X, means, covars, *args, **kwargs):
    """
    @params X        : shape=(n_samples,  n_features)
    @params menas    : shape=(n_gaussian, n_features)
    @params covars   : shape=(n_gaussian, n_features)
    @params *args    : No means.
    @params **kwargs : No means.
    @return og_probs : shape=(n_samples,  n_gaussian)
    """
    # If the input
    _,  n_features = X.shape
    cv = covars.copy()
    if covars.ndim == 1:
        # This indicates that covars.shape=(n_gaussian)
        cv = cv[:, np.newaxis]
    if cv.shape[1] == 1:
        # Assign single variance value to all diagonal elements.
        # shape=(n_gaussian, 1) → shape=(n_gaussian*1, 1*n_features)
        cv = np.tile(cv, (1, n_features))
    return _log_multivariate_normal_density_diag(X, means, cv, *args, **kwargs)

def _log_multivariate_normal_density_tied(X, means, covars, *args, **kwargs):
    """
    @params X         : shape=(n_samples,  n_features)
    @params menas     : shape=(n_gaussian, n_features)
    @params covars    : shape=(n_features, n_features)
    @params *args     : No means.
    @params **kwargs  : No means.
    @return log_probs : shape=(n_samples,  n_gaussian)
    """
    n_gaussian, _ = means.shape
    # shape=(n_features, n_features) → shape=(n_gaussian, n_features*1, n_features*1)
    cv = np.tile(covars, (n_gaussian, 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv, *args, **kwargs)

def _log_multivariate_normal_density_diag(X, means, covars, *args, **kwargs):
    """
    @params X         : shape=(n_samples,  n_features)
    @params menas     : shape=(n_gaussian, n_features)
    @params covars    : shape=(n_gaussian, n_features)
    @return log_probs : shape=(n_samples,  n_gaussian)
    """
    n_samples, n_features = X.shape
    # Avoid log(0) by setting `np.finfo(float).tiny` to the smallest value.
    covars = np.maximum(covars, np.finfo(float).tiny)
    with np.errstate(over="ignore"):
        return -n_features/2*np.log(2*np.pi) \
               -1/2*np.log(covars).sum(axis=-1) + \
               -1/2*( (X[:, np.newaxis, :]-means[np.newaxis, :, :])**2 / covars[np.newaxis, :, :] ).sum(axis=-1)

def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """
    @params X         : shape=(n_samples,  n_features)
    @params menas     : shape=(n_gaussian, n_features)
    @params covars    : shape=(n_gaussian, n_features, n_features)
    @return log_probs : shape=(n_samples,  n_gaussian)
    """
    n_samples, n_features = X.shape
    n_gaussian, _ = means.shape
    log_prob = np.empty(shape=(n_samples, n_gaussian))
    for n, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar*np.eye(n_features), lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, positive-definite")
        # cv_chol is lower-triangular Cholesky factor of cv.
        cv_log_det = 2*np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X-mu).T, lower=True).T
        log_prob[:,n] = -1/2*(np.sum(cv_sol**2, axis=1) + n_features*np.log(2*np.pi) + cv_log_det)

    return log_prob

# ref: https://github.com/hmmlearn/hmmlearn/blob/0e9274cb138427919c13ef79f11a7358c4e2b4a9/lib/hmmlearn/utils.py#L89
def decompress_based_on_covariance_type(covars, covariance_type='full', n_gaussian=1, n_features=1):
    """
    Create the correct shape of covariance matris
    from each feature based on the covariance type.
    ~~~~~~~~
    @params covars          : compressed covariance values.
    @params covariance_type : string.
    @params n_gauusian      : number of gaussian
    @params n_features      : number of features.
    @return covariances     : covariances. shape=(n_gaussian, n_features, n_features)
    ==========================================================
    | covariance_type |               Gaussian               |
    ----------------------------------------------------------
    | "spherical"     | (n_gaussian)                         |
    | "diag"          | (n_gaussian, n_features)             |
    | "full"          | (n_gaussian, n_features, n_features) |
    | "tied"          | (n_features, n_features)             |
    """
    if covariance_type == 'full':
        return covars
    elif covariance_type == 'diag':
        # np.diag(n_features).shape=(n_features, n_features)
        # shape=(n_gaussian, n_features) → shape=(n_gaussian, (n_features, n_features))
        return np.array(list(map(np.diag, covars)))
    elif covariance_type == 'tied':
        # shape=(n_features, n_features) → shape=(n_gaussian, n_features*1, n_features*1)
        return np.tile(covars, (n_gaussian, 1, 1))
    elif covariance_type == 'spherical':
        # shape=(1, n_features, n_features)
        eye = np.eye(n_features)[np.newaxis, :, :]
        # shape=(n_gaussian, 1, 1)
        covars = covars[:, np.newaxis, np.newaxis]
        # shape=(n_gaussian, n_features, n_features)
        return eye * covars

# ref: https://github.com/hmmlearn/hmmlearn/blob/0e9274cb138427919c13ef79f11a7358c4e2b4a9/lib/hmmlearn/_utils.py#L47
def compress_based_on_covariance_type_from_tied_shape(tied_cv, covariance_type, n_gaussian):
    """ Create all the covariance matrices from a given template.
    @params tied_cv         : Input tied covariance.
    @params covariance_type : string.
    @params n_gauusian      : number of gaussian
    @return cv              : compressed values.
    ==========================================================
    | covariance_type |               Gaussian               |
    ----------------------------------------------------------
    | "spherical"     | (n_gaussian)                         |
    | "diag"          | (n_gaussian, n_features)             |
    | "full"          | (n_gaussian, n_features, n_features) |
    | "tied"          | (n_features, n_features)             |
    """
    n_features = tied_cv.shape[1]
    if covariance_type == 'spherical':
        # shape=(n_features,) → shape=(n_gaussian, n_features*1)
        # Before:
        # cv = np.tile(tied_cv.mean()*np.ones(n_features), (n_gaussian, 1))
        # Modified by Shuto Iwasaki.
        cv = np.full(shape=n_gaussian, fill_value=tied_cv.mean())
    elif covariance_type == 'tied':
        # shape=((n_features, n_features)
        cv = tied_cv
    elif covariance_type == 'diag':
        # shape=(n_features,) → shape=(n_gaussian, n_features*1)
        cv = np.tile(np.diag(tied_cv), (n_gaussian, 1))
    elif covariance_type == 'full':
        # shape=(n_features, n_features) → shape=(n_gaussian, n_features*1, n_features*1)
        cv = np.tile(tied_cv, (n_gaussian, 1, 1))
    else:
        handleKeyError(['spherical', 'tied', 'diag', 'full'], covariance_type=covariance_type)
    return cv
