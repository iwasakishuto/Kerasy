# C-Extensions for Python

> **Cython** is an **optimising static compiler** for both the **[Python](https://www.python.org/about/)** programming language and the extended Cython programming language (based on **Pyrex**). It makes writing C extensions for Python as easy as Python itself. **([Documentation](https://cython.org/#documentation))**

## How to Compile

```sh
$ pip install setuptools>=30.3.0
$ python setup.py
```

## How to use Cython on Jupyter Notebook

```py
%load_ext Cython
```

```py
%%cython -n test_cython_code

import numpy as np
cimport numpy as np
from cython cimport floating
from libc.math cimport sqrt

cdef floating euclidean_distance(floating* a, floating* b, int n_features) nogil:
    cdef floating result, diff
    result = 0
    cdef int i
    for i in range(n_features):
        diff = (a[i] - b[i])
        result += diff * diff
    return sqrt(result)

def paired_euclidean_distances(
        np.ndarray[floating, ndim=2, mode='c'] X,
        np.ndarray[floating, ndim=2, mode='c'] Y):
    dtype = np.float32 if floating is float else np.float64
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef floating* X_p = <floating*>X.data
    cdef floating* Y_p = <floating*>Y.data
    cdef floating* x_p, y_p
    cdef Py_ssize_t idx
    distances = np.empty(n_samples, dtype=dtype)

    for idx in range(n_samples):
        distances[idx] = euclidean_distance(X_p+idx*n_features, Y_p+idx*n_features, n_features)
    return distances
```

```python
import numpy as np

X = np.arange(1,11).reshape(-1,1).astype(float)
Y = np.arange(1,11)[::-1].reshape(-1,1).astype(float)
paired_euclidean_distances(X,Y)
>>> array([9., 7., 5., 3., 1., 1., 3., 5., 7., 9.])
```
