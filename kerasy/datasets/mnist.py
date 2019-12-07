"""MNIST handwritten digits dataset.
"""
from __future__ import absolute_import

from ..utils import get_file
import numpy as np

def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.
    @params  path: (str)   path where to cache the dataset locally (relative to ~/.kerasy/datasets).
    @returns       (tuple) Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
