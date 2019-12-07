"""Numpy-related utilities."""
import numpy as np

class CategoricalEncoder():
    def __init__(self):
        self.obj2cls = {}
        self.cls2obj = {}
        self.dtype = None

    def _mk_obj2cls_dict(self, obj):
        if len(self.obj2cls) == 0:
            arr = np.asarray(obj)
            unique = np.unique(arr)
            self.obj2cls = dict(zip(unique, range(len(unique))))
            self.cls2obj = dict(zip(range(len(unique)), unique))
            self.dtype = arr.dtype

    def to_categorical(self, obj):
        self._mk_obj2cls_dict(obj)
        return np.asarray([self.obj2cls[e] for e in obj], dtype=int)

    def to_onehot(self, obj, num_classes=None, dtype='float32'):
        n = len(obj)
        self._mk_obj2cls_dict(obj)
        num_classes = num_classes if num_classes is not None else len(self.obj2cls)
        categorical = np.zeros(shape=(n, num_classes), dtype=int)
        categorical[np.arange(n), np.asarray([self.obj2cls[e] for e in obj], dtype=int)] = 1
        return categorical

    def reverse(self, cls):
        return np.asarray([self.cls2obj[c] for c in cls], dtype=self.dtype)

def findLowerUpper(data, margin=1e-2):
    """ data.shape=(N,D) """
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    margins = (maxs-mins)*margin
    return (mins-margin, maxs+margin)
