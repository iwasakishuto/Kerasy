# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    @param  size        : (int) total size of the data
    @param  batch_size  : (int) batch size.
    @return batch_slices: (list) contains (begin_idx, end_idx) for every batch.
    """
    num_batches = (size-1)//batch_size + 1
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(num_batches)]
