# coding: utf-8

from __future__ import absolute_import

import numpy as np

def _validate_shuffle_split(n_samples, test_size, train_size,
                            default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    ref: https://github.com/scikit-learn/scikit-learn/blob/e5698bde9a8b719514bf39e6e5d58f90cfe5bc01/sklearn/model_selection/_split.py#L1741
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == 'i' and (test_size >= n_samples or test_size <= 0)
       or test_size_type == 'f' and (test_size <= 0 or test_size >= 1)):
        raise ValueError('test_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(test_size, n_samples))

    if (train_size_type == 'i' and (train_size >= n_samples or train_size <= 0)
       or train_size_type == 'f' and (train_size <= 0 or train_size >= 1)):
        raise ValueError('train_size={0} should be either positive and smaller'
                         ' than the number of samples {1} or a float in the '
                         '(0, 1) range'.format(train_size, n_samples))

    if train_size is not None and train_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ('i', 'f'):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if (train_size_type == 'f' and test_size_type == 'f' and
            train_size + test_size > 1):
        raise ValueError(
            'The sum of test_size and train_size = {}, should be in the (0, 1)'
            ' range. Reduce test_size and/or train_size.'
            .format(train_size + test_size))

    if test_size_type == 'f':
        n_test = np.ceil(test_size * n_samples)
    elif test_size_type == 'i':
        n_test = float(test_size)

    if train_size_type == 'f':
        n_train = np.floor(train_size * n_samples)
    elif train_size_type == 'i':
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            'With n_samples={}, test_size={} and train_size={}, the '
            'resulting train set will be empty. Adjust any of the '
            'aforementioned parameters.'.format(n_samples, test_size,
                                                train_size)
        )

    return n_train, n_test

def train_test_split(*arrays, **options):
    """ Split arrays or matrices into random train and test subsets
    @params arrays      : sequence of indexables with same length (=shape[0])
    @params test_size   : float, int, or None (default=None)
        If float, should be between 0.0 and 1.0 and represent the propotion.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size. If `train_size` is also None, it will be set to 0.3.
    @params train_size  : float, int, or None (default=None)
    @params random_state: int, represetnts the seed used by the random number generator.
    @params shuffle     : boolean (default=True), Wheter or not to shuffle the data before splitting.
    @params stratify    : array-like or None, data is split in a stratified fashion, using this as the class labels.
    """
    test_size    = options.pop('test_size', None)
    train_size   = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify     = options.pop('stratify', None)
    shuffle      = options.pop('shuffle', True)
    if options:
        raise TypeError(f"Invalid parameters passed: {str(options)}")

    num_samples = np.asarray(arrays[0]).shape[0]
    n_train, n_test = _validate_shuffle_split(num_samples, test_size, train_size, default_test_size=0.3)

    train_idxes = np.random.RandomState(random_state).choice(np.arange(num_samples), n_train, replace=False)
    test_idexs = np.ones(shape=num_samples, dtype=bool)
    test_idexs[train_idxes] = False

    return list((a[train_idxes],a[test_idexs]) for a in arrays)

def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    @param  size        : (int) total size of the data
    @param  batch_size  : (int) batch size.
    @return batch_slices: (list) contains (begin_idx, end_idx) for every batch.
    """
    num_batches = (size-1)//batch_size + 1
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(num_batches)]

def iter_from_variable_len_samples(X, lengths=None):
    """ yield starts and ends indexes of `X` per variable sample.
    Each sample should have the `n_features` features.
    np.sum(`lengths`) must be equal to `n_samples`
    =============================================================
    @params X       : Multiple connected samples. shape=(n_samples, n_features)
    @params lengths : int array. shape=(n_sequences)
    """
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}".format(n_samples, lengths))
        for i in range(len(lengths)):
            yield start[i], end[i]
