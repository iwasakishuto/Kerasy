# coding: utf-8
import numpy as np

def cross_validation(k, x, y, metrics, modelcls, shuffle=True, seed=None, modelkwargs={}, trainkwargs={}):
    """
    @params k        : k-fold
    @params x        : Explanatory variables.
    @params y        : Objective variables.
    @params metrics  : (func) How to measure prediction error.
    @params modelcls : Model Class. We will create k instances.
    @params modelargs: Model instance's argments.
    """
    n_samples = len(x)
    idx = np.repeat(np.arange(k), np.round(n_samples/k))
    if shuffle: np.random.RandomState(seed).shuffle(idx)

    score = 0
    for cls in range(k):
        x_train = x[idx==cls]; x_test = x[idx!=cls]
        y_train = y[idx==cls]; y_test = y[idx!=cls]
        model = modelcls(**modelkwargs)
        model.fit(x_train, y_train, **trainkwargs)
        y_pred = model.predict(x_test)
        score += metrics(y_test, y_pred)
    return score/k
