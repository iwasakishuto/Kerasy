# coding: utf-8
import numpy as np
from kerasy.ML.decomposition import PCA, UMAP, tSNE
from kerasy.datasets import mnist
from kerasy.utils import cluster_accuracy

num_mnist = 300
n_components = 5
epochs = 10
seed = 123

def get_test_data():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:num_mnist].reshape(num_mnist, -1)
    y_train = y_train[:num_mnist]
    return x_train, y_train

def _test_decomposition(model, **kwargs):
    x_train, y_train = get_test_data()

    if hasattr(model, "fit_transform"):
        x_transformed = model.fit_transform(x_train, **kwargs)
    else:
        model.fit(x_train, **kwargs)
        x_transformed = model.transform(x_train)

    x_transformed = x_transformed.real
    for label in np.unique(y_train):
        center = np.mean(x_transformed[y_train==label], axis=0)
        var_within  = np.mean(np.sum(np.square(x_transformed[y_train==label] - center), axis=1))
        var_outside = np.mean(np.sum(np.square(x_transformed[y_train!=label] - center), axis=1))
        assert var_outside >= var_within

def test_pca():
    model = PCA(n_components=n_components)
    _test_decomposition(model)

def test_tsne():
    model = tSNE(
        initial_momentum=0.5,
        final_momoentum=0.8,
        eta=500,
        min_gain=0.1,
        tol=1e-05,
        prec_max_iter=50,
        random_state=seed
    )
    _test_decomposition(
        model,
        n_components=n_components,
        epochs=epochs,
        verbose=1
    )

def test_umap():
    model = UMAP(
        min_dist=0.1,
        spread=1.0,
        sigma_iter=40,
        sigma_init=1.0,
        sigma_tol=1e-5,
        sigma_lower=0,
        sigma_upper=np.inf,
        random_state=seed,
    )
    _test_decomposition(
        model,
        n_components=n_components,
        epochs=epochs,
        init_lr=1,
        verbose=-1
    )
