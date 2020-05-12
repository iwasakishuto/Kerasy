# coding: utf-8
from kerasy.ML.HMM import GaussianHMM
from kerasy.utils import generateWholeCakes
from kerasy.utils import cluster_accuracy

num_clusters = 3
num_samples = 300
max_iter = 100

def get_test_data():
    x_train, y_train = generateWholeCakes(num_classes=num_clusters*2,
                                          num_samples=num_samples*2,
                                          r_low=3,
                                          r_high=10,
                                          add_noise=False,
                                          same=True,
                                          seed=123)
    mask = (y_train%2)==0
    x_train = x_train[mask]
    y_train = y_train[mask]
    return x_train, y_train

def _test_HMM(model, target=0.75, **kwargs):
    x_train, y_train = get_test_data()
    model.fit(x_train, max_iter=max_iter, verbose=-1, **kwargs)
    y_pred = model.predict(x_train)

    assert cluster_accuracy(y_train, y_pred) > target

def test_gaussian_hmm_spherical():
    model = GaussianHMM(
        n_hstates=num_clusters,
        covariance_type="spherical",
        random_state=0,
    )
    _test_HMM(model)

def test_gaussian_hmm_diag():
    model = GaussianHMM(
        n_hstates=num_clusters,
        covariance_type="diag",
        random_state=0,
    )
    _test_HMM(model)

def test_gaussian_hmm_full():
    model = GaussianHMM(
        n_hstates=num_clusters,
        covariance_type="full",
        random_state=0,
    )
    _test_HMM(model)

def test_gaussian_hmm_tied():
    model = GaussianHMM(
        n_hstates=num_clusters,
        covariance_type="tied",
        random_state=0,
    )
    _test_HMM(model)
