# coding: utf-8

# coding: utf-8
from kerasy.ML.EM import KMeans, ElkanKMeans, HamerlyKMeans, MixedGaussian
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

def _test_EM(model, target=0.75, **kwargs):
    x_train, y_train = get_test_data()
    model.fit(x_train, max_iter=max_iter, verbose=-1, **kwargs)
    y_pred = model.predict(x_train)

    assert cluster_accuracy(y_train, y_pred) > target

def test_lloyd_kmeans():
    model = KMeans(n_clusters=num_clusters)
    _test_EM(model, tol=1e-4)

def test_elkan_kmeans():
    model = ElkanKMeans(n_clusters=num_clusters)
    _test_EM(model, tol=1e-4)

def test_hamerly_kmeans():
    model = HamerlyKMeans(n_clusters=num_clusters)
    _test_EM(model, tol=1e-4)

def test_mixed_gaussian():
    model = MixedGaussian(n_clusters=num_clusters, random_state=0)
    _test_EM(model)
