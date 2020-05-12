# coding: utf-8
from kerasy.ML.cluster import DBSCAN
from kerasy.utils import generateWholeCakes
from kerasy.utils import cluster_accuracy

num_clusters = 3
num_samples = 300

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

def test_dbscan(target=0.75):
    x_train, y_train = get_test_data()
    model = DBSCAN(eps=1)
    y_pred = model.fit_predict(x_train, verbose=-1)
    assert cluster_accuracy(y_train, y_pred) > target
