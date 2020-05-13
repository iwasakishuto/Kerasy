# coding: utf-8
from kerasy.ML.svm import hardSVC, SVC
from kerasy.utils import generateWhirlpool

num_samples = 150
max_iter = 10

def get_test_data():
    x_train, y_train = generateWhirlpool(num_samples,
                                         xmin=0,
                                         xmax=4,
                                         seed=0)
    return x_train, y_train

def _test_svm(model, target=0.75):
    x_train, y_train = get_test_data()
    model.fit(x_train, y_train, max_iter=max_iter, sparse_memorize=False, verbose=-1)
    assert model.accuracy(x_train, y_train) >= target

def test_hard_svc():
    model = hardSVC(kernel="gaussian", sigma=1.0)
    _test_svm(model, target=0.75)

def test_soft_svc():
    model = SVC(kernel="gaussian", sigma=1.0, C=10)
    _test_svm(model, target=0.75)
