# coding: utf-8
from kerasy.ML.linear import LinearRegression, LinearRegressionRidge, LinearRegressionLASSO
from kerasy.utils import generateSin
from kerasy.utils import root_mean_squared_error

num_samples = 30

def get_test_data():
    x_train, y_train = generateSin(num_samples,
                                   xmin=0,
                                   xmax=1,
                                   seed=0)
    return x_train, y_train


def _test_linear_polynomial(Model, num_feature_arr=[1,2,4,8], **kwargs):
    x_train, y_train = get_test_data()

    for i,num_features in enumerate(sorted(num_feature_arr)):
        model = Model(basis="polynomial", exponent=range(1,num_features+1), **kwargs)

        model.fit(x_train,y_train)
        y_pred = model.predict(x_train)
        score = root_mean_squared_error(y_pred, y_train)

        assert i==0 or prev_score >= score
        prev_score = score

def test_linear():
    _test_linear_polynomial(LinearRegression)

def test_linear_ridge():
    _test_linear_polynomial(LinearRegressionRidge)

def test_linear_lasss():
    _test_linear_polynomial(LinearRegressionLASSO)
