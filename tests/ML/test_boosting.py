# coding: utf-8
import numpy as np
from kerasy.ML.boosting import L2Boosting
from kerasy.models import Sequential
from kerasy.layers import Input, Dense
from kerasy import metrics

from kerasy.utils import generate_test_data
from kerasy.utils import CategoricalEncoder

num_models = 5

def get_test_regression_data():
    (x_train, y_train), _ = generate_test_data(num_train=1000,
                                               num_test=200,
                                               input_shape=(10,),
                                               output_shape=(1,),
                                               classification=False,
                                               random_state=123)
    return x_train, y_train

def get_test_classification_data():
    num_classes = 2
    (x_train, y_train), _ = generate_test_data(num_train=1000,
                                               num_test=200,
                                               input_shape=(10,),
                                               num_classes=num_classes,
                                               classification=True,
                                               random_state=123)
    encoder = CategoricalEncoder()
    y_train = encoder.to_onehot(y_train, num_classes)
    return x_train, y_train

def _test_adaboost():
    metric = metrics.get("categorical_accuracy")
    x_train, y_train = get_test_classification_data()

    Models = []
    weak_model_losses = []
    for _ in range(num_models):
        model = Sequential()
        model.add(Input(input_shape=(x_train.shape[1],)))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(y_train.shape[1], activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[metric]
        )
        model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=-1)
        weak_model_losses.append(metric.loss(model.predict(x_train), y_train))
        Models.append(model)

    boosting = AdaBoost(Models)
    boosting.fit(x_train, y_train, verbose=-1)
    y_boosting_pred = boosting.predict(x_train)
    boosting_loss = metric.loss(y_boosting_pred, y_train)

    assert boosting_loss <= min(weak_model_losses)

def test_l2boosting():
    metric = metrics.get("mse")
    x_train, y_train = get_test_regression_data()

    Models = []
    weak_model_losses = []
    for _ in range(num_models):
        model = Sequential()
        model.add(Input(input_shape=(x_train.shape[1],)))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(y_train.shape[1], activation="linear"))
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=[metric]
        )
        model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=-1)
        weak_model_losses.append(metric.loss(model.predict(x_train), y_train))
        Models.append(model)

    boosting = L2Boosting(Models)
    boosting.fit(x_train, y_train, verbose=-1)
    y_boosting_pred = boosting.predict(x_train)
    boosting_loss = metric.loss(y_boosting_pred, y_train)

    assert boosting_loss <= min(weak_model_losses)
