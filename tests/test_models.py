# coding: # coding: utf-8
import numpy as np
from kerasy.models import Sequential
from kerasy.layers import Input, Dense
from kerasy import optimizers
from kerasy import metrics

from kerasy.utils import generate_test_data
from kerasy.utils import CategoricalEncoder

num_classes = 2

def get_test_data():
    (x_train, y_train), _ = generate_test_data(num_train=1000,
                                               num_test=200,
                                               input_shape=(10,),
                                               classification=True,
                                               num_classes=num_classes,
                                               random_state=123)
    encoder = CategoricalEncoder()
    y_train = encoder.to_onehot(y_train, num_classes)
    return x_train, y_train

def _test_build_classification_model(x_train, y_train):
    model = Sequential()
    model.add(Input(input_shape=(x_train.shape[1],)))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["categorical_accuracy"]
    )
    return model

def test_classification_model():
    x_train, y_train = get_test_data()
    model = _test_build_classification_model(x_train, y_train)
    model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=-1)
    y_pred = model.predict(x_train)
    scores = [metric.loss(y_pred, y_train) for metric in model.metrics]

    weights = model.get_weights()
    model_ = _test_build_classification_model(x_train, y_train)
    model_.set_weights(weights)
    y_pred_ = model.predict(x_train)
    scores_ = [metric.loss(y_pred_, y_train) for metric in model_.metrics]

    assert np.all(y_pred == y_pred_)
    assert scores==scores_
