# coding: utf-8
from kerasy.models import Sequential
from kerasy.layers import Input, Dense
from kerasy import optimizers
from kerasy import metrics

from kerasy.utils import generate_test_data
from kerasy.utils import CategoricalEncoder

num_classes = 2
metric = metrics.get("categorical_accuracy")

def get_test_data():
    (x_train, y_train), _ = generate_test_data(num_train=1000,
                                               num_test=200,
                                               input_shape=(10,),
                                               classification=True,
                                               num_classes=num_classes)
    encoder = CategoricalEncoder()
    y_train = encoder.to_onehot(y_train, num_classes)
    return x_train, y_train

def _test_optimizer(optimizer, target=0.75):
    x_train, y_train = get_test_data()
    model = Sequential()
    model.add(Input(input_shape=(x_train.shape[1],)))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(y_train.shape[1], activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[metric]
    )
    model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=-1)
    y_pred = model.predict(x_train)
    score = metric.loss(y_pred, y_train)

    assert score >= target

def test_sgd():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    _test_optimizer(sgd)

def test_rmsprop():
    rmsprop = optimizers.RMSprop()
    _test_optimizer(rmsprop)

def test_adgrad():
    adgrad = optimizers.Adagrad()
    _test_optimizer(adgrad)

def test_adadelta():
    adadelta = optimizers.Adadelta()
    _test_optimizer(adadelta)

def test_adam():
    adam = optimizers.Adam()
    _test_optimizer(adam)

def test_adamax():
    adamax = optimizers.Adamax()
    _test_optimizer(adamax)

def test_nadam():
    nadam = optimizers.Nadam()
    _test_optimizer(nadam)
