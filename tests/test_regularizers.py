# coding: utf-8
from kerasy.models import Sequential
from kerasy.layers import Input, Dense
from kerasy import regularizers
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
                                               num_classes=num_classes,
                                               random_state=123)
    encoder = CategoricalEncoder()
    y_train = encoder.to_onehot(y_train, num_classes)
    return x_train, y_train

def _test_regularizer(regularizer, target=0.75):
    regularizer = regularizers.get(regularizer)
    x_train, y_train = get_test_data()
    model = Sequential()
    model.add(Input(input_shape=(x_train.shape[1],)))
    model.add(Dense(10, activation="relu", kernel_regularizer=regularizer))
    model.add(Dense(y_train.shape[1], activation="softmax", kernel_regularizer=regularizer))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[metric]
    )
    model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=-1)
    y_pred = model.predict(x_train)
    score = metric.loss(y_pred, y_train)
    assert score >= target

    return model.layers[-1].kernel

none_kernel = _test_regularizer("none")

def test_l1():
    l1 = regularizers.L1(lambda1=0.01)
    l1_kernel = _test_regularizer(l1)
    assert l1.loss(none_kernel) >= l1.loss(l1_kernel)

def test_l2():
    l2 = regularizers.L2(lambda2=0.01)
    l2_kernel = _test_regularizer(l2)
    assert l2.loss(none_kernel) >= l2.loss(l2_kernel)

def test_l1l2():
    l1l2 = regularizers.L1L2(lambda1=0.01, lambda2=0.01)
    l1l2_kernel = _test_regularizer(l1l2)
    assert l1l2.loss(none_kernel) >= l1l2.loss(l1l2_kernel)
