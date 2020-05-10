# coding: utf-8
import pickle
import numpy as np
import warnings

from .base_layer import Layer
from ..layers import Input
from ..layers import Dropout

from .. import optimizers
from .. import losses
from .. import metrics as _metrics
from .. import activations

from ..utils import make_batches
from ..utils import flush_progress_bar
from ..utils import handleTypeError
from ..utils import print_summary
from ..utils import Table
from ..utils import ProgressMonitor
from ..utils import handleRandomState
from ..utils import KerasyImprementationWarning

class Sequential():
    def __init__(self, random_state=None):
        self.layers = []
        self.rnd = handleRandomState(random_state)

    def add(self, layer):
        """Adds a layer instance."""
        if not isinstance(layer, Layer):
            raise TypeError(f"The added layer must be an instance of class Layer. Found: {str(layer)}")
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=[]):
        """ Creates the layer weights.
        @param optimizer: (String name of optimizer) or (Optimizer instance).
        @param loss     : (String name of loss function) or (Loss instance).
        @param metrics  : (List) Metrics to be evaluated by the model during training and testing.
        """
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)
        self.metrics = [_metrics.get(metric) for metric in set(metrics+[loss])]
        self.activation = activations.get("linear")

        input_layer = self.layers[0]
        handleTypeError(
            types=[Input], input_layer=input_layer,
            msg_="The initial layer should be Input Layer"
        )
        output_shape = input_layer.input_shape
        for layer in self.layers:
            output_shape = layer.build(output_shape)

        # TODO: Kerasy didn't support the computational graph, so it may occur to
        #       disappear the gradients in the middle of the backpropagation even though
        #       computational graph could convey them though to the end.
        if (self.loss.name == "categorical_crossentropy") and (
                hasattr(self.layers[-1], "activation") and \
                self.layers[-1].activation.name=="softmax"):
            self.layers[-1].activation = activations.get("linear")     # softmax -> linear
            self.loss = losses.get("softmax_categorical_crossentropy") # categorical crossentropy -> softmax categorical crossentropy
            self.activation = activations.get("softmax")               # linear -> softmax
            # Warnings.
            warnings.warn("When calculating the \033[34mCategoricalCrossentropy\033[0m loss and the derivative " + \
            "of the \033[34mSoftmax\033[0m layer, the gradient disappears when backpropagating the actual value, " + \
            "so the \033[34mSoftmaxCategoricalCrossentropy\033[0m is implemented instead.", category=KerasyImprementationWarning)

    def fit(self,
            x=None, y=None, batch_size=32, epochs=1, verbose=1, shuffle=True,
            validation_spilit=0, validation_data=None, validation_steps=None,
            class_weight=None, sample_weight=None, **kwargs):
        if kwargs:
            raise TypeError(f'Unrecognized keyword arguments: {str(kwargs)}')
        if (x is None) or (y is None):
            raise ValueError('Please specify the trainig data. (x,y)')
        # Prepare validation data.
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                x_val, y_val = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                x_val, y_val, val_sample_weight = validation_data
            else:
                raise ValueError(f"When passing validation_data, it must contain 2 (x_val, y_val) or 3 (x_val, y_val, val_sample_weights) items. However, it contains {len(validation)} items.")
            num_val_samples = len(x_val)

        # Prepare for the trainig.
        num_train_samples = len(x)
        batches = make_batches(num_train_samples, batch_size)
        num_batchs = len(batches)
        index_array = np.arange(num_train_samples)

        metrics = self.metrics
        num_metrics = len(metrics)

        for epoch in range(epochs):
            if shuffle:
                self.rnd.shuffle(index_array)

            monitor = ProgressMonitor(
                max_iter=num_batchs, verbose=verbose,
                barname=f"Epoch {epoch+1:>0{len(str(epochs))}}/{epochs} |"
            )
            metrics_vals = [0.]*num_metrics

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                num_curl_samples = min((batch_index+1)*batch_size, num_train_samples)
                batch_ids = index_array[batch_start:batch_end]
                for bs, (x_train, y_true) in enumerate(zip(x[batch_ids], y[batch_ids])):
                    y_pred = self.forward_train(x_train)
                    self.backprop(y_true=y_true, y_pred=y_pred)
                    for i,metric in enumerate(metrics):
                        metrics_vals[i]+=metric.loss(y_true=y_true, y_pred=y_pred)

                self.updates(bs+1)
                metric_contents = {
                    metric.name : metric.format_spec(
                        metric.aggr_method(metric_val, num_curl_samples)
                    ) for metric, metric_val in zip(metrics, metrics_vals)
                }
                monitor.report(it=batch_index, **metric_contents)

            if do_validation:
                y_val_pred = self.predict(x_val)
                metric_contents.update({
                    "val_" + metric.name : metric.format_spec(
                        metric.loss(y_true=y_val, y_pred=y_val_pred)
                    ) for metric in metrics
                })
                monitor.report(it=batch_index, **metric_contents)

            monitor.remove()

    def forward_train(self, input):
        out=input
        for layer in self.layers:
            out = layer.forward(out)
        return self.activation.forward(out)

    def forward_test(self, input):
        out=input
        for layer in self.layers:
            if isinstance(layer, Dropout):
                continue
            out = layer.forward(out)
        return self.activation.forward(out)

    def backprop(self, y_true, y_pred):
        dEdXout = self.loss.diff(y_true, y_pred)
        for layer in reversed(self.layers):
            dEdXout = layer.backprop(dEdXout)

    def predict(self, x_train):
        if np.ndim(x_train) == 1:
            return self.forward_test(x_train)
        else:
            return np.array([self.forward_test(x) for x in x_train])

    def updates(self, batch_size):
        for layer in reversed(self.layers):
            layer.update(self.optimizer, batch_size)
        self.optimizer.iterations += 1

    def summary(self):
        print_summary(self)

    @property
    def weights(self):
        return self.get_weights()

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_weights(self, weights):
        for layer,weight in zip(self.layers, weights):
            layer.set_weights(weight)

    def save_weights(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)

    def is_trainable(self):
        layers = self.layers
        num_layers = len(layers)

        table = Table()
        table.set_cols(colname="id", values=range(num_layers), zero_padding=True, width=len(str(num_layers)))
        table.set_cols(colname="name", values=[l.name for l in layers], align=">")
        table.set_cols(colname="trainable", values=[str(l.trainable) for l in layers], align="^", color="blue")
        table.show()
