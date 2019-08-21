# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

from ..activations import ActivationFunc
from ..losses import LossFunc
from ..layers.core import Input, Dense

class Sequential():
    def __init__(self, layer=None):
        self.layers = []
        self.epochs = 0
        if layer is not None: self.add(layer)
        self.loss = None
        self.config = None

    def add(self, layer):
        """Adds a layer instance."""
        self.layers.append(layer)

    def compile(self, loss, input_shape=None):
        """ Creates the layer weights. """
        self.loss = LossFunc(loss)
        units = [l.outdim for l in self.layers]
        for i,l in enumerate(self.layers):
            if l.name[-6:] == "inputs": continue
            l.build(indim=units[i-1])

    def fit(self, x_train, y_train, epochs=1000):
        goal_epochs = self.epochs+epochs
        digit=len(str(goal_epochs))
        for e in range(epochs):
            for x,y in zip(x_train,y_train):
                out = self.forward(x)
                self.backprop(y, out)
            self.epochs+=1
            y_pred = self.predict(x_train)
            mse = np.mean((y_pred-y_train)**2)
            if self.epochs % 100 == 99: print(f'[{self.epochs+1:{digit}d}/{goal_epochs:{digit}d}] mse={mse:{4}f}')

    def forward(self, input):
        out=input
        for l in self.layers:
            if l.name[-6:] == "inputs": continue
            out=l.forward(out)
        return out

    def backprop(self, y_true, out):
        dEdz_out = self.loss.diff(y_true, out)
        for l in reversed(self.layers):
            if l.name[-6:] == "inputs": continue
            dEdz_out = l.backprop(dEdz_out)

    def predict(self, x_train):
        if np.ndim(x_train) == 1:
            return self.forward(x_train)
        else:
            return np.array([self.forward(x) for x in x_train])
