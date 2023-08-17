from __future__ import annotations

import numpy as np

from algorithms.neural_networks.layers.layer import Layer


class Dense(Layer):

    def __init__(self, units) -> None:
        self.units = units
        self.weights = None
        self.bias = None

    def init_parameters(self, input_shape):
        self.weights = np.random.rand(input_shape[-1], self.units) * 0.01
        self.bias = np.zeros((input_shape[-1], self.units))
        self.output_shape = (input_shape[0], self.units)
        self.weight_gradients = None
        self.bias_gradients = None

    def forward_propagation(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        self.weight_gradients = np.dot(self.input.T, output_error)
        self.bias_gradients = output_error
        return input_error
