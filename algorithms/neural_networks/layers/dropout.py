from __future__ import annotations

import numpy as np

from algorithms.neural_networks.layers.layer import Layer


class Dropout(Layer):

    def __init__(self, rate=0.5, mode='train') -> None:
        self.rate = rate
        self.mask = None
        self.mode = mode

    def forward_propagation(self, input):
        if self.mode == 'train':
            self.mask = np.random.rand(*input.shape) > self.rate
            return (input * self.mask) / (1 - self.mask)
        return input

    def backward_propagation(self, output_error):
        if self.mode == 'train':
            return (output_error * self.mask) / (1 - self.rate)
        return output_error
