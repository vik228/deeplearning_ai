from __future__ import annotations

import numpy as np

from algorithms.neural_networks.layers.layer import Layer


class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8) -> None:
        self.mean = None
        self.std = None
        self.gamma = None
        self.beta = None
        self.epsilon = epsilon

    def forward_propagation(self, input):
        self.input = input
        self.mean = np.mean(input, axis=-1, keepdims=True)
        self.std = np.std(input, axis=-1, keepdims=True)
        if self.gamma is None:
            self.gamma = np.ones((1, input.shape[-1]))
        if self.beta is None:
            self.beta = np.ones((1, input.shape[-1]))
        self.x_norm = (input - self.mean) / (self.std + self.epsilon)
        return self.x_norm * self.gamma + self.beta

    def backward_propagation(self, output_error):
        N, D = self.input.shape
        dgamma = np.sum(output_error * self.x_norm, axis=0, keepdims=True)
        dbeta = np.sum(output_error, axis=0, keepdims=True)
        dx_norm = output_error * self.gamma
        dstd_inv = np.sum(dx_norm * (self.input - self.mean),
                          axis=-1,
                          keepdims=True)
        dstd = -1.0 / (self.std + self.epsilon)**2 * dstd_inv
        dmu = np.sum(
            dx_norm * -1 / np.sqrt(self.std + self.epsilon),
            axis=-1,
            keepdims=True) + dstd * np.mean(
                -2.0 * (self.input - self.mean), axis=-1, keepdims=True)
        dx = dx_norm / np.sqrt(self.std + self.epsilon) + dstd * 2.0 * (
            self.input - self.mean) / D + dmu / D
        return dx, dgamma, dbeta
