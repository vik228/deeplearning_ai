from __future__ import annotations

import numpy as np

from algorithms.neural_networks.activations import softmax
from algorithms.neural_networks.activations import softmax_prime
from algorithms.neural_networks.layers.layer import Layer


class ScaledDotProductAttention(Layer):

    def __init__(self) -> None:
        self.key = None
        self.query = None
        self.value = None
        self.attention = None
        self.query_gradients = None
        self.key_gradients = None
        self.value_gradients = None

    def forward_propagation(self, query, key, value):
        self.key = key
        self.query = query
        self.value = value
        score = np.matmul(query, key.T) / np.sqrt(self.key.shape[-1])
        self.attention = softmax(score, axis=-1)
        output = np.matmul(self.attention, value)
        return output

    def backward_propagation(self, output_error):
        self.softmax_prime = softmax_prime(self.attention)
        self.query_gradients = np.matmul(
            np.matmul(self.softmax_prime, self.key.T), self.value) / np.sqrt(
                self.key.shape[-1])
        self.key_gradients = np.matmul(
            np.matmul(self.softmax_prime, self.query), self.value) / np.sqrt(
                self.key.shape[-1])
        self.value_gradients = self.softmax_prime

        q_error = output_error * self.query_gradients
        k_error = output_error * self.key_gradients
        v_error = output_error * self.value_gradients
        return q_error, k_error, v_error
