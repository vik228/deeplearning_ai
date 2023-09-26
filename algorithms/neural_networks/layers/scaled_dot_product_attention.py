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

    def forward_propagation(self, query, key, value, mask=None):
        self.key = key
        self.query = query
        self.value = value
        score = np.matmul(query, key.T) / np.sqrt(self.key.shape[-1])
        if mask is not None:
            score += (mask * -1e9)
        self.attention = softmax(score, axis=-1)
        output = np.matmul(self.attention, value)
        return output

    def backward_propagation(self, output_error):
        dk = self.key.shape[-1]
        dL_df = output_error
        dL_dx = dL_df * self.value.T * softmax_prime(self.attention)
        dL_dQ = np.matmul(dL_dx, self.key) / np.sqrt(dk)
        dL_dk = np.matmul(dL_dx, self.query) / np.sqrt(dk)
        dL_dv = np.matmul(self.attention, dL_df)
        self.query_gradients = dL_dQ
        self.key_gradients = dL_dk
        self.value_gradients = dL_dv
        return dL_dQ, dL_dk, dL_dv
