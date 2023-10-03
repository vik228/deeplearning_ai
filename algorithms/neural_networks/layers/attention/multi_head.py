from __future__ import annotations

import numpy as np

from algorithms.neural_networks.layers.attention.scaled_dot_product_attention import ScaledDotProductAttention
from algorithms.neural_networks.layers.layer import Layer


class MultiHeadAttention(Layer):

    def __init__(self, num_heads) -> None:
        self.num_heads = num_heads
        self.heads = [ScaledDotProductAttention() for _ in range(num_heads)]
        self.weights = []
        self.weight_gradients = []

    def init_weights(self, queries, keys, value):
        if not self.weights:
            for _ in range(self.num_heads):
                WQ = np.random.randn(queries.shape[-1], keys.shape[-1])
                WK = np.random.randn(keys.shape[-1], keys.shape[-1])
                WV = np.random.randn(value.shape[-1], keys.shape[-1])
                self.weights.append((WQ, WK, WV))

    def forward_propagation(self, query, key, value, mask=None):
        self.init_weights(query, key, value)
        all_head_attention_weights = []
        for idx, head in enumerate(self.heads):
            query = np.matmul(query, self.weights[idx][0])
            key = np.matmul(key, self.weights[idx][1])
            value = np.matmul(value, self.weights[idx][2])
            attention_weights = head.forward_propagation(
                query, key, value, mask)
            all_head_attention_weights.append(attention_weights)
        return np.concatenate(all_head_attention_weights, axis=-1)

    def backward_propagation(self, output_error):
        total_dL_dQ = 0
        total_dL_dK = 0
        total_dL_dV = 0
        for idx, head in enumerate(self.heads):
            dL_dQ, dL_dK, dL_dV = head.backward_propagation(output_error)
            dL_dWQ = np.matmul(dL_dQ, self.weights[idx][0].T)
            dL_dWK = np.matmul(dL_dK, self.weights[idx][1].T)
            dL_dWV = np.matmul(dL_dV, self.weights[idx][2].T)
            self.weight_gradients.append((dL_dWQ, dL_dWK, dL_dWV))
            total_dL_dQ += dL_dWQ
            total_dL_dK += dL_dWK
            total_dL_dV += dL_dWV
        return total_dL_dQ, total_dL_dK, total_dL_dV
