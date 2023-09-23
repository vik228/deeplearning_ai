from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return np.where(z > 0, 1, 0)


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - np.tanh(z)**2


def softmax(z, axis=0):
    return np.exp(z) / np.expand_dims(np.sum(np.exp(z), axis=axis), axis)


def softmax_prime(z, axis=0):
    s = softmax(z)
    return np.diagflat(s) - np.outer(s, s)


def get_activation(activation, return_detivative=False):
    activation_func = globals().get(activation)
    if activation_func and callable(activation_func):
        if return_detivative:
            return globals().get(f"{activation}_prime")
        return activation_func
    return None
