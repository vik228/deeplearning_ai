from __future__ import annotations

import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def categorical_crossentropy(y_true, y_pred):
    return np.mean(-np.log(y_pred) * y_true + np.log(1 - y_pred) * (1 - y_true))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def get_loss_function(loss_fn, return_detivative=False):
    loss_function = globals().get(loss_fn)
    if loss_function and callable(loss_function):
        if return_detivative:
            return globals().get(f"{loss_fn}_prime")
        return loss_function
    return None
