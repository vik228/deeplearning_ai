from __future__ import annotations


class SGD:

    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "weights") and hasattr(layer, "weight_gradients"):
                layer.weights -= self.learning_rate * layer.weights_gradient
            if hasattr(layer, "bias") and hasattr(layer, "bias_gradients"):
                layer.bias -= self.learning_rate * layer.bias_gradient
