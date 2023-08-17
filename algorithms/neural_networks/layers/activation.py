from __future__ import annotations

from neural_networks import activations

from algorithms.neural_networks.layers.layer import Layer


class ActivationLayer(Layer):

    def __init__(self, activation) -> None:
        self.activation = activations.get_activation(activation,
                                                     return_detivative=False)
        self.activation_prime = activations.get_activation(
            activation, return_detivative=True)

    def forward_propagation(self, input):
        self.input = input
        return self.activation(input)

    def backward_propagation(self, output_error):
        return self.activation_prime(self.input) * output_error
