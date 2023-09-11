from __future__ import annotations

import numpy as np

from algorithms.neural_networks import activations
from algorithms.neural_networks.layers.layer import Layer


class SimpleRNN(Layer):

    def __init__(self,
                 units,
                 return_sequences=False,
                 activation="tanh") -> None:
        self.units = units
        self.input_weight = None
        self.hidden_weight = None
        self.hidden_bias = None
        self.input_weight_gradients = None
        self.hidden_weight_gradients = None
        self.weights_initialised = False
        self.activation = activations.get_activation(activation,
                                                     return_detivative=False)
        self.activation_prime = activations.get_activation(
            activation, return_detivative=True)
        self.return_sequences = return_sequences
        self.hidden_states = []
        self.z_t = []

    def initialise_weights(self, input_shape):
        if not self.weights_initialised:
            self.input_weight = np.random.rand(input_shape[-1],
                                               self.units) - 0.5
            self.hidden_weight = np.random.rand(self.units, self.units) - 0.5
            self.hidden_bias = np.zeros((input_shape[0], self.units))
            self.weights_initialised = True

    def forward_propagation(self, input):
        self.input = input
        if len(input.shape) == 2:
            input.reshape(*input.shape, 1)
        self.hidden_states = []
        self.z_t = []
        h_t = np.zeros((input.shape[0], self.units))
        self.initialise_weights(input.shape)
        time_steps = input.shape[1]
        for t in range(time_steps):
            input_t = input[:, t, :]
            hidden_t_input = np.dot(input_t, self.input_weight)
            hidden_t_prev_input = np.dot(h_t, self.hidden_weight)
            z_t = hidden_t_input + hidden_t_prev_input + self.hidden_bias
            self.z_t.append(z_t)
            h_t = self.activation(z_t)
            self.hidden_states.append(h_t)
        if self.return_sequences:
            return np.stack(self.hidden_states, axis=1)
        return self.hidden_states[-1]

    def backward_propagation(self, output_error):
        dWh = np.zeros_like(self.hidden_weight)
        dWx = np.zeros_like(self.input_weight)
        db = np.zeros_like(self.hidden_bias)
        dh_next = np.zeros((output_error.shape[0], self.units))
