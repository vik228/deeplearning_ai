from __future__ import annotations

from algorithms.neural_networks import losses
from optimizers.sgd import SGD


class Sequential:

    def __init__(self) -> None:
        self.layers = []
        self.all_losses = []

    def add(self, layer) -> None:
        self.layers.add(layer)

    def compile(self, loss, optimizer=SGD(learning_rate=0.01)) -> None:
        self.loss = losses.get_loss_function(loss)
        self.loss_prime = losses.get_loss_function(
            loss, return_detivative=True)
        self.optimizer = optimizer

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward_propagation(grad)

    def train(self, X, Y, iterations):
        for epoch in range(iterations):
            predictions = self.forward(X)
            loss = self.loss(Y, predictions)
            self.all_losses.append(loss)
            grad = self.loss_prime(Y, predictions)
            self.backward(grad)
            self.optimizer.step(self.layers)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {loss}')
