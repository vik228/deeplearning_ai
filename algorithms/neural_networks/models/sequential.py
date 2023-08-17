from __future__ import annotations

from algorithms.neural_networks import losses
from optimizers.sgd import SGD


class Sequential:

    def __init__(self) -> None:
        self.layers = []
        self.all_losses = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def compile(self, loss, optimizer=SGD(learning_rate=0.1)) -> None:
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

    def predict(self, input):
        output = []
        if len(input.shape) == 2:
            input.reshape(1, *input.shape)
        batch_size = input.shape[0]
        for i in range(batch_size):
            output.append(self.forward(input[i]))
        return output

    def train(self, X, Y, epochs=1000):
        if len(X.shape) == 2:
            X.reshape(1, *X.shape)
        batch_size = X.shape[0]
        for epoch in range(epochs):
            batch_loss = 0
            for i in range(batch_size):
                input = X[i]
                predictions = self.forward(input)
                batch_loss += self.loss(Y[i], predictions)
                grad = self.loss_prime(Y[i], predictions)
                self.backward(grad)
                self.optimizer.step(self.layers)
            self.all_losses.append(batch_loss)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {batch_loss/batch_size}")
