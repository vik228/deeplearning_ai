from __future__ import annotations


class Layer:

    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error):
        raise NotImplementedError
