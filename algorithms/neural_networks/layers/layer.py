from __future__ import annotations


class Layer:

    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_propagation(self, *arg, **kwargs):
        raise NotImplementedError

    def backward_propagation(self, output_error):
        raise NotImplementedError
