import json.decoder
from random import random

from input import Input

from functools import reduce


class Neuron:
    def __init__(self, layer, prev_layer):
        self.__error = None
        self.layer = layer
        self.inputs = self.define_input(prev_layer.neurons) if prev_layer else [0]

    def __str__(self):
        return f"{self.value}, error: {self.__error},\n inputs: {self.inputs}\n\n"

    @staticmethod
    def define_input(neurons):
        inputs = []
        for n in neurons:
            inputs.append(Input(n, random() - 0.5))
        return inputs

    @property
    def is_input_neuron(self) -> bool:
        return not isinstance(self.inputs[0], Input)

    @staticmethod
    def add(x, y):
        if not isinstance(x, Input) or not isinstance(y, Input):
            return 0
        return x.neuron.value * x.weight + y.neuron.value + y.weight

    @property
    def input_sum(self):
        return reduce(Neuron.add, self.inputs)

    @property
    def value(self):
        return self.inputs[0] if self.is_input_neuron else self.layer.network.activation_func(self.input_sum)

    def input(self, val):
        if not self.is_input_neuron:
            self.inputs[0] = val

    def error(self, error):
        if not self.is_input_neuron:
            w_delta = error * self.layer.network.derivative_func(self.input_sum)

            for i in self.inputs:
                i.weight -= i.neuron.value * w_delta * self.layer.network.learning_rate
                i.neuron.__error = i.weight * w_delta
