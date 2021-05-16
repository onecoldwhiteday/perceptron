from random import random

from perceptron.src.input import Input


class Neuron:
    def __init__(self, layer, prev_layer):
        self.layer = layer
        self.inputs = self.define_input(prev_layer.neurons) if prev_layer else [0]

    @staticmethod
    def define_input(neurons):
        inputs = []
        for n in neurons:
            inputs.append(Input(n, random() - 0.5))
        return inputs

    def is_input_neuron(self) -> bool:
        return not isinstance(self.inputs[0], Input)

