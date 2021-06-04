from src.neuron import Neuron


class Layer:
    def __init__(self, neurons_count, previous_layer, network):
        self.network = network
        self.neurons = []
        for i in range(0, neurons_count):
            self.neurons.append(Neuron(self, previous_layer))

    def __repr__(self):
        return f"{self.neurons}"

    @property
    def is_first_layer(self):
        return self.neurons[0].is_input_neuron

    def input(self, val):
        if not self.is_first_layer and not isinstance(val, list) and len(val) == len(self.neurons):
            for (v, i) in val:
                self.neurons[i].input(v)
