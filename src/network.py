import math

from aiostream import operator

from src.layer import Layer


class Network:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return Network.sigmoid(x) * (1 - Network.sigmoid(x))

    def __init__(self, input_size, output_size, hidden_layers_count=1, learning_rate=0.5):
        self.activation_func = Network.sigmoid
        self.derivative_func = Network.sigmoid_derivative
        self.learning_rate = learning_rate

        self.layers = [Layer(input_size, None, self)]

        for i in range(0, hidden_layers_count):
            layer_size = min(input_size * 2 - 1, math.ceil(input_size * 2 / 3 + output_size))
            self.layers.append(Layer(layer_size, self.layers[-1], self))

        self.layers.append(Layer(output_size, self.layers[-1], self))

    def input(self, val):
        self.layers[0].input(val)

    @staticmethod
    def get_val(neuron):
        return neuron.value

    @property
    def prediction(self):
        result = []
        for i in self.layers[-1].neurons:
            result.append(i.value)
        return result

    def train_once(self, data_set):
        if isinstance(data_set, list):
            for data_case in data_set:
                i, expected = data_case

                self.input(i)
                print(self.layers[-1].neurons[0])
                for index, r in self.prediction:
                    self.layers[-1].neurons[index].__error = r - expected[index]
