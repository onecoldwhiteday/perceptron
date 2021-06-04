class Input:
    def __init__(self, neuron, weight):
        self.neuron = neuron
        self.weight = weight

    def __repr__(self):
        return f"n: {self.neuron}, w: {self.weight}"
