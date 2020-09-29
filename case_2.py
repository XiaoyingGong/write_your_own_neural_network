import numpy as np
from case_1 import Neuron

class OurNeuralNetwork:
    '''
        A neural network with:
        - 2 inputs
        - 2 hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (O1)
        Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs):
        output_h1 = self.h1.feedforward(inputs)
        output_h2 = self.h2.feedforward(inputs)
        output_o1 = self.o1.feedforward(np.array([output_h1, output_h2]))
        return output_o1

if __name__ == '__main__':
    our_nn = OurNeuralNetwork()
    inputs = np.array([2, 3])
    outputs = our_nn.feedforward(inputs)
