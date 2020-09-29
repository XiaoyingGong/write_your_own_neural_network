# 如何自己从零实现一个神经网络? - 量子位的回答 - 知乎
# https://www.zhihu.com/question/314879954/answer/638380202

import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sigmoid(self, x):
        # activation function:f(x) = 1/(1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs):
        result = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(result)



if __name__ == '__main__':
    weights = np.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)

    x = np.array([2, 3])
    print(n.feedforward(x))
