# This is a easy sample for the training and testing of BPNN
import numpy as np


def sigmoid():
    return lambda x : 1 / (1 + np.exp(-1 * x))


def MSE_LOSS(pre_res, gt):
    return np.mean((pre_res - gt) ** 2)


def derivation_sigmoid(x):
    fx = sigmoid()(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, bias, activation_fun):
        self.weights = weights
        self.bias = bias
        self.activation_fun = activation_fun

    def output(self, x):
        return self.activation_fun(np.dot(self.weights, x) + self.bias)

class Network:
    def __init__(self, activation_fun):
        self.w_1 = np.random.normal()
        self.w_2 = np.random.normal()
        self.w_3 = np.random.normal()
        self.w_4 = np.random.normal()
        self.w_5 = np.random.normal()
        self.w_6 = np.random.normal()

        self.b_1 = np.random.normal()
        self.b_2 = np.random.normal()
        self.b_3 = np.random.normal()

        self.neuron_h_1 = Neuron(np.array([self.w_1, self.w_2]), self.b_1, activation_fun)
        self.neuron_h_2 = Neuron(np.array([self.w_3, self.w_4]), self.b_2, activation_fun)
        self.neuron_o_1 = Neuron(np.array([self.w_5, self.w_6]), self.b_3, activation_fun)

    def feed_forward(self, x):
        output_1 = self.neuron_h_1.output(x)
        output_2 = self.neuron_h_2.output(x)
        output_3 = self.neuron_o_1.output(np.array([output_1, output_2]))
        return output_3

    def training(self, data, ground_truths, epoch):
        for i in range(epoch):
            for x, gt in zip(data, ground_truths):
                x_1 = x[0]
                x_2 = x[1]

                h_1 = self.neuron_h_1.output(x)
                h_2 = self.neuron_h_2.output(x)
                y_pred = self.neuron_o_1.output(np.array([h_1, h_2]))

                d_L_d_y = -2 * (gt - y_pred)
                d_y_d_h1 = self.w_5 * derivation_sigmoid(h_1 * self.w_5 + h_2 * self.w_6 + self.b_3)
                d_y_d_h2 = self.w_6 * derivation_sigmoid(h_1 * self.w_5 + h_2 * self.w_6 + self.b_3)

                d_h1_d_w1 = x_1 * derivation_sigmoid(x_1 * self.w_1 + x_2 * self.w_2 + self.b_1)
                d_h1_d_w2 = x_2 * derivation_sigmoid(x_1 * self.w_1 + x_2 * self.w_2 + self.b_1)

                d_h2_d_w3 = x_1 * derivation_sigmoid(x_1 * self.w_3 + x_2 * self.w_4 + self.b_2)
                d_h2_d_w4 = x_2 * derivation_sigmoid(x_1 * self.w_3 + x_2 * self.w_4 + self.b_2)

                d_y_d_w5 = h_1 * derivation_sigmoid(h_1 * self.w_5 + h_2 * self.w_6 + self.b_3)
                d_y_d_w6 = h_2 * derivation_sigmoid(h_1 * self.w_5 + h_2 * self.w_6 + self.b_3)

                d_h1_d_b1 = derivation_sigmoid(x_1 * self.w_1 + x_2 * self.w_2 + self.b_1)
                d_h2_d_b2 = derivation_sigmoid(x_1 * self.w_3 + x_2 * self.w_4 + self.b_2)
                d_y_d_b3 = derivation_sigmoid(h_1 * self.w_5 + h_2 * self.w_6 + self.b_3)

                # SGD
                eta = 0.1
                self.w_1 -= eta * d_L_d_y * d_y_d_h1 * d_h1_d_w1
                self.w_2 -= eta * d_L_d_y * d_y_d_h1 * d_h1_d_w2
                self.w_3 -= eta * d_L_d_y * d_y_d_h2 * d_h2_d_w3
                self.w_4 -= eta * d_L_d_y * d_y_d_h2 * d_h2_d_w4
                self.w_5 -= eta * d_L_d_y * d_y_d_w5
                self.w_6 -= eta * d_L_d_y * d_y_d_w6

                self.b_1 -= eta * d_L_d_y * d_y_d_h1 * d_h1_d_b1
                self.b_2 -= eta * d_L_d_y * d_y_d_h2 * d_h2_d_b2
                self.b_3 -= eta * d_L_d_y * d_y_d_b3

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Define dataset
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    all_y_trues = np.array([
        1,  # Alice
        0,  # Bob
        0,  # Charlie
        1,  # Diana
    ])


    activation_fun = sigmoid()

    network = Network(activation_fun)

    epoch = 1000

    network.training(data, all_y_trues, epoch)

    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print(network.feed_forward(emily)) # 0.951 - F
    print(network.feed_forward(frank)) # 0.039 - M
