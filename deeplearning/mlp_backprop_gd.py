# Multilayer perceptron
import numpy as np
from random import random

# saving activations and partial derivatives
# implement backprop and gradient descent


class MLP:

    def __init__(self, num_inputs=3, num_hidden_layer=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden_layer = num_hidden_layer
        self.num_outputs = num_outputs

        layers = [num_inputs] + num_hidden_layer + [num_outputs]

        # initiating weights
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros(layers[i], layers[i+1])
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
       # first layer activation is the inputs
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # net inputs calculation
            net_inputs = np.dot(activations, w)

            # calculation of activation
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        # activ_3 = s(h_3)
        # h_3 = activ_2 * W_2

        return activations

    def backpropagate(self, error):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            # ndarray[0.1, 0.2] --> [[0.1],[0.2]]
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(
                current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

    def gradient_descent(self, learning_rate=1):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights = weights + derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):

            net_error = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                # forward_propagate
                output = self.forward_propagate(input)
                # calculate error
                error = target - output
                # backpropagate
                self.backpropagate(error)
                # apply gradient_descent
                self.gradient_descent(learning_rate)

                net_error += self._mse(target, output)
            # report error
            print("Error: {} at epoch {}".format(net_error / len(inputs), i))

        print("Training is complete")
        print("======================")

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    # INITIATE THE MLP
    mlp = MLP()
    # TRAIN
    mlp.train(inputs, targets, 50, 0.1)
