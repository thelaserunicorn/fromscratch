# Multilayer perceptron
import numpy as np


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

    def forward_propagate(self, inputs):
       # first layer activation is the inputs
        activations = inputs

        for w in self.weights:
            # net inputs calculation
            net_inputs = np.dot(activations, w)

            # calculation of activation
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == "__main__":

    # INITIATE THE MLP
    mlp = MLP()
    # INPUTS
    inputs = np.random.rand(mlp.num_inputs)
    # FORWARD PROP
    output = mlp.forward_propagate(inputs)
    # RESULT
    print("The input of the network is: {}".format(inputs))

    print("The output of the network is: {}".format(output))
