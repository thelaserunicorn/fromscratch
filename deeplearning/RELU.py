import numpy as np


class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs):
        # Calculating output values from inputs
        self.output = np.maximum(0, inputs)
