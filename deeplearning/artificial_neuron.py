import math

def sigmoid(x):
    y = 1.0/(1 + math.exp(-x))
    return y

def activate(input, weights):
    # looping through inputs and multiplying them with respective weights
    
    h = 0
    for x, w in zip(input, weights):
        h += x*w
    # perform activation
    return sigmoid(h)


inputs = [.6,.5,.4]
weights = [.4,.5,.6]

output = activate(inputs, weights)
print(output)
