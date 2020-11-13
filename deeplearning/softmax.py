import numpy as np

inputs = [1, 1, 0]

exp_values = np.exp(inputs - np.max(inputs))

probs = exp_values / np.sum(exp_values)

print(probs)
print(np.sum(probs))
