import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

nnfs.init()

np.random.seed(0)

class dense_layer:
    def __init__(self, n_inputs, n_neurones):
        self.weights = np.random.randn(n_inputs, n_neurones) 
        self.biases = np.zeros((1,n_neurones))
    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

#ReLU means positive values gets passed and negative values get mappped to 0
class activation_ReLU:
    def forward_pass(self, inputs):
        self.output = np.maximum(0, inputs)

# in order to make sense of negative values (which will need to be deciphered when we do back propagation), we use the expontential 
# function, as it maps every value of x into a positive value of y, so every single x value (be it neg or pos),
# will have only one unique value of y (call it a many to a many to one function) 
class activation_softmax:
    def forward_pass(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalised_values = (exp_values)/(np.sum(exp_values, axis=1, keepdims=True))
        self.output = normalised_values

X, y = spiral_data(samples=100, classes=3)

dense1 = dense_layer(2,3)
activation1 = activation_ReLU()
dense2 = dense_layer(3,2)
activation2 = activation_softmax()

dense1.forward_pass(X)
activation1.forward_pass(dense1.output)

dense2.forward_pass(activation1.output)
activation2.forward_pass(dense2.output)
print(activation2.output[:3])

