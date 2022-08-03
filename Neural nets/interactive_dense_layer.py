import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

nnfs.init()

# input --> exponentiate --> noramlise --> relu --> output
X, y = spiral_data(100,3)


np.random.seed(0)

input_rows = int(input("Input matrix rows: "))
input_columns = int(input("Input matrix columns: "))
print("Enter elements: ")
input_elements = list(map(int, input().split()))
X = np.array(input_elements).reshape(input_rows, input_columns)
print("Input matrix: ")
print(X)


class dense_layer:
    def __init__(self, n_inputs, n_neurones):
        self.weights = np.random.randn(n_inputs, n_neurones) 
        self.biases = np.zeros((1,n_neurones))
    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

#ReLU means positive values gets passed and negative values get mappped to 0
class activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# in order to make sense of negative values (which will need to be deciphered when we do back propagation), we use the expontential 
# function, as it maps every value of x into a positive value of y, so every single x value (be it neg or pos),
# will have only one unique value of y (call it a many to a many to one function) 
class activation_softmax:
    def forward_pass(self, inputs):
        exp_values = np.exp(inputs) - np.max(inputs, axis=1, keepdims=True)
        normalised_values = (exp_values)/(np.sum(exp_values, axis=1, keepdims=True))
        self.output = normalised_values



n_inputs1 = int(input("Number of inputs [layer 1] "))
n_neurones1 = int(input("Number of neurones [layer 1]  "))

#n_inputs2 = int(input("Number of inputs [layer 2]: "))
#n_neurones2 = int(input("Number of neurones [layer 2]: "))


layer1 = dense_layer(n_inputs1, n_neurones1)

# layer2 = dense_layer(n_inputs2,n_neurones2)

activation1 = activation_ReLU()

print("layer 1: ")
layer1.forward_pass(X)
print(layer1.output)

activation1.forward(layer1.output)
print("Activation function: ")
print(activation1.output)



#print("layer 2: ")
#layer2.forward_pass(layer1.output)
#print(layer2.output)
