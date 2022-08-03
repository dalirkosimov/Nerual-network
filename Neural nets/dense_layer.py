import numpy as np

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

layer1 = dense_layer(3,3)
layer2 = dense_layer(3,3)

layer1.forward_pass(X)
print(layer1.output)

