import numpy as np

# input matrix 

input_rows = int(input("Input matrix rows: "))
input_columns = int(input("Input matrix columns: "))
print("Enter elements: ")
input_elements = list(map(int, input().split()))
inputs = np.array(input_elements).reshape(input_rows, input_columns)
print("Input matrix: ")
print(inputs)

# weights matrix 

weights_rows = int(input("Weights matrix rows: "))
weights_columns = int(input("Weights matrix columns: "))
print("Enter elements: ")
weights_elements = list(map(int, input().split()))
weights = np.array(weights_elements).reshape(weights_rows, weights_columns)
print("Weights matrix: ")
print(weights)

# bias matrix

bias_length = int(input("Bias array length: "))
bias = list(map(int, input("Enter bias array: ").strip().split())) [:bias_length]
print("Bias:")
print(bias)

# dot crosser and output

output = np.dot(inputs, np.array(weights).T) + bias
print("Output: ")
print(output)
