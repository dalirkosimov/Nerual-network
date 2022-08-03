from sklearn import preprocessing
import numpy as np

rows = int(input("Rows: "))
columns = int(input("Columns: "))
elements = list(map(int, input().split()))
X = np.array(elements).reshape(rows, columns)
print("Input matrix: ")
print(X)


exp_values = np.exp(X)
normalised_values = (exp_values)/(np.sum(exp_values, axis=1, keepdims=True))
print(normalised_values)