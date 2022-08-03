import numpy as np

# 4 inputs going into three neurones, giving out 3 outputs
# raw method
# 1 by 3 matrix 
input = [1, 2, 3, 4]

weight1 = [0.5, 1.2, 2.4, 5.7]
weight2 = [0.5, 1.2, 2.4, 5.7]
weight3 = [2.3, 2.1, 2.5, 3.6]


bias1 = 1
bias2 = 2
bias3 = 3


output = [input[0]*weight1[0] + input[1]*weight1[1] + input[2]*weight1[2] +input[3]*weight1[3] + bias1,
          input[0]*weight2[0] + input[1]*weight2[1] + input[2]*weight2[2] +input[3]*weight2[3] + bias2,
          input[0]*weight3[0] + input[1]*weight3[1] + input[2]*weight3[2] +input[3]*weight3[3] + bias3]

# dot product method

# 3 by 4 matrix 
weights = [[0.5, 1.2, 2.4, 5.7],
           [0.5, 1.2, 2.4, 5.7],
           [2.3, 2.1, 2.5, 3.6]]

biases = [1,2,3]

# weights first as it splits the matrix into each array and selects the dot product individually
output2 = np.dot(weights, input) + biases 

print(output, output2)