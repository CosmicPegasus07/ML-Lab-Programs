import numpy as np

# Define the sigmoid function for activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input and output data
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize the weights randomly
np.random.seed(1)
weights_1 = 2 * np.random.random((3, 4)) - 1
weights_2 = 2 * np.random.random((4, 1)) - 1

# Set the number of training iterations
num_iterations = 10000

# Perform backpropagation
for i in range(num_iterations):
    # Forward propagation
    layer_1 = sigmoid(np.dot(x, weights_1))
    layer_2 = sigmoid(np.dot(layer_1, weights_2))

    # Calculate the error
    error = y - layer_2

    # Backpropagation
    layer_2_delta = error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weights_2.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update the weights
    weights_2 += layer_1.T.dot(layer_2_delta)
    weights_1 += x.T.dot(layer_1_delta)

# Print the final output of the network
print(layer_2)
