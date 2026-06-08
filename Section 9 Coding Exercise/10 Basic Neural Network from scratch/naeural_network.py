# import libraries

## matrix operation and mathematical function
import numpy as np

## Sigmoid activation function and its derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

## Mean Square Error (MSE) Loss Function
## Calculates the mean squared error between the true and the predicted output
## This is the loss function used to measure the performance of the model
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

## Basic Neural Network Class
class BasicNeuralNetwork:
    # Initialize by pass four items which is the object itself input size hidden size and output size which initializes the neural network with random weights and biases
    # weights_input_hidden is the weights between the input layer and the hidden layer
    # weights_hidden_output is the weights between the hidden layer and the output layer
    # bias_hidden is the bias for the hidden layer
    # bias_output - bias for the output layer
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    # To create the forward pass function it takes two parameters - where the input is passed through the network to produce an output
    def foreward(self, X):
        # his will give us the the first hidden input
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    # Backward pass and the weights update - where the error is propagated back through the network, and the weights and biases are updated using the gradient descent.
    def backwords(self, X, y, output, learning_rate):
        # This is output error is the difference between the actual output and the predicted output
        output_error = y - output
        # Adjustments for the output layer based on the error and the derivative of the sigmoid function
        output_delta = output_error * sigmoid_derivative(output)
        # This is the error for the hidden layer propagated from the output layer
        # This will give us the hidden error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        # Adjustments for the hidden layer.
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        # Get the weights and biases update which are gradients are used to update the weights and biases using gradient descent
        # So that will give us update the self weights hidden output
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        # this will give us backward pass and weights update
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Train the neural network - where the neural network is trained over multiple epochs to minimize the loss
    def train(self, X, y, epochs, learning_rate):
        # this trains the neural network for a specified number of epochs
        for epoch in range(epochs):
            # Create forward Pass
            output = self.foreward(X)

            # Create the backward pass
            self.backwords(X, y, output, learning_rate)

            # do the loss calculation which is every 100 epochs
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f"Epoch: {epoch}, Loss: {loss}")
            
# Training with the XOR data set
## Create XOR data set
## input data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
# target - expected output from the above above calculation
y = np.array([[0],[1],[1],[0]])

# Create a neural network instance with two input neurons two hidden neurons and one output neuron
nn = BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1)

## Train the neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

## After training The network's performance is evaluated by comparing predicted and actual outputs
## Test the neural network with the XOR data set.
print("\nTesting the trained neural network:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {nn.foreward(X[i])}, Actual Output: {y[i]}")