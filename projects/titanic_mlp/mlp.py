import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from projects.utilities.activation_func import relu, relu_derivative, softmax
from projects.utilities.loss_func import categorical_cross_entropy, cross_entropy_derivative

class Mlp:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with He initialization and biases with zeros
        self.weights_input_hidden = np.random.randn(input_size, hidden_size).astype(np.float64) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size).astype(np.float64) * np.sqrt(2. / hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size), dtype=np.float64)
        self.bias_output = np.zeros((1, output_size), dtype=np.float64)
       
    def forward(self, X):
        self.input = X.astype(np.float64)  # Ensure input is float64
        # Input to hidden layer
        self.hidden_layer = np.dot(self.input, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_layer)
        # Hidden to output layer
        self.output_layer = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = softmax(self.output_layer)
        return self.final_output
    
    def backward(self, y_true, output, learning_rate=0.01):
        # Ensure y_true and output are float64 arrays
        y_true = y_true.astype(np.float64)
        output = output.astype(np.float64)
        # Calculate error at output layer (gradient of loss w.r.t. output)
        output_error = cross_entropy_derivative(y_true, output)  # shape (m, output_size)
        # Gradients for weights and biases from hidden to output
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)
        # Backpropagate error to hidden layer
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * relu_derivative(self.hidden_layer)
        # Gradients for weights and biases from input to hidden
        d_weights_input_hidden = np.dot(self.input.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)
        # Update weights and biases with gradient descent
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_output -= learning_rate * d_bias_output
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden
        
    def train(self, X, y, epochs=5000, learning_rate=0.01):
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y, output, learning_rate)
            if epoch % 100 == 0:
                loss = categorical_cross_entropy(y, output)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
                
    def predict(self, X):
        X = X.astype(np.float64)
        output = self.forward(X)
        return np.argmax(output, axis=1)
