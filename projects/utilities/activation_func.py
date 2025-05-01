import math
from typing import List
import numpy as np

def sigmoid(x: float) -> float:
    """
    Sigmoid activation function.
    :return: A float representing the sigmoid activation function.
    """
    return 1 / (1 + math.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """
    Leaky ReLU activation function.
    :return: A float representing the leaky ReLU activation function.
    """
    return x if x > 0 else alpha * x


def softmax(x):
    x = np.array(x, dtype=float)  # Ensures x is a numeric NumPy array
    shifted_x = x - np.max(x, axis=-1, keepdims=True)  # For numerical stability
    exp_x = np.exp(shifted_x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x