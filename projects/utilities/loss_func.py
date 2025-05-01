import numpy as np

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def binary_cross_entropy(y_true, y_pred):
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce


def categorical_cross_entropy(y_true, y_pred):
    cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return cce


def cross_entropy_derivative(y_true, y_pred):
    # Derivative of loss w.r.t. softmax output
    return (y_pred - y_true) / y_true.shape[0]