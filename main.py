from keras.src.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np


def sigmoid_activation(x: float):
	return 1 / ( 1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
	layer_1 = np.dot(input_vector, weights) + bias
	layer_2 = sigmoid_activation(layer_1)
	return layer_2

def mse_loss(target, pred):
	return np.square(target - pred)

def deriv_loss(target, pred):
	return 2 * (pred - target)

learning_rate = 0.03

inputs = [np.array([1.66, 1.56]), np.array([2, 1.5])]
weights = [np.array([1.45, -0.66])]
expected_results = [np.array([1.0]), np.array([0.0])]
bias = np.array([0.0])


if __name__ == "__main__":
	for iter in range(50):
		cur_input = inputs[1] if iter % 2 == 0 else inputs[0]
		cur_expected = expected_results[1] if iter % 2 == 0 else expected_results[0]

		prediction = make_prediction(cur_input, weights[0], bias)
		deriv = deriv_loss(cur_expected, prediction)
		weights[0] = weights[0] - learning_rate * deriv
		error = mse_loss(cur_expected, prediction)
		print(f"Prediction for input {1 if iter % 2 == 0 else 2}, expected {int(cur_expected[0])}: {prediction}; Error: {error}")