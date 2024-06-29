import numpy as np
from layers.dense import Dense
from activation.relu import ReLU
from activation.softmax import SoftMax
from keras.src.datasets import mnist
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def sigmoid(x: float):
	return 1 / ( 1 + np.exp(-x))

def sigmoid_deriv(x: float):
	return sigmoid(x) * (1-sigmoid(x))

def make_prediction(input_vector, weights, bias):
	layer_1 = np.dot(input_vector, weights) + bias
	layer_2 = sigmoid(layer_1)
	return layer_2

def mse_loss(target, pred):
	return np.square(target - pred)

def deriv_loss(target, pred):
	return 2 * (pred - target)

learning_rate = 0.03

layer1 = Dense(784, 100)
activation1 = ReLU()

layer2 = Dense(100, 10)
activation2 = SoftMax()

layer1.forward(x_train[0].flatten())
print(layer1.output.shape)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)
print(activation2.output.shape)
print(y_train[0])

print(np.sum(activation2.output))
plt.imshow(x_train[0])
plt.show()
