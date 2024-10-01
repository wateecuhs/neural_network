import numpy as np

class Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		self.inputs = inputs

	def backward(self, dvalues):
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
		self.dinputs = np.dot(dvalues, self.weights.T)
