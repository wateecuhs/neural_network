import numpy as np

class Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
		self.bias = np.zeros((1, n_neurons))
	
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.bias
		return self.output
