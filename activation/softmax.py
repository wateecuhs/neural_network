import numpy as np

class SoftMax:
	def forward(self, inputs):
		self.inputs = inputs
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities
		return self.output