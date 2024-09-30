import numpy as np
from loss.categorical_cross_entropy import CategoricalCrossEntropy
from activation.softmax import SoftMax

class Activation_Softmax_Loss_CategoricalCrossentropy():
# Creates activation and loss function objects
	def __init__(self):
		self.activation = SoftMax()
		self.loss = CategoricalCrossEntropy()

	def forward(self, inputs, y_true):
		self.activation.forward(inputs)
		self.output = self.activation.output
		return self.loss.calculate(self.output, y_true)

	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)
		self.dinputs = dvalues.copy()
		self.dinputs[range(samples), y_true] -= 1
		self.dinputs = self.dinputs / samples