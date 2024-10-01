import numpy as np
from layers.dense import Dense

class SGD():
	def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
		self.learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.current_learning_rate = learning_rate
		self.momentum = momentum

	def pre_handling(self):
		self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

	def update_params(self, layer: Dense):
		if self.momentum:
			if not hasattr(layer, "weights_momentum"):
				layer.weights_momentum = np.zeros_like(layer.weights)
				layer.biases_momentum = np.zeros_like(layer.biases)

			weights_update = self.momentum * layer.weights_momentum - self.learning_rate * layer.dweights
			biases_update = self.momentum * layer.biases_momentum - self.learning_rate * layer.dbiases

			layer.weights_momentum = weights_update
			layer.biases_momentum = biases_update

		else:
			weights_update = -self.current_learning_rate * layer.dweights
			biases_update = -self.current_learning_rate * layer.dbiases


		layer.weights += weights_update
		layer.biases += biases_update
	
	def post_handling(self):
		self.iterations += 1