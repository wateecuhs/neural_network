from loss.categorical_cross_entropy import CategoricalCrossEntropy
from loss.softmax_cce import Activation_Softmax_Loss_CategoricalCrossentropy
from activation.softmax import SoftMax
from activation.relu import ReLU
from optimizer.sgd import SGD
from layers.dense import Dense
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np

######################
X, y = spiral_data(samples=100, classes=3)
dense1 = Dense(2, 64)
activation1 = ReLU()
dense2 = Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = SGD(learning_rate=1, decay=0.001, momentum=.5)
losses = []
acc = []

for epoch in range(50001):
	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	loss = loss_activation.forward(dense2.output, y)

	predictions = np.argmax(loss_activation.output, axis=1)

	if len(y.shape) == 2:
		y = np.argmax(y, axis=1)

	accuracy = np.mean(predictions==y)

	loss_activation.backward(loss_activation.output, y)
	dense2.backward(loss_activation.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	optimizer.pre_handling()
	if epoch % 100 == 0:
		print(f"epoch {epoch}: acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}")
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_handling()
	acc.append(accuracy)
	losses.append(loss)

figure, axis = plt.subplots(2, 1)
axis[0].plot(acc)
axis[0].set_title("Accuracy")
axis[1].plot(losses)
axis[1].set_title("Loss")
plt.show()