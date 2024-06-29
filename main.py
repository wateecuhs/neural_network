from loss.categorical_cross_entropy import CategoricalCrossEntropy
from activation.softmax import SoftMax
from activation.relu import ReLU
from layers.dense import Dense
from nnfs.datasets import spiral_data
import numpy as np
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 3)
activation1 = ReLU()

dense2 = Dense(3, 3)
activation2 = SoftMax()

loss_function = CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])
print(f"Loss: {loss}")