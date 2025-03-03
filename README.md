
# Neural Network

Building a neural network library from scratch in C to learn how they work

## Current tasks

- Cleaning up the code so its more flexible and library-like. (Dataset structure, train function)
- Optimization (l1 cache and whatnot idk what this means)

## Current progress

- Single digit OCR works, which was the original goal of this project

- Implemented basic MLP with dense layers
- Activations: ReLU, Sigmoid, Softmax
- Loss: Categorical Cross-Entropy
- Optimizers: Stochastic Gradient Descent (single sample at a time)
- Batches

## Todo

- ONNX Model saving/loading
- Implement batches for SGD
- Graphical interface for single digit OCR with input frame and NN representation

## Ressources

- [Sentdex](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [Optimizing Matrix Multiplication](https://coffeebeforearch.github.io/2020/06/23/mmul.html)