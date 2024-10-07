#include "layer.h"
#include "activation.h"
#include <stdlib.h>
#include <stdio.h>

int create_layer(Layer *layer, int nb_neurons_in, int nb_neurons_out, Activation activation)
{
	layer->outputs = malloc(nb_neurons_out * sizeof(double));
	layer->weights = malloc(nb_neurons_out * nb_neurons_in * sizeof(double));
	layer->biases = malloc(nb_neurons_out * sizeof(double));
	layer->gradients = malloc(nb_neurons_out * nb_neurons_in * sizeof(double));
	layer->inputs = malloc(nb_neurons_in * sizeof(double));
	if (!layer->outputs || !layer->weights || !layer->biases || !layer->gradients || !layer->inputs)
		return (-1);
	layer->nb_neurons = nb_neurons_out;
	layer->inputs_len = nb_neurons_in;
	layer->capacity = nb_neurons_out;

	switch (activation)
	{
		case SIGMOID:
			layer->activation = sigmoid;
			break;
		case RELU:
			layer->activation = relu;
			break;
		case SOFTMAX:
			layer->activation = softmax;
			break;
		default:
			layer->activation = NULL;
			break;
	}

	for (int i = 0; i < layer->nb_neurons; i++) {
		for (int j = 0; j < layer->inputs_len; j++)
			layer->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
		layer->biases[i] = .1;
	}
	return (0);
}

int layer_forward(Layer *layer, double *inputs, int nb_inputs)
{
	if (!layer || !inputs || nb_inputs != layer->inputs_len)
		return (-1);

	for (int i = 0; i < layer->nb_neurons; i++)
	{
		layer->outputs[i] = layer->biases[i];
		for (int j = 0; j < nb_inputs; j++) {
			layer->inputs[j] = inputs[j];
			layer->outputs[i] += layer->inputs[j] * layer->weights[i * nb_inputs + j];
		}
	}
	layer->activation(layer->outputs, layer->nb_neurons);
	return (0);
}

int layer_backward(Layer *layer, double *inputs, int nb_inputs)
{
	
	(void)layer;
	(void)inputs;
	(void)nb_inputs;
	return (0);
}

void free_layer(Layer *layer)
{
	if (!layer)
		return;
	if (layer->outputs)
		free(layer->outputs);
	if (layer->weights)
		free(layer->weights);
	if (layer->biases)
		free(layer->biases);
	if (layer->gradients)
		free(layer->gradients);
	if (layer->inputs)
		free(layer->inputs);
	
}