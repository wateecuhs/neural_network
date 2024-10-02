#include "layer.h"
#include "activation.h"
#include <stdlib.h>
#include <stdio.h>

int create_layer(Layer *layer, int nb_neurons_in, int nb_neurons_out, Activation activation)
{
	layer->neurons = malloc(nb_neurons_out * sizeof(Neuron));
	layer->outputs = malloc(nb_neurons_out * sizeof(double));
	if (!layer->neurons || !layer->outputs)
		return (-1);
	layer->nb_neurons = nb_neurons_out;
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
	for (int i = 0; i < layer->nb_neurons; i++)
	{
		if (init_neuron(&layer->neurons[i], nb_neurons_in) == -1)
			return (-1);
	}
	return (0);
}

int layer_forward(Layer *layer, double *inputs, int nb_inputs)
{
	if (!layer || !inputs)
		return (-1);

	for (int i = 0; i < layer->nb_neurons; i++)
	{
		if (neuron_forward(&layer->neurons[i], inputs, nb_inputs) < 0)
			return (-1);
		layer->outputs[i] = layer->neurons[i].output;
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
	if (layer->neurons)
	{
		for (int i = 0; i < layer->nb_neurons; i++)
			free_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}