#include "neuron.h"
#include <stdlib.h>
#include "layer.h"
#include <stdio.h>
#include "activation.h"

int init_neuron(Neuron *neuron, int nn_prev_layer)
{
	neuron->input_len = nn_prev_layer;
	neuron->input = malloc(sizeof(double) * neuron->input_len);
	neuron->weights = malloc(sizeof(double) * neuron->input_len);
	if (!neuron->input || !neuron->weights)
		return (-1);
	for (int i = 0; i < neuron->input_len; i++)
		neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
	neuron->bias = .1;
	return (0);
}

int neuron_forward(Neuron *neuron, double *inputs, int nb_inputs)
{
	if (!neuron || !inputs || nb_inputs != neuron->input_len)
		return (-1);

	neuron->output = neuron->bias;
	for (int i = 0; i < nb_inputs; i++)
		neuron->output += inputs[i] * neuron->weights[i];
	return (0);
}

void free_neuron(Neuron *neuron)
{
	if (!neuron)
		return;
	if (neuron->input)
		free(neuron->input);
	if (neuron->weights)
		free(neuron->weights);
}