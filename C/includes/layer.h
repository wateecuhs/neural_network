#ifndef LAYER_H
# define LAYER_H

#include "neuron.h"
typedef enum e_activation {
	NONE,
	SIGMOID,
	RELU,
} Activation;

typedef struct s_layer {
	int		nb_neurons;
	int		capacity;
	Neuron	*neurons;
	double	*outputs;
	double	(*activation)(double x);
} Layer;

int		create_layer(Layer *layer, int nb_neurons_in, int nb_neurons_out, Activation activation);
int		layer_forward(Layer *layer, double *inputs, int nb_inputs);
void	free_layer(Layer *layer);


#endif