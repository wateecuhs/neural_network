#ifndef NETWORK_H
# define NETWORK_H

#include "layer.h"

typedef struct s_network {
	int		nb_layers;
	int		capacity;
	Layer	*layers;
} Network;

void	init_network(Network *network);
void	free_network(Network *network);
int		add_layer(Network *network, int nn, Activation activation);
int		add_layers(Network *nn, int nb_layers_to_add, int nb_neurons, Activation activation);
int		nn_forward(Network *nn, double *inputs, int nb_inputs);

#endif