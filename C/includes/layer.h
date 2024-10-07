#ifndef LAYER_H
# define LAYER_H

#include <stddef.h>

typedef enum e_activation {
	NONE,
	SIGMOID,
	RELU,
	SOFTMAX
} Activation;

typedef struct s_layer {
	double	*inputs;
	double	*weights;
	double	*biases;
	double	*gradients;
	int		nb_neurons;
	int		inputs_len;
	int		capacity;
	double	*outputs;
	double	*(*activation)(double *inputs, size_t nb_inputs);
} Layer;

int		create_layer(Layer *layer, int nb_neurons_in, int nb_neurons_out, Activation activation);
int		layer_forward(Layer *layer, double *inputs, int nb_inputs);
void	free_layer(Layer *layer);

#endif