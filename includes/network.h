#ifndef NETWORK_H
# define NETWORK_H

#include "layer.h"

typedef struct s_network {
    int     nb_layers;
    int     capacity;
    Layer   *layers;
} Network;

Network *init_network(int capacity);
void    destroy_network(Network *network);
int     add_layer(Network *network, int nn, Activation activation);
int     add_layers(Network *nn, int nb_layers_to_add, int nb_neurons, Activation activation);
int     nn_forward(Network *nn, double *inputs, int nb_inputs);
int     nn_backward(Network *nn, double *loss);
void    print_network_output(Network *nn);
int     nn_predict(Network *nn, double *inputs, int nb_inputs);

#endif