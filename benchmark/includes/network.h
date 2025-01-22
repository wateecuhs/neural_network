#ifndef NETWORK_H
# define NETWORK_H

#include "layer.h"
#include "dataset.h"

typedef struct s_network {
    size_t  nb_layers;
    size_t  batch_size;
    double  learning_rate;
    size_t  capacity;
    Layer   *layers;
} Network;

Network *init_network(size_t capacity, size_t batch_size, double learning_rate);
void    destroy_network(Network *network);
int     add_layer(Network *network, size_t nb_neurons, Activation activation);
int     add_layers(Network *nn, size_t nb_layers_to_add, size_t nb_neurons, Activation activation);
int     nn_forward(Network *nn, double *inputs, size_t nb_inputs);
int     nn_backward(Network *nn, double *loss);
void    print_network_output(Network *nn);
int     nn_predict(Network *nn, double *inputs, size_t nb_inputs);
void    train_network(Network *nn, Dataset dataset, size_t epochs);

#endif