#ifndef LAYER_H
# define LAYER_H

#include <stddef.h>

typedef enum e_type {
    DENSE,
} LayerType;

typedef enum e_activation {
    NONE,
    SIGMOID,
    RELU,
    SOFTMAX
} Activation;

typedef struct s_layer {
    // LayerType   type;
    double      *inputs;
    double      *outputs;
    double      *weights;
    double      *biases;
    double      *d_weights;
    double      *d_biases;
    double      *d_inputs;
    int         nb_neurons;
    int         inputs_len;
    int         capacity;
    int         batch_size;
    Activation  activation_type;
    double      *(*activation)(double *inputs, size_t nb_inputs);
    double      *(*d_activation)(double *inputs, size_t nb_inputs);
} Layer;

int     create_layer(Layer *layer, int nb_neurons_in, int nb_neurons_out, Activation activation);
int     layer_forward(Layer *layer, double *inputs, int nb_inputs);
int     layer_backward(Layer *layer, double *d_outputs);
void    free_layer(Layer *layer);
void    update_parameters(Layer *layer, double learning_rate);

#endif