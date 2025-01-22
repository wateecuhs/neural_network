#include "layer.h"
#include "activation.h"
#include "loss.h"
#include <math.h>
#include "matmuls.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int create_layer(Layer *layer, size_t nb_neurons_in, size_t nb_neurons_out, Activation activation)
{
    layer->outputs = calloc(nb_neurons_out, sizeof(double));
    layer->weights = calloc(nb_neurons_out * nb_neurons_in, sizeof(double));
    layer->biases = calloc(nb_neurons_out, sizeof(double));
    layer->d_weights = calloc(nb_neurons_out * nb_neurons_in, sizeof(double));
    layer->inputs = calloc(nb_neurons_in, sizeof(double));
    layer->d_inputs = calloc(nb_neurons_in, sizeof(double));
    layer->d_biases = calloc(nb_neurons_out, sizeof(double));
    if (!layer->outputs || !layer->weights || !layer->biases || !layer->d_weights || !layer->inputs)
        return (-1);
    layer->nb_neurons = nb_neurons_out;
    layer->inputs_len = nb_neurons_in;
    layer->capacity = nb_neurons_out;

    layer->activation_type = activation;
    switch (activation)
    {
        case SIGMOID:
            layer->activation = sigmoid;
            layer->d_activation = d_sigmoid;
            break;
        case RELU:
            layer->activation = relu;
            layer->d_activation = d_relu;
            break;
        case SOFTMAX:
            layer->activation = softmax;
            layer->d_activation = NULL;
            break;
        default:
            free_layer(layer);
            return (-1);
    }
    // Xavier initialization
    double limit = sqrt(2.0 / (nb_neurons_in + nb_neurons_out));
    for (size_t i = 0; i < layer->nb_neurons; i++) {
        for (size_t j = 0; j < layer->inputs_len; j++) {
            layer->weights[i * layer->inputs_len + j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
        layer->biases[i] = .1;
    }
    return (0);
}

int layer_forward(Layer *layer, double *inputs, size_t nb_inputs)
{
    if (!layer || !inputs || nb_inputs != layer->inputs_len)
        return (-1);

    memcpy(layer->inputs, inputs, nb_inputs * sizeof(double));
    memcpy(layer->outputs, layer->biases, layer->nb_neurons * sizeof(double));

    base_matmul(layer->inputs, layer->weights, layer->outputs, (t_dims){layer->inputs_len, 1}, (t_dims){layer->inputs_len, layer->nb_neurons}); // nx1 & nxm 
    // for (size_t i = 0; i < layer->nb_neurons; i++) {
    //     layer->outputs[i] = layer->biases[i];
    //     for (size_t j = 0; j < nb_inputs; j++)
    //         layer->outputs[i] += layer->inputs[j] * layer->weights[i * nb_inputs + j];
    // }

    layer->activation(layer->outputs, layer->nb_neurons);
    return (0);
}

int layer_backward(Layer *layer, double *d_outputs)
{
    double *activation_derivatives = NULL;
    if (layer->activation_type != SOFTMAX)
        activation_derivatives = layer->d_activation(layer->outputs, layer->nb_neurons);
    else {
        activation_derivatives = malloc(layer->nb_neurons * sizeof(double));
        if (!activation_derivatives)
            return -1;
        for (size_t i = 0; i < layer->nb_neurons; i++)
            activation_derivatives[i] = 1;
    }

    for (size_t i = 0; i < layer->nb_neurons; i++) {
        double d_output = d_outputs[i] * activation_derivatives[i];
        for (size_t j = 0; j < layer->inputs_len; j++)
            layer->d_weights[i * layer->inputs_len + j] += d_output * layer->inputs[j];
        layer->d_biases[i] += d_output;
    }
    for (size_t i = 0; i < layer->inputs_len; i++) {
        layer->d_inputs[i] = 0;
        for (size_t j = 0; j < layer->nb_neurons; j++)
            layer->d_inputs[i] += d_outputs[j] * activation_derivatives[j] * layer->weights[j * layer->inputs_len + i];
    }
    if (activation_derivatives)
        free(activation_derivatives);
    return (0);
}

void update_parameters(Layer *layer, double learning_rate, size_t batch_size)
{
    for (size_t i = 0; i < layer->nb_neurons; i++) {
        for (size_t j = 0; j < layer->inputs_len; j++) {
            layer->weights[i * layer->inputs_len + j] -= learning_rate * (layer->d_weights[i * layer->inputs_len + j] / batch_size);
            layer->d_weights[i * layer->inputs_len + j] = 0;
        }
        layer->biases[i] -= learning_rate * (layer->d_biases[i] / batch_size);
        layer->d_biases[i] = 0;
    }
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
    if (layer->inputs)
        free(layer->inputs);
    if (layer->d_inputs)
        free(layer->d_inputs);
    if (layer->d_biases)
        free(layer->d_biases);
    if (layer->d_weights)
        free(layer->d_weights);
}