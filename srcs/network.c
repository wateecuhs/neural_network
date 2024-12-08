#include "network.h"
#include <stdlib.h>
#include <stdio.h>

Network *init_network(int capacity)
{
    
    Network *nn = malloc(sizeof(Network));
    if (!nn)
        return (NULL);

    nn->capacity = capacity;
    if (capacity < 1)
        capacity = 4;

    nn->nb_layers = 0;
    nn->layers = malloc(sizeof(Layer) * nn->capacity);
    if (!nn->layers)
    {
        free(nn);
        return (NULL);
    }
    return (nn);
}

int add_layer(Network *nn, int nb_neurons, Activation activation)
{
    int ret_val;

    if (nn->nb_layers >= nn->capacity)
    {
        nn->capacity = nn->nb_layers + 1;
        nn->layers = realloc(nn->layers, nn->capacity * sizeof(Layer));
    }
    if (nn->nb_layers == 0) 
        ret_val = create_layer(&nn->layers[nn->nb_layers], nb_neurons, nb_neurons, activation);
    else
        ret_val = create_layer(&nn->layers[nn->nb_layers], nn->layers[nn->nb_layers - 1].nb_neurons, nb_neurons, activation);
    if (ret_val < 0)
        return (-1);
    nn->nb_layers++;
    return 0;
}

int add_layers(Network *nn, int nb_layers_to_add, int nb_neurons, Activation activation)
{
    int ret_val;
    for (int i = 0; i < nb_layers_to_add; i++)
    {
        ret_val = add_layer(nn, nb_neurons, activation);
        if (ret_val < 0)
            return (ret_val);
    }
    return (0);
}

int nn_forward(Network *nn, double *inputs, int nb_inputs)
{
    if (!nn || !inputs || nn->nb_layers < 1 || nb_inputs != nn->layers[0].nb_neurons)
        return (-1);

    // printf("Layer 0\n");
    if (layer_forward(&nn->layers[0], inputs, nb_inputs) < 0)
        return (-1);
    for (int i = 1; i < nn->nb_layers; i++)
    {
        // printf("Layer %d\n", i);
        if (layer_forward(&nn->layers[i], nn->layers[i - 1].outputs, nn->layers[i - 1].nb_neurons) < 0)
            return (-1);
    }
    return (0);
}

int nn_predict(Network *nn, double *inputs, int nb_inputs)
{
    if (!nn || !inputs || nn->nb_layers < 1 || nb_inputs != nn->layers[0].nb_neurons)
        return (-1);

    if (layer_forward(&nn->layers[0], inputs, nb_inputs) < 0)
        return (-1);
    for (int i = 1; i < nn->nb_layers; i++)
    {
        if (layer_forward(&nn->layers[i], nn->layers[i - 1].outputs, nn->layers[i - 1].nb_neurons) < 0)
            return (-1);
    }
    int pred = 0;
    for (int i = 0; i < nn->layers[nn->nb_layers - 1].nb_neurons; i++)
    {
        if (nn->layers[nn->nb_layers - 1].outputs[i] > nn->layers[nn->nb_layers - 1].outputs[pred])
            pred = i;
    }
    return (pred);
}

int nn_backward(Network *nn, double *loss)
{
    if (!nn)
        return (-1);
    layer_backward(&nn->layers[nn->nb_layers - 1], loss);
    for (int i = nn->nb_layers - 2; i >= 0; --i)
    {
        layer_backward(&nn->layers[i], nn->layers[i + 1].d_inputs);
    }
    return (0);
}

void destroy_network(Network *nn)
{
    if (!nn)
        return;
    if (nn->layers)
    {
        for (int i = 0; i < nn->nb_layers; i++)
            free_layer(&nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}


void print_network_output(Network *nn)
{
    if (!nn || nn->nb_layers < 1)
        return;
    printf("Prediction: ");
    int pred = 0;
    for (int i = 0; i < nn->layers[nn->nb_layers - 1].nb_neurons; i++)
    {
        if (nn->layers[nn->nb_layers - 1].outputs[i] > nn->layers[nn->nb_layers - 1].outputs[pred])
            pred = i;
    }
    printf("%d\n", pred);
    
    for (int i = 0; i < nn->layers[nn->nb_layers - 1].nb_neurons; i++)
    {
        printf("%f", nn->layers[nn->nb_layers - 1].outputs[i]);
        if (i + 1 < nn->layers[nn->nb_layers - 1].nb_neurons)
            printf(" ");
        else
            printf("\n");
    }
}
