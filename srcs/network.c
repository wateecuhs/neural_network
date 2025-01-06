#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include "loss.h"

Network *init_network(int capacity, int batch_size)
{
    
    Network *nn = malloc(sizeof(Network));
    if (!nn)
        return (NULL);

    if (batch_size < 1)
        batch_size = 1;
    nn->batch_size = batch_size;
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

    if (layer_forward(&nn->layers[0], inputs, nb_inputs) < 0)
        return (-1);
    for (int i = 1; i < nn->nb_layers; i++)
    {
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
        layer_backward(&nn->layers[i], nn->layers[i + 1].d_inputs);
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

void train_network(Network *nn, Dataset dataset, int epochs, double learning_rate)
{
    if (!nn || nn->nb_layers < 1)
        return;
    double *d_loss = NULL;
    double *inputs = malloc(nn->layers[0].inputs_len * sizeof(double));
    double expected[nn->layers[nn->nb_layers - 1].nb_neurons];
    printf("Batch size: %d\n", nn->batch_size);
    for (int i = 0; i < epochs; i++)
    {
        for (int j = 0; j < dataset.len_training / nn->batch_size; j++)
        {
            for (int k = 0; k < nn->batch_size; k++) {
                memcpy(inputs, &dataset.training_inputs[(j * nn->batch_size + k) * nn->layers[0].inputs_len], nn->layers[0].inputs_len * sizeof(double));
                nn_forward(nn, inputs, nn->layers[0].inputs_len);
                bzero(expected, nn->layers[nn->nb_layers - 1].nb_neurons * sizeof(double));
                expected[dataset.training_targets[j * nn->batch_size + k]] = 1;
                d_loss = d_loss_softmax_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, dataset.training_targets[j * nn->batch_size + k]);
                nn_backward(nn, d_loss);
                free(d_loss);
            }
            for (int k = 0; k < nn->nb_layers; k++)
                update_parameters(&nn->layers[k], learning_rate, nn->batch_size);
        }
        printf("Epoch %d\n", i);
    }
    printf("Training done\n");
    free(inputs);
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
