#include "network.h"
#include "stdio.h"

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