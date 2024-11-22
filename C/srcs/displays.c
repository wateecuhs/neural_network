#include "network.h"
#include "stdio.h"

void print_network_output(Network *nn)
{
    if (!nn || nn->nb_layers < 1)
        return;
    printf("Output: ");
    for (int i = 0; i < nn->layers[nn->nb_layers - 1].nb_neurons; i++)
    {
        printf("%f", nn->layers[nn->nb_layers - 1].outputs[i]);
        if (i + 1 < nn->layers[nn->nb_layers - 1].nb_neurons)
            printf(" ");
        else
            printf("\n");
    }
}