#include "network.h"
#include "activation.h"
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include "loss.h"
#include <stdio.h>
#include "displays.h"
#include <string.h>
#include <strings.h>
#include "dataset.h"

void seed_rand(void);



int	main(void)
{
    seed_rand();
    Network	*nn;
    Dataset dataset = unpack_mnist();
    double *d_loss = NULL;

    nn = malloc(sizeof(Network));
    if (!nn)
        exit(EXIT_FAILURE);

    init_network(nn);
    add_layer(nn, 784, RELU);
    // add_layer(nn, 128, RELU);
    // add_layer(nn, 128, RELU);
    add_layer(nn, 10, SOFTMAX);

    double *inputs = malloc(784 * sizeof(double));
    double expected[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 1; i++)
    {
        memcpy(inputs, &dataset.inputs[i * 784], 784 * sizeof(uint8_t));
        nn_forward(nn, inputs, 784);
        d_loss = d_loss_softmax_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, 3);
        print_network_output(nn);
        nn_backward(nn, d_loss);
        printf("loss: %f\n", loss_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, expected));
        free(d_loss);
        for (int i = 0; i < nn->nb_layers; i++)
            update_parameters(&nn->layers[i], 0.01);
        bzero(expected, 10 * sizeof(double));
        expected[dataset.targets[i]] = 1;
    }
    free(inputs);
    free_network(nn);
    return (0);
}

void seed_rand(void)
{
    char rdm[4] = {};
    int urdm_fd = open("/dev/urandom", O_RDONLY);

    if (urdm_fd < 0 || read(urdm_fd, rdm, 4) < 0)
        exit(EXIT_FAILURE);

    srand(rdm[0] * rdm[1] * rdm[2] * rdm[3]);
}
