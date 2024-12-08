#include "network.h"
#include "activation.h"
#include "loss.h"
#include <stdio.h>
#include "dataset.h"

void seed_rand(void);


int	main(void)
{
    seed_rand();
    Network	*nn;
    Dataset dataset = unpack_mnist();
    double *d_loss = NULL;

    nn = init_network(4);
    add_layer(nn, 784, RELU);
    add_layer(nn, 64, RELU);
    add_layer(nn, 32, RELU);
    add_layer(nn, 10, SOFTMAX);

    double *inputs = malloc(784 * sizeof(double));
    double expected[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < dataset.len_training; i++)
    {
        // printf("Epoch %d\n", i);
        memcpy(inputs, &dataset.training_inputs[i * 784], 784 * sizeof(double));
        nn_forward(nn, inputs, 784);
        bzero(expected, 10 * sizeof(double));
        expected[dataset.training_targets[i]] = 1;
        d_loss = d_loss_softmax_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, dataset.training_targets[i]);
        // print_network_output(nn);
        nn_backward(nn, d_loss);
        // printf("Target: %d\n", dataset.training_targets[i]);
        // printf("Loss: %f\n\n", loss_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, expected));
        free(d_loss);
        for (int i = 0; i < nn->nb_layers; i++)
            update_parameters(&nn->layers[i], 0.001);
        if (i % 1000 == 0)
            printf("Epoch %d\n", i);
    }
    printf("Training done\n");
    int pred;
    int correct = 0;
    for (int i = 0; i < dataset.len_test; i++)
    {
        memcpy(inputs, &dataset.test_inputs[i * 784], 784 * sizeof(double));
        pred = nn_predict(nn, inputs, 784);
        if (pred == dataset.test_targets[i])
            correct++;
    }
    printf("Accuracy: %f\n", (double)correct / 10000);
    printf("Total correct: %d\n", correct);
    free(inputs);
    destroy_network(nn);
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
