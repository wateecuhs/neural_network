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

    nn = init_network(4, 3);
    add_layer(nn, 784, RELU);
    add_layer(nn, 64, RELU);
    add_layer(nn, 32, RELU);
    add_layer(nn, 10, SOFTMAX);
    double *inputs = malloc(784 * sizeof(double));

    train_network(nn, dataset, 1, 0.1);
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
