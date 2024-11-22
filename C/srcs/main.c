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

void seed_rand(void);

void unpack_mnist(void)
{
    int fd = open("./archive/train-images.idx3-ubyte", 0);

    unsigned char titles[16];

    int o = read(fd, titles, 16);
    __uint32_t magic_number;
    __uint32_t num_images;
    __uint32_t num_rows;
    __uint32_t num_columns;

    memcpy(&magic_number, &titles[0], sizeof(__uint32_t));
    memcpy(&num_images, &titles[4], sizeof(__uint32_t));
    memcpy(&num_rows, &titles[8], sizeof(__uint32_t));
    memcpy(&num_columns, &titles[12], sizeof(__uint32_t));

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_columns = __builtin_bswap32(num_columns);

    printf("Read %d bytes\n", o);
    printf("Magic number: %d\n", magic_number);
    printf("Number of images: %d\n", num_images);
    printf("Number of rows: %d\n", num_rows);
    printf("Number of columns: %d\n", num_columns);
}

int	main(void)
{
    seed_rand();
    Network	*nn;
    double *d_loss = NULL;
    unpack_mnist();
    return 1;
    nn = malloc(sizeof(Network));
    if (!nn)
        exit(EXIT_FAILURE);

    init_network(nn);
    add_layer(nn, 784, RELU);
    add_layer(nn, 128, RELU);
    add_layer(nn, 128, RELU);
    add_layer(nn, 10, SOFTMAX);

    double *inputs = malloc(784 * sizeof(double));
    printf("Inputs\n");
    for (int i = 0; i < 784; i++) {
        printf("%f ", (double)(rand() % 5000) / 1000);
        inputs[i] = (double)(rand() % 5000) / 1000;
    }
    printf("\n");

    double expected[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 20; i++)
    {
        nn_forward(nn, inputs, 784);
        d_loss = d_loss_softmax_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, 3);
        nn_backward(nn, d_loss);
        free(d_loss);
        for (int i = 0; i < nn->nb_layers; i++)
            update_parameters(&nn->layers[i], 0.01);
        print_network_output(nn);
        printf("Loss : %f\n", loss_cce(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, expected));
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
