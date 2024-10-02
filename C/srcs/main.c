#include "network.h"
#include "activation.h"
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include "loss.h"
#include <stdio.h>

void seed_rand(void);

int	main(void)
{
	seed_rand();
	Network	*nn;

	nn = malloc(sizeof(Network));
	if (!nn)
		exit(EXIT_FAILURE);

	init_network(nn);
	if (add_layer(nn, 4, RELU) < 0)
		printf("1\n");
	if (add_layer(nn, 8, RELU) < 0)
		printf("2\n");
	if (add_layer(nn, 8, RELU) < 0)
		printf("3\n");
	if (add_layer(nn, 4, SOFTMAX) < 0)
		printf("4\n");
	// if (add_layers(nn, 3, 4, RELU) < 0)
	// {
	// 	free_network(nn);
	// 	exit(EXIT_FAILURE);
	// }

	double *inputs = malloc(4 * sizeof(double));
	for (int i = 0; i < 4; i++)
		inputs[i] = (double)(rand() % 5000) / 1000;

	if (nn_forward(nn, inputs, 4) < 0)
	{
		printf("Failure\n");
		free(inputs);
		free_network(nn);
		exit(EXIT_FAILURE);
	}
	double expected[4] = {0, 0, 0, 1};

	printf("loss : %f\n", cce_loss(nn->layers[nn->nb_layers - 1].outputs, nn->layers[nn->nb_layers - 1].nb_neurons, expected));
	for (int i = 0; i < nn->layers[nn->nb_layers - 1].nb_neurons; i++)
	{
		printf("%f", nn->layers[nn->nb_layers - 1].outputs[i]);
		if (i + 1 < nn->layers[nn->nb_layers - 1].nb_neurons)
			printf(" ");
		else
			printf("\n");
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
