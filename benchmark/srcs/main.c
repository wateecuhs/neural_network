#include "network.h"
#include "activation.h"
#include "loss.h"
#include <stdio.h>
#include "dataset.h"
#include <sys/time.h>

void seed_rand(void);


int	main(void)
{
    srand(3111957);
    struct timeval start, end;

    gettimeofday(&start, NULL);
    Network	*nn;
    Dataset dataset = unpack_mnist();

    gettimeofday(&end, NULL);
    printf("Time to load dataset: %fms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
    gettimeofday(&start, NULL);

    nn = init_network(4, 64, 0.2);
    add_layer(nn, 784, RELU);
    add_layer(nn, 64, RELU);
    add_layer(nn, 32, RELU);
    add_layer(nn, 10, SOFTMAX);
    double *inputs = malloc(784 * sizeof(double));
    
    gettimeofday(&end, NULL);
    printf("Time to init Neural Network: %fms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
    gettimeofday(&start, NULL);
    
    train_network(nn, dataset, 1);
    
    gettimeofday(&end, NULL);
    printf("Time to train Neural Network: %fms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
    gettimeofday(&start, NULL);

    int pred;
    int correct = 0;
    for (int i = 0; i < dataset.len_test; i++)
    {
        memcpy(inputs, &dataset.test_inputs[i * 784], 784 * sizeof(double));
        pred = nn_predict(nn, inputs, 784);
        if (pred == dataset.test_targets[i])
            correct++;
    }
    gettimeofday(&end, NULL);
    printf("Time to predict: %fms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
    gettimeofday(&start, NULL);
    printf("Accuracy: %f\n", (double)correct / 10000);
    printf("Total correct: %d\n", correct);
    free(inputs);
    destroy_network(nn);
    return (0);
}
