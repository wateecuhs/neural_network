#include "activation.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>

double *relu(double *inputs, size_t nb_inputs)
{
    if (!inputs || nb_inputs <= 0)
        return (NULL);
    for (size_t i = 0; i < nb_inputs; i++)
        inputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    return inputs;
}

double *sigmoid(double *inputs, size_t nb_inputs)
{
    if (!inputs || nb_inputs <= 0)
        return (NULL);
    for (size_t i = 0; i < nb_inputs; i++)
        inputs[i] = (1 / (1 + exp(-inputs[i])));
    return (inputs);
}

double *softmax(double *inputs, size_t nb_inputs)
{
    if (!inputs || nb_inputs <= 0)
        return (NULL);

    double max = inputs[0];

    for (size_t i = 0; i < nb_inputs; i++)
    {
        if (inputs[i] > max)
            max = inputs[i];
    }

    double sum = 0;
    for (size_t i = 0; i < nb_inputs; i++)
        sum += exp(inputs[i] - max);
    
    for (size_t i = 0; i < nb_inputs; i++)
        inputs[i] = exp(inputs[i] - max - log(sum));

    return (inputs);
}

double *d_relu(double *inputs, size_t nb_inputs)
{
    double *d_inputs = malloc(sizeof(double) * nb_inputs);
    if (!d_inputs)
        return (NULL);

    for (size_t i = 0; i < nb_inputs; i++)
    {
        if (inputs[i] <= 0)
            d_inputs[i] = 0;
        else
            d_inputs[i] = 1;
    }
    return d_inputs;
}

double *d_sigmoid(double *inputs, size_t nb_inputs)
{
    double *d_inputs = malloc(sizeof(double) * nb_inputs);
    if (!d_inputs)
        return (NULL);

    for (size_t i = 0; i < nb_inputs; i++)
        d_inputs[i] = inputs[i] * (1 - inputs[i]);
    return d_inputs;
}