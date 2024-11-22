#ifndef ACTIVATION_H
# define ACTIVATION_H

#include <stdlib.h>

double  *relu(double *inputs, size_t nb_inputs);
double  *sigmoid(double *inputs, size_t nb_inputs);
double  *softmax(double *inputs, size_t nb_inputs);
double  *d_sigmoid(double *inputs, size_t nb_inputs);
double  *d_relu(double *inputs, size_t nb_inputs);


#endif