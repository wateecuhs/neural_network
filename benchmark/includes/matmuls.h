#ifndef MATMULS_H
# define MATMULS_H

#include <stddef.h>
#include "layer.h"
#include <pthread.h>
#include <stdio.h>

typedef struct s_dims {
    size_t x;
    size_t y;
} t_dims;

void base_matmul(double *A, double *B, double *C, t_dims dims_a, t_dims dims_b);

#endif