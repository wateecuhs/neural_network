#ifndef DATASET_H
# define DATASET_H

#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

typedef struct s_dataset {
    uint8_t  *inputs;
    uint8_t *targets;
    int     nb_samples;
    int     inputs_len;
} Dataset;

Dataset unpack_mnist(void);

#endif