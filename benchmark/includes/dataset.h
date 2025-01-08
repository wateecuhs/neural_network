#ifndef DATASET_H
# define DATASET_H

#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

typedef struct s_dataset {
    double  *training_inputs;
    uint8_t *training_targets;
    int     len_training;

    double  *test_inputs;
    uint8_t *test_targets;
    int     len_test;

} Dataset;

Dataset unpack_mnist(void);

#endif