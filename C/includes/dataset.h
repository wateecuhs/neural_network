#ifndef DATASET_H
# define DATASET_H

#include <stdint.h>

typedef struct s_dataset {
    uint8_t  **inputs;
    uint8_t *pred;
    int     nb_samples;
    int     inputs_len; 
} Dataset;

#endif