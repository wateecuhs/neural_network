#include "dataset.h"
#include <stdio.h>

void free_dataset(Dataset dataset)
{
    if (dataset.training_inputs)
        free(dataset.training_inputs);
    if (dataset.training_targets)
        free(dataset.training_targets);
    if (dataset.test_inputs)
        free(dataset.test_inputs);
    if (dataset.test_targets)
        free(dataset.test_targets);
}

void free_exit(Dataset dataset)
{
    free_dataset(dataset);
    exit(EXIT_FAILURE);
}

double *unpack_imgs(Dataset *dataset, char *path, int *inputs_len)
{
    unsigned char raw_headers[16];
    __uint32_t headers[4];
    double *ret = NULL;
    uint8_t *raw_values = NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0 || read(fd, raw_headers, 16) < 0)
        free_exit(*dataset);

    for (int i = 0; i < 4; i++) {
        memcpy(&headers[i], &raw_headers[i * 4], sizeof(__uint32_t));
        headers[i] = __builtin_bswap32(headers[i]);
    }
    if (headers[0] != 2051)
        free_exit(*dataset);

    *inputs_len = headers[1];
    if (!(raw_values = malloc(headers[1] * headers[2] * headers[3])) ||
        !(ret = malloc((headers[1] * headers[2] * headers[3]) * sizeof(double))))
        free_exit(*dataset);

    if (read(fd, raw_values, headers[1] * headers[2] * headers[3]) < 0)
        free_exit(*dataset);
    for (size_t i = 0; i < headers[1] * headers[2] * headers[3]; i++)
        ret[i] = raw_values[i] / 255.0;
    close(fd);
    return ret;
}

uint8_t *unpack_labels(Dataset *dataset, char *path)
{
    unsigned char raw_headers[8];
    __uint32_t headers[2];
    uint8_t *tmp = NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0 || read(fd, raw_headers, 8) < 0)
        free_exit(*dataset);

    for (int i = 0; i < 2; i++) {
        memcpy(&headers[i], &raw_headers[i * 4], sizeof(__uint32_t));
        headers[i] = __builtin_bswap32(headers[i]);
    }

    tmp = malloc(headers[1]);
    if (!tmp)
        free_exit(*dataset);

    if (read(fd, tmp, headers[1]) < 0)
        free_exit(*dataset);
    close(fd);
    return tmp;
}

Dataset unpack_mnist(void)
{
    Dataset dataset = {0};

    dataset.training_inputs = unpack_imgs(&dataset, "./archive/train-images.idx3-ubyte", &dataset.len_training);
    dataset.training_targets = unpack_labels(&dataset, "./archive/train-labels.idx1-ubyte");
    dataset.test_inputs = unpack_imgs(&dataset, "./archive/t10k-images.idx3-ubyte", &dataset.len_test);
    dataset.test_targets = unpack_labels(&dataset, "./archive/t10k-labels.idx1-ubyte");
    return dataset;
}
