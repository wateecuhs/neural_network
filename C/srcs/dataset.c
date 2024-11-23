#include "dataset.h"
#include <stdio.h>

void free_dataset(Dataset dataset)
{
    if (dataset.inputs)
        free(dataset.inputs);
    if (dataset.targets)
        free(dataset.targets);
    
}

void free_exit(Dataset dataset)
{
    free_dataset(dataset);
    exit(EXIT_FAILURE);
}

void unpack_imgs(Dataset *dataset)
{
    unsigned char raw_headers[16];
    __uint32_t headers[4];
    uint8_t *tmp = NULL;

    int fd = open("./archive/train-images.idx3-ubyte", O_RDONLY);
    if (fd < 0 || read(fd, raw_headers, 16) < 0)
        free_exit(*dataset);

    for (int i = 0; i < 4; i++) {
        memcpy(&headers[i], &raw_headers[i * 4], sizeof(__uint32_t));
        headers[i] = __builtin_bswap32(headers[i]);
    }
    if (headers[0] != 2051)
        free_exit(*dataset);

    dataset->inputs_len = headers[1];
    if (!(tmp = malloc(headers[1] * headers[2] * headers[3])) ||
        !(dataset->inputs = malloc((headers[1] * headers[2] * headers[3]) * sizeof(double))) ||
        !(dataset->targets = malloc(headers[1] * sizeof(uint8_t))))
        free_exit(*dataset);

    if (read(fd, tmp, headers[1] * headers[2] * headers[3]) < 0)
        free_exit(*dataset);
    for (size_t i = 0; i < headers[1] * headers[2] * headers[3]; i++)
        dataset->inputs[i] = tmp[i] / 255.0;
    close(fd);
}

void unpack_labels(Dataset *dataset)
{
    unsigned char raw_headers[8];
    __uint32_t headers[2];

    int fd = open("./archive/train-labels.idx1-ubyte", O_RDONLY);
    if (fd < 0 || read(fd, raw_headers, 8) < 0)
        free_exit(*dataset);

    for (int i = 0; i < 2; i++) {
        memcpy(&headers[i], &raw_headers[i * 4], sizeof(__uint32_t));
        headers[i] = __builtin_bswap32(headers[i]);
    }

    if (read(fd, dataset->targets, headers[1]) < 0)
        free_exit(*dataset);
    close(fd);
}

Dataset unpack_mnist(void)
{
    Dataset dataset = {0};

    unpack_imgs(&dataset);
    unpack_labels(&dataset);
    return dataset;
}
