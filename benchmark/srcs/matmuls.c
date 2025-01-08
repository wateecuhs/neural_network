#include <stddef.h>
#include "layer.h"


// void base_matmul(double *A, double *B, double *C, t_dims dims_a, t_dims dims_b)
// {
//     for (size_t i = 0; i < dims_a.y; i++) {
//         for (size_t j = 0; j < dims_b.x; j++) {
//             C[i * dims_b + j] = 0;
//             for (size_t k = 0; k < dims_a.x; k++) {
//                 C[i * dims_b + j] += A[i * dims_a + k] * B[k * dims_b + j];
//             }
//         }
//     }
// }