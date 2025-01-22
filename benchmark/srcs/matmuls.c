#include "matmuls.h"

    void base_matmul(double *A, double *B, double *C, t_dims dims_a, t_dims dims_b)
    {
        // this is assuming B is transposed
        // A is the input matrix of size nx1 with n being the number of inputs
        // B is the weight matrix of size nxm with n being the number of weights (or inputs) and m the number of neurons
        // C is the output matrix of size mx1 with m being the number of neurons
        // printf("i: %zu, j: %zu, k: %zu\n", i, j, k);
                    // double tmp_a = A[i * dims_a.x + k];
                    // double tmp_b = B[j * dims_b.x + k];
                    // printf("tmp_a: %f, tmp_b: %f\n", tmp_a, tmp_b);
        for (size_t i = 0; i < dims_a.y; i++) { // dims_a.y == 1
            for (size_t j = 0; j < dims_b.y; j++) { // dims_b.y == m, iterating over the neurons
                for (size_t k = 0; k < dims_b.x; k++) { // dims_b.x == n, iterating over the weights
                    
                    C[i * dims_b.y + j] += A[i * dims_a.x + k] * B[j * dims_b.x + k]; // dims_a.x == dimbs_b.x == n
                }
            }
        }
    }