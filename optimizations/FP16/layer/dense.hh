#ifndef DENSE_H
#define DENSE_H

#include "layer.hh"
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <curand_kernel.h>


#pragma once

void initialize_dense(__half *d_weights, __half *d_biases, int rows, int cols, dim3 blockSize, unsigned long seed);

void matmul(const __half *A, const __half *B, __half *C, int M, int K, int N, dim3 blockSize);
void transpose(const __half *in, __half *out, int M, int N);

class Dense : public Layer
{
private:
    __half *weights = nullptr; // 1D array to represent weights (row-major)
    __half *biases = nullptr;  // 1D array to represent biases

    // Gradients
    __half *grad_weights = nullptr;
    __half *grad_biases = nullptr;

public:
    Dense(int batch_size, int input_size, int output_size, bool init, std::mt19937 &gen);
    ~Dense();

    float *get_weights() const override;
    float *get_biases() const override;

    void forward(const __half *input, __half *output, dim3 blockSize) override;
    void backward(const __half *output_d, __half *input_d, dim3 blockSize) override;
    void update_weights(const __half learning_rate, dim3 blockSize) override;

    void load_weights(const float* weights, const float* biases) override;

};

#endif 