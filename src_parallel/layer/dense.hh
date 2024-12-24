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


class Dense : public Layer
{
private:
    float *weights = nullptr; // 1D array to represent weights (row-major)
    float *biases = nullptr;  // 1D array to represent biases

    // Gradients
    float *grad_weights = nullptr;
    float *grad_biases = nullptr;

public:
    Dense(int batch_size, int input_size, int output_size, bool init, std::mt19937 &gen);
    ~Dense();

    void initialize_dense(float *d_weights, float *d_biases, int rows, int cols, std::mt19937 &gen);
    void matmul(const float *A, const float *B, float *C, int M, int K, int N, dim3 blockSize);
    void transpose(const float *in, float *out, int M, int N, dim3 blockSize);

    float *get_weights() const override;
    float *get_biases() const override;

    void forward(const float *input, float *output, dim3 blockSize) override;
    void backward(const float *output_d, float *input_d, dim3 blockSize) override;
    void update_weights(const float learning_rate, dim3 blockSize) override;

    void load_weights(const float* weights, const float* biases) override;

};

#endif 