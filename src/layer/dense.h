#ifndef DENSE_H
#define DENSE_H

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include "layer.h"

#pragma once

void initialize_dense(float *weights, float *biases, int rows, int cols, std::mt19937 &gen);

void matmul(const float *A, const float *B, float *C, int M, int K, int N);

class Dense : public Layer
{
private:
    float *weights = nullptr; // 1D array to represent weights (row-major)
    float *biases = nullptr;  // 1D array to represent biases

    // Gradients
    float *grad_weights = nullptr;
    float *grad_biases = nullptr;

public:
    Dense();
    Dense(int batch_size, int input_size, int output_size, std::mt19937 &gen);
    ~Dense();

    float *get_weights() const { return weights; }
    float *get_biases() const { return biases; }
    float *get_grad_weights() const { return grad_weights; }
    float *get_grad_biases() const { return grad_biases; }

    void forward(const float *input, float *output);
    void backward(const float *output_d, float *input_d);

    void update_weights_and_biases(float learning_rate);

};

#endif 