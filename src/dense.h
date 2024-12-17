#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include "framework.h"
#pragma once

void initialize_dense(float *weights, float *biases, int rows, int cols, std::mt19937 &gen);

class Dense
{
private:
    int input_size = 0;
    int output_size = 0;
    int batch_size = 0;
    std::string activation_type = "none";

    float *weights = nullptr; // 1D array to represent weights (row-major)
    float *biases = nullptr;  // 1D array to represent biases

    // Gradients
    float *grad_weights = nullptr;
    float *grad_biases = nullptr;

public:
    Dense();
    Dense(int input_size, int output_size, int batch_size, std::string activation_type, std::mt19937 &gen);
    Dense &operator=(const Dense &other);
    ~Dense();

    int get_input_size() const { return input_size; }
    int get_batch_size() const { return batch_size; }
    int get_output_size() const { return output_size; }
    float *get_weights() const { return weights; }
    float *get_biases() const { return biases; }
    float *get_grad_weights() const { return grad_weights; }
    float *get_grad_biases() const { return grad_biases; }

    // Forward pass
    void forward(const float *input, float *output) const;

    // Backward pass
    void backward(const float *input, const float *grad_output, float *grad_input);

    void update_weights(float learning_rate);

};

#endif 