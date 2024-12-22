#ifndef LAYER_H
#define LAYER_H

#include "../utils/helpers.hh"
#include <iostream>
#include <cstring>
#include <stdio.h>
#include <stdint.h>
#pragma once

class Layer
{
public:
    int batch_size;
    int input_size;
    int output_size;
    std::string name;    
protected:
    float *input;  // Pointer to store input values
    float *output; // Pointer to store output values

public:
    Layer(int batch_size, int input_size, int output_size);
    virtual ~Layer();
    
    virtual void forward(const float *input, float *output, dim3 blockSize) = 0;
    virtual void backward(const float *output_d, float *input_d, dim3 blockSize) = 0;
    virtual float* get_weights() const;
    virtual float* get_biases() const;
    virtual void update_weights(const float learning_rate, dim3 blockSize);

    virtual void load_weights(const float* weights, const float* biases);
};

#endif
