#ifndef RELU_H
#define RELU_H

#include "layer.hh"
#include <algorithm>
#pragma once

class ReLU : public Layer
{
public:
    ReLU(int batch_size, int input_size);

    void forward(const float *input, float *output);
    void backward(const float *output_d, float *input_d);
};

#endif // RELU_H
