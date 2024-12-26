#ifndef RELU_H
#define RELU_H

#include "layer.hh"
#include <algorithm>
#pragma once

class ReLU : public Layer
{
private:
    int total_size;
public:
    ReLU(int batch_size, int input_size);

    void forward(const float *input, float *output, dim3 blockSize);
    void backward(const float *output_d, float *input_d, dim3 blockSize);
};

#endif // RELU_H
