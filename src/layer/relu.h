#ifndef RELU_H
#define RELU_H

#include "layer.h"
#include <algorithm>

class ReLU : public Layer
{
public:
    ReLU() {}
    ~ReLU() {}

    void forward(const float *input, float *output);
    void backward(const float *output_d, float *input_d);
};

#endif // RELU_H
