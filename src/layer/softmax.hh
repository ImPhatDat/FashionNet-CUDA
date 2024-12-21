#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.hh"
#include <cmath>
#include <algorithm>
#pragma once

class Softmax : public Layer
{
public:
    Softmax(int batch_size, int input_size);
    void forward(const float *input, float *output) override;
    void backward(const float *output_d, float *input_d) override;
};

#endif // SOFTMAX_H
