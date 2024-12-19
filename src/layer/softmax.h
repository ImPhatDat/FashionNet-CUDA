#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.h"
#include <cmath>
#include <algorithm>

class Softmax : public Layer
{
public:
    Softmax() {}
    ~Softmax() {}

    void forward(const float *input, float *output) override;
    void backward(const float *output_d, float *input_d) override;
};

#endif // SOFTMAX_H
