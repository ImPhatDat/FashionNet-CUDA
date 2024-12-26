#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "layer.hh"
#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat>
#pragma once

class Softmax : public Layer
{
public:
    Softmax(int batch_size, int input_size);
    void forward(const __half *input, __half *output, dim3 blockSize) override;
    void backward(const __half *output_d, __half *input_d, dim3 blockSize) override;
};

#endif // SOFTMAX_H
