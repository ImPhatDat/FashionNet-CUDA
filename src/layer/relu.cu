#include "relu.h"

void ReLU::forward(const float *input, float *output)
{
    int total_size = batch_size * input_size;
    for (int i = 0; i < total_size; ++i)
    {
        output[i] = std::max(0.0f, input[i]);
        this->input[i] = input[i];
    }
}

void ReLU::backward(const float *output_d, float *input_d)
{
    int total_size = batch_size * input_size;
    for (int i = 0; i < total_size; ++i)
    {
        input_d[i] = output_d[i] * (this->input[i] > 0);
    }
}