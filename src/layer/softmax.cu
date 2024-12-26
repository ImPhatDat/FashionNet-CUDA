#include "softmax.hh"

Softmax::Softmax(int batch_size, int input_size)
    : Layer(batch_size, input_size, input_size) // Reuse parent constructor
{
    this->name = "softmax";
}


void Softmax::forward(const float *input, float *output)
{
    for (int b = 0; b < batch_size; ++b)
    {
        const float *input_row = input + b * input_size;
        float *output_row = output + b * input_size;

        // Numerical stability
        float max_val = *std::max_element(input_row, input_row + input_size);
        // Compute exponentials and their sum
        float sum_exp = 0.0f;
        for (int c = 0; c < input_size; ++c)
        {
            output_row[c] = std::exp(input_row[c] - max_val);
            sum_exp += output_row[c];
        }

        // Normalize
        for (int c = 0; c < input_size; ++c)
        {
            output_row[c] /= sum_exp;
            this->output[b * input_size + c] = output_row[c];
        }
    }
}

void Softmax::backward(const float *output_d, float *input_d)
{
    for (int b = 0; b < batch_size; ++b)
    {
        const float *output_row = this->output + b * input_size;
        const float *output_d_row = output_d + b * input_size;
        float *input_d_row = input_d + b * input_size;

        for (int i = 0; i < input_size; ++i)
        {
            input_d_row[i] = 0.0f;
            for (int j = 0; j < input_size; ++j)
            {
                input_d_row[i] += output_d_row[j] * output_row[j] * ((i == j) - output_row[i]);
            }
        }
    }
}