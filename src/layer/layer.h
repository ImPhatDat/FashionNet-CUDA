#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <cstring>

class Layer
{
public:
    int batch_size;
    int input_size;
    int output_size;
protected:
    float *input;  // Pointer to store input values
    float *output; // Pointer to store output values

public:
    Layer();
    Layer(int batch_size, int input_size, int output_size);
    virtual ~Layer();
    
    virtual void forward(const float *input, float *output) = 0;
    virtual void backward(const float *output_d, float *input_d) = 0;

    float* get_input() const { this->input; } 
    float* get_output() const { this->output; }
};

#endif
