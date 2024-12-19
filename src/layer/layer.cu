#include "layer.h"

Layer::Layer() : input(nullptr), output(nullptr), batch_size(0), input_size(0), output_size(0) {}
Layer::Layer(int batch_size, int input_size, int output_size) {
    this->batch_size = batch_size;
    this->input_size = input_size;

    input = new float[batch_size * input_size];
    output = new float[batch_size * output_size];
}

Layer::~Layer()
{
    delete[] input;
    delete[] output;
}
