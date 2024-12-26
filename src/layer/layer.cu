#include "layer.hh"

Layer::Layer(int batch_size, int input_size, int output_size) {
    this->batch_size = batch_size;
    this->input_size = input_size;
    this->output_size = output_size;

    input = new float[batch_size * input_size];
    output = new float[batch_size * output_size];
}

Layer::~Layer()
{
    delete[] input;
    delete[] output;
}

float* Layer::get_weights() const { return nullptr; }
float* Layer::get_biases() const { return nullptr; }

void Layer::update_weights(const float learning_rate) {}

void Layer::load_weights(const float* weights, const float* biases) {}
