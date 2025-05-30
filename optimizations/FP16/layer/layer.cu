#include "layer.hh"

Layer::Layer(int batch_size, int input_size, int output_size) {
    this->batch_size = batch_size;
    this->input_size = input_size;
    this->output_size = output_size;

    CHECK(cudaMalloc(&input, sizeof(__half) * batch_size * input_size));
    CHECK(cudaMalloc(&output, sizeof(__half) * batch_size * output_size));
}

Layer::~Layer()
{
    CHECK(cudaFree(input));
    CHECK(cudaFree(output));
}

float* Layer::get_weights() const { return nullptr; }
float* Layer::get_biases() const { return nullptr; }

void Layer::update_weights(const __half learning_rate, dim3 blockSize) {}

void Layer::load_weights(const float* weights, const float* biases) {}
