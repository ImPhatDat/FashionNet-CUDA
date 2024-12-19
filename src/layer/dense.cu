#include "dense.h"

void matmul(const float *A, const float *B, float *C, int M, int K, int N)
{
    // Matrix multiplication: C[M x N] = A[M x K] * B[K x N]
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;              // Initialize output
            for (int k = 0; k < K; ++k) // Shared dimension
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
}

void initialize_dense(float *weights, float *biases, int rows, int cols, std::mt19937 &gen)
{
    std::uniform_real_distribution<float> dis(-1.0, 1.0); // Uniform distribution
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i * cols + j] = dis(gen); // Random value between -1 and 1
        }
    }

    for (int j = 0; j < cols; ++j)
    {
        biases[j] = 0; // Set biases to 0
    }
}

// Default Constructor
Dense::Dense()
{
    this->weights = nullptr;
    this->biases = nullptr;
    this->grad_weights = nullptr;
    this->grad_biases = nullptr;
}

Dense::Dense(int batch_size, int input_size, int output_size, std::mt19937 &gen)
{
    // Allocate and initialize weights and biases
    weights = new float[input_size * output_size];
    biases = new float[output_size];

    grad_weights = new float[input_size * output_size];
    grad_biases = new float[output_size];

    initialize_dense(weights, biases, input_size, output_size, gen); // Initialize weights
}

Dense::~Dense()
{
    delete[] weights;
    delete[] biases;
    delete[] grad_weights;
    delete[] grad_biases;
}

// Forward pass
void Dense::forward(const float *input, float *output)
{
    matmul(input, weights, output, this->batch_size, this->input_size, this->output_size);
    for (int i = 0; i < this->batch_size; ++i)
    {
        for (int j = 0; j < this->output_size; ++j)
        {
            output[i * output_size + j] += this->biases[j];
        }
    }
}

// Backward pass
void Dense::backward(const float *output_d, float *input_d)
{
    // Compute grad_weights: input^T * output_d
    matmul(input_d, output_d, grad_weights, input_size, batch_size, output_size);

    // Compute grad_biases: sum over batch_size
    std::fill(grad_biases, grad_biases + output_size, 0.0f);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int o = 0; o < output_size; ++o)
        {
            grad_biases[o] += output_d[b * output_size + o];
        }
    }

    // Compute input_d: output_d * weights^T
    matmul(output_d, weights, input_d, batch_size, output_size, input_size);
}

void Dense::update_weights_and_biases(float learning_rate)
{
    // Update weights
    for (int i = 0; i < input_size * output_size; ++i)
    {
        weights[i] -= learning_rate * grad_weights[i];
    }

    // Update biases
    for (int i = 0; i < output_size; ++i)
    {
        biases[i] -= learning_rate * grad_biases[i];
    }
}
