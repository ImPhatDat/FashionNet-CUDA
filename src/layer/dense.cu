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

void transpose(const float *in, float *out, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            // Transpose by swapping rows and columns
            out[j * M + i] = in[i * N + j];
        }
    }
}

// glorot uniform
void initialize_dense(float *weights, float *biases, int rows, int cols, std::mt19937 &gen)
{
    // Calculate the Glorot Uniform limit
    float limit = std::sqrt(6.0f / (rows + cols)); 

    // Create a uniform distribution between -limit and +limit
    std::uniform_real_distribution<float> dis(-limit, limit);

    // Initialize weights
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i * cols + j] = dis(gen);
        }
    }

    // Initialize biases to 0
    for (int j = 0; j < cols; ++j)
    {
        biases[j] = 0.0f;
    }
}

Dense::Dense(int batch_size, int input_size, int output_size, std::mt19937 &gen) : Layer(batch_size, input_size, output_size)
{
    this->name = "dense";
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
    std::memcpy(this->input, input, sizeof(float) * this->batch_size * this->input_size);
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
    // Initialize gradients to zero
    std::fill(grad_weights, grad_weights + input_size * output_size, 0.0f);
    std::fill(grad_biases, grad_biases + output_size, 0.0f);

    // Compute grad_weights: input^T * output_d
    float *tmp_tranpose = new float[this->batch_size * this->input_size];
    transpose(this->input, tmp_tranpose, this->batch_size, this->input_size);
    matmul(tmp_tranpose, output_d, grad_weights, input_size, batch_size, output_size);
    delete[] tmp_tranpose;

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
    tmp_tranpose = new float[this->input_size * this->output_size];
    transpose(this->weights, tmp_tranpose, this->input_size, this->output_size);
    matmul(output_d, tmp_tranpose, input_d, batch_size, output_size, input_size);
    delete[] tmp_tranpose;
}

void Dense::update_weights(float learning_rate)
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
