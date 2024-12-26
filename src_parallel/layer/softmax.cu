#include "softmax.hh"

Softmax::Softmax(int batch_size, int input_size)
    : Layer(batch_size, input_size, input_size)
{
    this->name = "softmax";
}

// Device kernel for forward pass
__global__ void softmax_forward_kernel(const float *input, float *output, int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / input_size;
    int c = idx % input_size;

    if (b >= batch_size || c >= input_size) return;

    // Compute max for numerical stability
    float max_val = input[b * input_size + 0];
    for (int i = 1; i < input_size; ++i) {
        max_val = fmaxf(max_val, input[b * input_size + i]);
    }

    // Compute exponentials and their sum
    float sum_exp = 0.0f;
    float tmp;
    for (int i = 0; i < input_size; ++i) 
    {
        tmp = expf(input[b * input_size + i] - max_val);
        output[b * input_size + i] = tmp;
        sum_exp += tmp;
    }

    // Normalize
    output[b * input_size + c] /= sum_exp;
}

// Device kernel for backward pass
__global__ void softmax_backward_kernel(const float *output, const float *output_d, float *input_d, int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / input_size;
    int i = idx % input_size;

    if (b >= batch_size || i >= input_size) return;

    float grad = 0.0f;
    for (int j = 0; j < input_size; ++j) {
        grad += output_d[b * input_size + j] * output[b * input_size + j] * ((i == j) - output[b * input_size + i]);
    }
    input_d[b * input_size + i] = grad;
}

void Softmax::forward(const float *input, float *output, dim3 blockSize) {
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    softmax_forward_kernel<<<gridSize, blockSize>>>(input, output, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(this->output, output, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Softmax::backward(const float *output_d, float *input_d, dim3 blockSize) {
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    softmax_backward_kernel<<<gridSize, blockSize>>>(this->output, output_d, input_d, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
