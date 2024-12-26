#include "relu.hh"

ReLU::ReLU(int batch_size, int input_size)
    : Layer(batch_size, input_size, input_size) // Reuse parent constructor
{
    this->name = "relu";
    this->total_size = batch_size * input_size;
}

// Device kernel for ReLU forward pass
__global__ void relu_forward_kernel(const float *input, float *output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    output[idx] = fmaxf(0.0f, input[idx]);
}

// Device kernel for ReLU backward pass
__global__ void relu_backward_kernel(const float *input, const float *output_d, float *input_d, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    input_d[idx] = output_d[idx] * (input[idx] > 0);
}

void ReLU::forward(const float *input, float *output, dim3 blockSize) {
    CHECK(cudaMemcpy(this->input, input, sizeof(float) * total_size, cudaMemcpyDeviceToDevice));
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    relu_forward_kernel<<<gridSize, blockSize>>>(input, output, total_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

void ReLU::backward(const float *output_d, float *input_d, dim3 blockSize) {
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    relu_backward_kernel<<<gridSize, blockSize>>>(this->input, output_d, input_d, total_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}