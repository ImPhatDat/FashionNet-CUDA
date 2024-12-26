#include "relu.hh"
#include <cuda_fp16.h>

ReLU::ReLU(int batch_size, int input_size)
    : Layer(batch_size, input_size, input_size) // Reuse parent constructor
{
    this->name = "relu";
    this->total_size = batch_size * input_size;
}

// Device kernel for ReLU forward pass
__global__ void relu_forward_kernel(const __half *input, __half *output, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    output[idx] = __hmax(__float2half(0.0f), input[idx]);
}

// Device kernel for ReLU backward pass
__global__ void relu_backward_kernel(const __half *input, const __half *output_d, __half *input_d, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    input_d[idx] = __hmul(output_d[idx], __hgt(input[idx], __float2half(0.0f)));
}

void ReLU::forward(const __half *input, __half *output, dim3 blockSize) {
    CHECK(cudaMemcpy(this->input, input, sizeof(__half) * total_size, cudaMemcpyDeviceToDevice));
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    relu_forward_kernel<<<gridSize, blockSize>>>(input, output, total_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

void ReLU::backward(const __half *output_d, __half *input_d, dim3 blockSize) {
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    relu_backward_kernel<<<gridSize, blockSize>>>(this->input, output_d, input_d, total_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
