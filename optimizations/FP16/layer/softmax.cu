#include "softmax.hh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
Softmax::Softmax(int batch_size, int input_size)
    : Layer(batch_size, input_size, input_size)
{
    this->name = "softmax";
}

// Device kernel for forward pass
__global__ void softmax_forward_kernel(const __half *input, __half *output, int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / input_size;
    int c = idx % input_size;

    if (b >= batch_size || c >= input_size) return;

    // Compute max for numerical stability
    __half max_val = input[b * input_size];
    for (int i = 1; i < input_size; ++i) {
        max_val = __hmax(max_val, input[b * input_size + i]);
    }

    // Compute exponentials and their sum
    __half sum_exp = __float2half(0.0f);
    __half tmp;
    for (int i = 0; i < input_size; ++i) 
    {
        float exp_val = expf(__half2float(__hsub(input[b * input_size + i], max_val))); // Convert to float, exponentiate, convert back to half
        tmp = __float2half(exp_val);
        output[b * input_size + i] = tmp;
        sum_exp = __hadd(sum_exp, tmp);
    }

    // Normalize
    output[b * input_size + c] = __hdiv(output[b * input_size + c], sum_exp);
}

// Device kernel for backward pass
__global__ void softmax_backward_kernel(const __half *output, const __half *output_d, __half *input_d, int batch_size, int input_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / input_size;
    int i = idx % input_size;

    if (b >= batch_size || i >= input_size) return;

    __half grad = __float2half(0.0f);
    for (int j = 0; j < input_size; ++j) {
        __half delta = (i == j) ? __float2half(1.0f) : __float2half(0.0f);
        grad = __hadd(grad, __hmul(output_d[b * input_size + j], __hmul(output[b * input_size + j], __hsub(delta, output[b * input_size + i]))));
    }
    input_d[b * input_size + i] = grad;
}

void Softmax::forward(const __half *input, __half *output, dim3 blockSize) {
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    softmax_forward_kernel<<<gridSize, blockSize>>>(input, output, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(this->output, output, batch_size * input_size * sizeof(__half), cudaMemcpyDeviceToDevice));
}

void Softmax::backward(const __half *output_d, __half *input_d, dim3 blockSize) {
    dim3 gridSize((batch_size * input_size + blockSize.x - 1) / blockSize.x);
    softmax_backward_kernel<<<gridSize, blockSize>>>(this->output, output_d, input_d, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
