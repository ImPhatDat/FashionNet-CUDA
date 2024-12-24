#include "../src_parallel/layer/softmax.hh"

Softmax::Softmax(int batch_size, int input_size) 
    : Layer(batch_size, input_size, input_size) {
    this->name = "softmax";
}

__global__ void softmax_forward_kernel(const float* input, float* output, int batch_size, int input_size) {
    extern __shared__ float shared_data[];
    float* max_shared = shared_data;
    float* sum_shared = &shared_data[blockDim.x];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    // Find max value
    float local_max = -INFINITY;
    for (int i = tid; i < input_size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[bid * input_size + i]);
    }
    max_shared[tid] = local_max;
    __syncthreads();
    
    // Parallel reduction for max
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = max_shared[0];
    
    // Compute exponentials and sum
    float local_sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x) {
        float exp_val = expf(input[bid * input_size + i] - max_val);
        output[bid * input_size + i] = exp_val;
        local_sum += exp_val;
    }
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    // Parallel reduction for sum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Normalize
    for (int i = tid; i < input_size; i += blockDim.x) {
        output[bid * input_size + i] /= sum_shared[0];
    }
}

__global__ void softmax_backward_kernel(const float* output, const float* output_d, float* input_d, 
                                      int batch_size, int input_size) {
    extern __shared__ float shared_data[];
    float* shared_out = shared_data;
    float* shared_grad = &shared_data[input_size];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    // Load to shared memory
    for (int i = tid; i < input_size; i += blockDim.x) {
        shared_out[i] = output[bid * input_size + i];
        shared_grad[i] = output_d[bid * input_size + i];
    }
    __syncthreads();
    
    for (int i = tid; i < input_size; i += blockDim.x) {
        float grad = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            grad += shared_grad[j] * shared_out[j] * ((i == j) - shared_out[i]);
        }
        input_d[bid * input_size + i] = grad;
    }
}

void Softmax::forward(const float* input, float* output, dim3 blockSize) {
    dim3 gridSize(batch_size);
    size_t shared_mem_size = 2 * blockSize.x * sizeof(float);
    
    softmax_forward_kernel<<<gridSize, blockSize, shared_mem_size>>>(input, output, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(this->output, output, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Softmax::backward(const float* output_d, float* input_d, dim3 blockSize) {
    dim3 gridSize(batch_size);
    size_t shared_mem_size = (2 * input_size) * sizeof(float);
    
    softmax_backward_kernel<<<gridSize, blockSize, shared_mem_size>>>(this->output, output_d, input_d, 
                                                                     batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}