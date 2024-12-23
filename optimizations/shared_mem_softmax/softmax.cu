#include "../src_parallel/layer/softmax.hh"

Softmax::Softmax(int batch_size, int input_size)
    : Layer(batch_size, input_size, input_size)
{
    this->name = "softmax";
}

// Optimized kernel for forward pass using shared memory
__global__ void softmax_forward_kernel(const float *input, float *output, int batch_size, int input_size) {
    extern __shared__ float shared_mem[];
    float *max_vals = shared_mem;
    float *sum_exp = &shared_mem[blockDim.x];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Each block handles one batch
    if (bid >= batch_size) return;
    
    // Initialize max value for this thread
    float local_max = -INFINITY;
    
    // Each thread finds its local max across strided elements
    for (int i = tid; i < input_size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[bid * input_size + i]);
    }
    
    // Reduce to find max value for the batch
    max_vals[tid] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
        }
        __syncthreads();
    }
    
    float batch_max = max_vals[0];
    
    // Compute local sum of exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x) {
        float exp_val = expf(input[bid * input_size + i] - batch_max);
        output[bid * input_size + i] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduce to find total sum
    sum_exp[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_exp[tid] += sum_exp[tid + stride];
        }
        __syncthreads();
    }
    
    float batch_sum = sum_exp[0];
    
    // Normalize with the batch sum
    for (int i = tid; i < input_size; i += blockDim.x) {
        output[bid * input_size + i] /= batch_sum;
    }
}

// Optimized kernel for backward pass using shared memory
__global__ void softmax_backward_kernel(const float *output, const float *output_d, float *input_d, 
                                      int batch_size, int input_size) {
    extern __shared__ float shared_mem[];
    float *shared_output = shared_mem;
    float *shared_grad = &shared_mem[input_size];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    // Load output and output_d into shared memory
    for (int i = tid; i < input_size; i += blockDim.x) {
        shared_output[i] = output[bid * input_size + i];
        shared_grad[i] = output_d[bid * input_size + i];
    }
    __syncthreads();
    
    // Compute dot product between output and output_d
    float dot_prod = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x) {
        dot_prod += shared_output[i] * shared_grad[i];
    }
    
    // Reduce to get final dot product
    atomicAdd(&shared_mem[0], dot_prod);
    __syncthreads();
    
    // Calculate gradient for each element
    for (int i = tid; i < input_size; i += blockDim.x) {
        input_d[bid * input_size + i] = shared_output[i] * 
            (shared_grad[i] - shared_mem[0]);
    }
}

void Softmax::forward(const float *input, float *output, dim3 blockSize) {
    // One block per batch item, with shared memory for reduction
    dim3 gridSize(batch_size);
    size_t shared_mem_size = 2 * blockSize.x * sizeof(float);
    
    softmax_forward_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        input, output, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(this->output, output, batch_size * input_size * sizeof(float), 
                    cudaMemcpyDeviceToDevice));
}

void Softmax::backward(const float *output_d, float *input_d, dim3 blockSize) {
    // One block per batch item, with shared memory for output and gradients
    dim3 gridSize(batch_size);
    size_t shared_mem_size = (2 * input_size + 1) * sizeof(float);
    
    softmax_backward_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        this->output, output_d, input_d, batch_size, input_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}