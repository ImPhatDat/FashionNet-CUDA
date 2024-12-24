#include "dense.hh"

__global__ void initialize_weights_kernel(__half *weights, int rows, int cols, float limit, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx >= total) return;
    curandState state;
    curand_init(seed, idx, 0, &state);

    // random in range [-limit, +limit]
    weights[idx] = __float2half(curand_uniform(&state) * 2.0f * limit - limit);
}

// Glorot Uniform initialization
void initialize_dense(__half *d_weights, __half *d_biases, int rows, int cols, dim3 blockSize, unsigned long seed) {
    // Glorot Uniform limit
    float limit = std::sqrt(6.0f / (rows + cols));

    int total_weights = rows * cols;
    dim3 gridSizeWeights((total_weights + blockSize.x - 1) / blockSize.x);
    initialize_weights_kernel<<<gridSizeWeights, blockSize>>>(d_weights, rows, cols, limit, seed);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Initialize biases to 0
    CHECK(cudaMemset(d_biases, 0, cols * sizeof(__half)));
}

__global__ void matmul_kernel(const __half *A, const __half *B, __half *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        __half sum = __float2half(0.0f);
        for (int k = 0; k < K; ++k) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}

void matmul(const __half *A, const __half *B, __half *C, int M, int K, int N, dim3 blockSize) {
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, M, K, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

__global__ void transpose_kernel(const __half *in, __half *out, int M, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N)
    {
        out[j * M + i] = in[i * N + j];
    }
}

void transpose(const __half *in, __half *out, int M, int N, dim3 blockSize)
{
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (
        M + blockSize.y - 1) / blockSize.y);
    transpose_kernel<<<gridSize, blockSize>>>(in, out, M, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}


Dense::Dense(int batch_size, int input_size, int output_size, dim3 blockSize, bool init, unsigned long seed) : Layer(batch_size, input_size, output_size)
{
    this->name = "dense";
    CHECK(cudaMalloc(&weights, sizeof(__half) * input_size * output_size));
    CHECK(cudaMalloc(&biases, sizeof(__half) * output_size));

    CHECK(cudaMalloc(&grad_weights, sizeof(__half) * input_size * output_size));
    CHECK(cudaMalloc(&grad_biases, sizeof(__half) * output_size));

    if (init)
        initialize_dense(weights, biases, input_size, output_size, blockSize, seed);
}

Dense::~Dense()
{
    CHECK(cudaFree(weights));
    CHECK(cudaFree(biases));
    CHECK(cudaFree(grad_weights));
    CHECK(cudaFree(grad_biases));
}

__global__ void add_bias_kernel(__half *output, const __half *biases, int batch_size, int output_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size)
    {
        output[row * output_size + col] = __hadd(output[row * output_size + col], biases[col]);
    }
}

void Dense::forward(const __half *input, __half *output, dim3 blockSize)
{
    CHECK(cudaMemcpy(this->input, input, sizeof(__half) * this->batch_size * this->input_size, cudaMemcpyDeviceToDevice));

    matmul(input, weights, output, this->batch_size, this->input_size, this->output_size, blockSize);

    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
    add_bias_kernel<<<gridSize, blockSize>>>(output, biases, batch_size, output_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}



__global__ void grad_biases_kernel(const __half *output_d, __half *grad_biases, int batch_size, int output_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < output_size) {
        __half sum = __float2half(0.0f);
        for (int row = 0; row < batch_size; ++row) {
            sum = __hadd(sum, output_d[row * output_size + col]);
        }
        grad_biases[col] = sum;
    }
}

// Backward pass
void Dense::backward(const __half *output_d, __half *input_d, dim3 blockSize)
{
    // Initialize gradients to zero
    CHECK(cudaMemset(grad_weights, 0, sizeof(__half) * input_size * output_size));
    CHECK(cudaMemset(grad_biases, 0, sizeof(__half) * output_size));

    __half* input_T;
    CHECK(cudaMalloc(&input_T, sizeof(__half) * this->batch_size * this->input_size));
    // pre-transpose for backprop
    transpose(this->input, input_T, this->batch_size, this->input_size, blockSize);
    // Compute grad_weights: input^T * output_d
    matmul(input_T, output_d, grad_weights, input_size, batch_size, output_size, blockSize);
    CHECK(cudaFree(input_T));
    
    // Compute grad_biases: sum over batch_size
    __half *d_output_d_sum;
    CHECK(cudaMalloc(&d_output_d_sum, sizeof(__half) * output_size));
    CHECK(cudaMemset(d_output_d_sum, 0, sizeof(__half) * output_size));
    
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x);
    grad_biases_kernel<<<gridSize, blockSize>>>(output_d, grad_biases, batch_size, output_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Compute input_d: output_d * weights^T
    __half *d_weights_transpose;
    CHECK(cudaMalloc(&d_weights_transpose, sizeof(__half) * this->input_size * this->output_size));
    transpose(this->weights, d_weights_transpose, this->input_size, this->output_size, blockSize);
    matmul(output_d, d_weights_transpose, input_d, batch_size, output_size, input_size, blockSize);
    CHECK(cudaFree(d_weights_transpose));
}

__global__ void update_with_gradient(__half *weights, const __half *grad_weights, __half learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] = __hsub(weights[idx], __hmul(learning_rate, grad_weights[idx]));
    }
}

void Dense::update_weights(__half learning_rate, dim3 blockSize) {
    int total_weights = input_size * output_size;
    dim3 gridSizeWeights((total_weights + blockSize.x - 1) / blockSize.x);
    update_with_gradient<<<gridSizeWeights, blockSize>>>(weights, grad_weights, learning_rate, total_weights);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int total_biases = output_size;
    dim3 gridSizeBiases((total_biases + blockSize.x - 1) / blockSize.x);
    update_with_gradient<<<gridSizeBiases, blockSize>>>(biases, grad_biases, learning_rate, total_biases);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

void Dense::load_weights(const float* weights, const float* biases) {
    if (weights != nullptr && biases != nullptr) {
        __half* host_weights = new __half[this->input_size * this->output_size];
        for (int i = 0; i < this->input_size * this->output_size; i++) {
            host_weights[i] = __float2half(weights[i]);
        }

        __half* host_biases = new __half[this->output_size];
        for (int i = 0; i < this->output_size; i++) {
            host_biases[i] = __float2half(biases[i]);
        }

        CHECK(cudaMemcpy(this->weights, host_weights, sizeof(__half) * this->input_size * this->output_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(this->biases, host_biases, sizeof(__half) * this->output_size, cudaMemcpyHostToDevice));

        delete[] host_weights;
        delete[] host_biases;
    }
    else {
        std::cerr << "Can't load weights with nullptr" << std::endl;
    }
}

float* Dense::get_weights() const {
    __half* h_weights = new __half[this->input_size * this->output_size];
    CHECK(cudaMemcpy(h_weights, this->weights, sizeof(__half) * this->input_size * this->output_size, cudaMemcpyDeviceToHost));
    float* h_float_weights = new float[this->input_size * this->output_size];
    for (int i = 0; i < this->input_size * this->output_size; i++) {
        h_float_weights[i] = __half2float(h_weights[i]);
    }
    delete[] h_weights;
    return h_float_weights;
}

float* Dense::get_biases() const {
    __half* h_biases = new __half[this->output_size];
    CHECK(cudaMemcpy(h_biases, this->biases, sizeof(__half) * this->output_size, cudaMemcpyDeviceToHost));
    float* h_float_biases = new float[this->output_size];
    for (int i = 0; i < this->output_size; i++) {
        h_float_biases[i] = __half2float(h_biases[i]);
    }
    delete[] h_biases;
    return h_float_biases;
}