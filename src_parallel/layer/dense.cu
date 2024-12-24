#include "dense.hh"

// glorot uniform
void Dense::initialize_dense(float *weights, float *biases, int rows, int cols, std::mt19937 &gen)
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



__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N) {
    // Compute row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure valid thread indices
    if (row < M && col < N) {
        float sum = 0.0f;

        // Perform dot product for the row of A and column of B
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Write result to C
        C[row * N + col] = sum;
    }
}

void Dense::matmul(const float *A, const float *B, float *C, int M, int K, int N, dim3 blockSize) {
    // Calculate grid size
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, M, K, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

__global__ void transpose_kernel(const float *in, float *out, int M, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N)
    {
        // Transpose by swapping rows and columns
        out[j * M + i] = in[i * N + j];
    }
}

void Dense::transpose(const float *in, float *out, int M, int N, dim3 blockSize)
{
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (
        M + blockSize.y - 1) / blockSize.y); // Grid size calculation
    // Launch the kernel
    transpose_kernel<<<gridSize, blockSize>>>(in, out, M, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}


Dense::Dense(int batch_size, int input_size, int output_size, bool init, std::mt19937 &gen) : Layer(batch_size, input_size, output_size)
{
    this->name = "dense";
    // Allocate and initialize weights and biases
    CHECK(cudaMalloc(&weights, sizeof(float) * input_size * output_size));
    CHECK(cudaMalloc(&biases, sizeof(float) * output_size));

    CHECK(cudaMalloc(&grad_weights, sizeof(float) * input_size * output_size));
    CHECK(cudaMalloc(&grad_biases, sizeof(float) * output_size));

    if (init) {
        float* host_weights = new float[input_size * output_size];
        float* host_biases = new float[input_size * output_size];
        initialize_dense(host_weights, host_biases, input_size, output_size, gen); // Initialize weights

        CHECK(cudaMemcpy(weights, host_weights, sizeof(float) * input_size * output_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(biases, host_biases, sizeof(float) * output_size, cudaMemcpyHostToDevice));


        delete[] host_weights;
        delete[] host_biases;
    }
}

Dense::~Dense()
{

    CHECK(cudaFree(weights));
    CHECK(cudaFree(biases));
    CHECK(cudaFree(grad_weights));
    CHECK(cudaFree(grad_biases));
}

__global__ void add_bias_kernel(float *output, const float *biases, int batch_size, int output_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size)
    {
        output[row * output_size + col] += biases[col];
    }
}

// Forward pass
void Dense::forward(const float *input, float *output, dim3 blockSize)
{
    CHECK(cudaMemcpy(this->input, input, sizeof(float) * this->batch_size * this->input_size, cudaMemcpyDeviceToDevice));

    matmul(input, weights, output, this->batch_size, this->input_size, this->output_size, blockSize);

    // Add biases on the CUDA device
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
    add_bias_kernel<<<gridSize, blockSize>>>(output, biases, batch_size, output_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    
}


__global__ void grad_biases_kernel(const float *output_d, float *grad_biases, int batch_size, int output_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < output_size) {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; ++row) {
            sum += output_d[row * output_size + col];
        }
        grad_biases[col] = sum;
    }
}

// Backward pass
void Dense::backward(const float *output_d, float *input_d, dim3 blockSize)
{
    // Initialize gradients to zero
    CHECK(cudaMemset(grad_weights, 0, sizeof(float) * input_size * output_size));
    CHECK(cudaMemset(grad_biases, 0, sizeof(float) * output_size));

    float* input_T;
    CHECK(cudaMalloc(&input_T, sizeof(float) * this->batch_size * this->input_size));
    // pre-transpose for backprop
    transpose(this->input, input_T, this->batch_size, this->input_size, blockSize);
    // Compute grad_weights: input^T * output_d
    matmul(input_T, output_d, grad_weights, input_size, batch_size, output_size, blockSize);
    CHECK(cudaFree(input_T));
    
    // Compute grad_biases: sum over batch_size
    float *d_output_d_sum;
    CHECK(cudaMalloc(&d_output_d_sum, sizeof(float) * output_size));
    CHECK(cudaMemset(d_output_d_sum, 0, sizeof(float) * output_size));
    
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x);
    grad_biases_kernel<<<gridSize, blockSize>>>(output_d, grad_biases, batch_size, output_size);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Compute input_d: output_d * weights^T
    float *d_weights_transpose;
    CHECK(cudaMalloc(&d_weights_transpose, sizeof(float) * this->input_size * this->output_size));
    transpose(this->weights, d_weights_transpose, this->input_size, this->output_size, blockSize);
    matmul(output_d, d_weights_transpose, input_d, batch_size, output_size, input_size, blockSize);
    CHECK(cudaFree(d_weights_transpose));
}

__global__ void update_with_gradient(float *weights, const float *grad_weights, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

void Dense::update_weights(float learning_rate, dim3 blockSize) {
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
        CHECK(cudaMemcpy(this->weights, weights, sizeof(float) * this->input_size * this->output_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(this->biases, biases, sizeof(float) * this->output_size, cudaMemcpyHostToDevice));
    }
    else {
        std::cerr << "Can't load weights with nullptr" << std::endl;
    }
}

float* Dense::get_weights() const {
    float* h_weights = new float[this->input_size * this->output_size];
    CHECK(cudaMemcpy(h_weights, this->weights, sizeof(float) * this->input_size * this->output_size, cudaMemcpyDeviceToHost));
    return h_weights;
}
float* Dense::get_biases() const {
    float* h_biases = new float[this->output_size];
    CHECK(cudaMemcpy(h_biases, this->biases, sizeof(float) * this->output_size, cudaMemcpyDeviceToHost));
    return h_biases;
}