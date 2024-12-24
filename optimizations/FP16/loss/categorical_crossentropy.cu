#include "categorical_crossentropy.hh"

CategoricalCrossentropy::CategoricalCrossentropy(__half epsilon) : epsilon(epsilon) {}

CategoricalCrossentropy::~CategoricalCrossentropy() {}

// Kernel for forward pass
__global__ void forward_kernel(const uint8_t *y_true, const __half *y_pred, float *loss, int batch_size, int num_classes, __half epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        const __half *pred_row = y_pred + idx * num_classes;
        int label = y_true[idx];

        // Prevent log(0) by adding a small constant
        __half pred = __hmax(pred_row[label], epsilon);  // Using __hmax for __half type
        atomicAdd(loss, -logf(__half2float(pred)));  // __log2 for half-precision, equivalent to log for half
    }
}

// Forward pass: Computes the loss
float CategoricalCrossentropy::forward(const uint8_t *y_true, const __half *y_pred, int batch_size, int num_classes, dim3 blockSize)
{
    float *d_loss;
    CHECK(cudaMalloc(&d_loss, sizeof(float)));  // Allocating memory for half-precision loss
    CHECK(cudaMemset(d_loss, 0, sizeof(float)));  // Initialize the loss to 0

    int gridSize = (batch_size + blockSize.x - 1) / blockSize.x;
    forward_kernel<<<gridSize, blockSize>>>(y_true, y_pred, d_loss, batch_size, num_classes, this->epsilon);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float h_loss;
    CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));  // Copy loss to host
    CHECK(cudaFree(d_loss));

    this->loss_val = h_loss / batch_size;  // Normalize the loss
    return this->loss_val;
}

// Kernel for backward pass
__global__ void backward_kernel(const uint8_t *y_true, const __half *y_pred, __half *gradients, int batch_size, int num_classes, __half epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        const __half *pred_row = y_pred + idx * num_classes;
        __half *grad_row = gradients + idx * num_classes;
        int label = y_true[idx];

        // Compute gradients
        for (int c = 0; c < num_classes; ++c)
        {
            __half pred_round = __hmax(pred_row[c], epsilon);  // Prevent underflow for small predictions
            __half delta = (c == label) ? __float2half(1.0f) : __float2half(0.0f);
            grad_row[c] = __hmul(__hneg(__hdiv(__float2half(1.0f), pred_round)), delta);
        }
    }
}

// Backward pass: Computes the gradient with respect to predictions
void CategoricalCrossentropy::backward(const uint8_t *y_true, const __half *y_pred, int batch_size, int num_classes, __half *gradients, dim3 blockSize)
{
    int gridSize = (batch_size + blockSize.x - 1) / blockSize.x;
    backward_kernel<<<gridSize, blockSize>>>(y_true, y_pred, gradients, batch_size, num_classes, this->epsilon);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}
