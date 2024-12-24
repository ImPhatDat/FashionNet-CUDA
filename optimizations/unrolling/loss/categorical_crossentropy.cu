#include "../../../src_parallel/loss/categorical_crossentropy.hh"

CategoricalCrossentropy::CategoricalCrossentropy(float epsilon) : epsilon(epsilon) {}

CategoricalCrossentropy::~CategoricalCrossentropy() {}

// Kernel for forward pass
__global__ void forward_kernel(const uint8_t *y_true, const float *y_pred, float *loss, int batch_size, int num_classes, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        const float *pred_row = y_pred + idx * num_classes;
        int label = y_true[idx];

        // Prevent log(0) by adding a small constant
        float pred = fmaxf(pred_row[label], epsilon);
        atomicAdd(loss, -logf(pred));
    }
}

// Forward pass: Computes the loss
float CategoricalCrossentropy::forward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes, dim3 blockSize)
{
    float *d_loss;
    CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    int gridSize = (batch_size + blockSize.x - 1) / blockSize.x;
    forward_kernel<<<gridSize, blockSize>>>(y_true, y_pred, d_loss, batch_size, num_classes, this->epsilon);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float h_loss;
    CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_loss));

    this->loss_val = h_loss / batch_size;
    return this->loss_val;
}

// Kernel for backward pass
__global__ void backward_kernel(const uint8_t *y_true, const float *y_pred, float *gradients, int batch_size, int num_classes, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        const float *pred_row = y_pred + idx * num_classes;
        float *grad_row = gradients + idx * num_classes;
        int label = y_true[idx];

        // Compute gradients
        #pragma unroll
        for (int c = 0; c < num_classes; ++c)
        {
            float pred_round = fmaxf(pred_row[c], epsilon);
            grad_row[c] = (-1.0f / pred_round) * (c == label);
        }
    }
}

// Backward pass: Computes the gradient with respect to predictions
void CategoricalCrossentropy::backward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes, float *gradients, dim3 blockSize)
{
    int gridSize = (batch_size + blockSize.x - 1) / blockSize.x;
    backward_kernel<<<gridSize, blockSize>>>(y_true, y_pred, gradients, batch_size, num_classes, this->epsilon);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}