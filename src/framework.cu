#include "framework.h"

void relu(float *input, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        int rowStart = i * cols;
        for (int j = 0; j < cols; ++j)
        {
            input[rowStart + j] = std::max(0.0f, input[rowStart + j]);
        }
    }
}

void softmax(float *input, int rows, int cols)
{
    float *expValues = new float[cols];
    for (int i = 0; i < rows; ++i)
    {
        // Find max value for numerical stability
        float maxVal = input[i * cols];
        for (int j = 1; j < cols; ++j)
        {
            maxVal = std::max(maxVal, input[i * cols + j]);
        }

        // Compute exponentials and sum
        float expSum = 0.0f;

        for (int j = 0; j < cols; ++j)
        {
            int index = i * cols + j;
            // Subtract max for numerical stability
            expValues[j] = std::exp(input[index] - maxVal);
            expSum += expValues[j];
        }

        // Normalize to get probabilities
        for (int j = 0; j < cols; ++j)
        {
            int index = i * cols + j;
            input[index] = expValues[j] / expSum;
        }
    }
    delete[] expValues;
}

void matmul(const float *A, const float *B, float *C, int M, int K, int N)
{
    // Matrix multiplication: C[M x N] = A[M x K] * B[K x N]
    for (int i = 0; i < M; ++i)       
        for (int j = 0; j < N; ++j)   
        {
            float sum = 0; // Initialize output
            for (int k = 0; k < K; ++k) // Shared dimension
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
}

// assume sum_over_batch
float categorical_crossentropy_loss(int *y_true, float *y_pred, int batch_size, int num_classes)
{
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; ++i)
    {
        int true_class = y_true[i];
        float predicted_prob = y_pred[i * num_classes + true_class];

        // Avoid log(0) by clamping probabilities to a small positive value
        const float epsilon = 1e-7f;
        predicted_prob = std::max(predicted_prob, epsilon);
        total_loss -= std::log(predicted_prob);
    }

    return total_loss / batch_size;
}