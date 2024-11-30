#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cmath>

void relu(float* input, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            // ReLU activation: max(0, x)
            input[index] = std::max(0.0f, input[index]);
        }
    }
}

void softmax(float* input, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        // Find max value for numerical stability
        float maxVal = input[i * cols];
        for (int j = 1; j < cols; ++j) {
            maxVal = std::max(maxVal, input[i * cols + j]);
        }

        // Compute exponentials and sum
        float expSum = 0.0f;
        float* expValues = new float[cols];

        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            // Subtract max for numerical stability
            expValues[j] = std::exp(input[index] - maxVal);
            expSum += expValues[j];
        }

        // Normalize to get probabilities
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            input[index] = expValues[j] / expSum;
        }

        delete[] expValues;
    }
}

void matmul(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    // Perform matrix multiplication
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            float sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
    
}


void dense_forward(const float* input, const float* weights, const float* biases, float* output, 
           int input_size, int output_size, int batch_size, std::string activation_type = "none") {
    
    matmul(input, weights, output, batch_size, input_size, output_size);

    if (biases != NULL) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                output[i * output_size + j] += biases[j];
            }
        }
    }

    if (activation_type == "relu") {
        relu(output, batch_size, output_size);
    } else if (activation_type == "softmax") {
        softmax(output, batch_size, output_size);
    } else if (activation_type != "none") {
        std::cerr << "Error: Unsupported activation type \"" << activation_type << "\". Supported types are: relu, softmax, none.\n";
    }
    
}