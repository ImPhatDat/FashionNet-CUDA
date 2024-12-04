#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>

// Model configurations
const int DENSE_OUTPUT[] = {128, 128, 10};
const std::string ACTIVATION_TYPES[] = {"relu", "relu", "softmax"};
const int BATCH_SIZE = 64;


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


void dense_forward(const float* input, float* output, const float* weights, const float* biases, 
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

void model_forward(const float* input, int input_size, float* output, const float* weights[], const float* biases[], 
           int output_sizes[], int num_dense, int batch_size, std::string activation_types[]) {
    
    float* x = new float[batch_size * output_sizes[0]];
    dense_forward(input, x, weights[0], biases[0], input_size, output_sizes[0], batch_size, activation_types[0]);

    for (int i = 1; i < num_dense; i++) {
        float* tmp_x = new float[batch_size * output_sizes[i]];
        dense_forward(x, tmp_x, weights[i], biases[i], output_sizes[i - 1], output_sizes[i], batch_size, activation_types[i]);

        delete[] x;
        x = tmp_x;
    }
    std::memcpy(output, x, sizeof(x) * batch_size * output_sizes[num_dense - 1]);
}

// assume sum_over_batch
float categorical_crossentropy_loss(int* y_true, float* y_pred, int batch_size, int num_classes) {
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        int true_class = y_true[i];
        float predicted_prob = y_pred[i * num_classes + true_class];
        
        // Avoid log(0) by clamping probabilities to a small positive value
        const float epsilon = 1e-7f;
        predicted_prob = std::max(predicted_prob, epsilon);

        total_loss -= std::log(predicted_prob);
    }

    return total_loss / batch_size;
}
