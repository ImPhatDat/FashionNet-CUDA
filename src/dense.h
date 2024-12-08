#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include "framework.h"

void initialize_1d_array(float* array, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<float> dis(-1.0, 1.0); // Uniform distribution
    for (int i = 0; i < rows * cols; ++i) {
        array[i] = dis(gen); // Random value between -1 and 1
    }
}

// Initialize weights and biases for multiple dense layers
void initialize_dense_layers(
    int num_layers,
    int BATCH_SIZE,
    const int* output_sizes,
    float** weights,
    float** biases,
    std::mt19937& gen
) {
    for (int i = 0; i < num_layers; ++i) {
        int output_size = output_sizes[i];

        // Allocate memory for weights and initialize
        weights[i] = new float[BATCH_SIZE * output_size];
        initialize_1d_array(weights[i], BATCH_SIZE, output_size, gen);

        // Allocate memory for biases and initialize (biases are 1D with length = output_size)
        biases[i] = new float[output_size];
        for (int j = 0; j < output_size; ++j) {
            biases[i][j] = 0.0f; // Initialize biases to zero
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

class Dense {
private:
    int input_size;
    int output_size;
    int batch_size;

    float* weights; // 1D array to represent weights (row-major)
    float* biases;  // 1D array to represent biases

public:
    // Constructor
    Dense(int input_size, int output_size, int batch_size, std::mt19937& gen) 
        : input_size(input_size), output_size(output_size), batch_size(batch_size) {
        // Allocate and initialize weights and biases
        weights = new float[input_size * output_size];
        biases = new float[output_size];

        initialize_1d_array(weights, input_size, output_size, gen); // Initialize weights
        std::memset(biases, 0, sizeof(float) * output_size);       // Initialize biases to zero
    }

    // Destructor
    ~Dense() {
        delete[] weights;
        delete[] biases;
    }

    // Forward pass
    void forward(const float* input, float* output, const std::string& activation_type = "none") const {
        // Perform matrix multiplication and bias addition
        matmul(input, weights, output, batch_size, input_size, output_size);

        // Add biases
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                output[i * output_size + j] += biases[j];
            }
        }

        // Apply activation function if specified
        if (activation_type == "relu") {
            relu(output, batch_size, output_size);
        } else if (activation_type == "softmax") {
            softmax(output, batch_size, output_size);
        } else if (activation_type != "none") {
            std::cerr << "Error: Unsupported activation type \"" << activation_type << "\". Supported types are: relu, softmax, none.\n";
        }
    }

    const int get_batch_size() const { return batch_size; }

    const int get_output_size() const { return output_size; }

    const float* get_weights() const { return weights; }

    const float* get_biases() const { return biases; }

    // Utility for debugging: Print weights and biases
    void print_params() const {
        std::cout << "Weights (row-major):\n";
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                std::cout << weights[i * output_size + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Biases:\n";
        for (int j = 0; j < output_size; ++j) {
            std::cout << biases[j] << " ";
        }
        std::cout << "\n";
    }
};

void model_forward(const float* input, int input_size, float* output, Dense layers[], int num_dense) {
    // Allocate a temporary array for intermediate results
    float* x = new float[layers[0].get_batch_size() * layers[0].get_output_size()];
    layers[0].forward(input, x);

    for (int i = 1; i < num_dense; ++i) {
        int batch_size = layers[i].get_batch_size();
        int output_size = layers[i].get_output_size();

        float* tmp_x = new float[batch_size * output_size];
        layers[i].forward(x, tmp_x);

        delete[] x;
        x = tmp_x;
    }

    // Copy the final result to the output array
    std::memcpy(output, x, sizeof(float) * layers[num_dense - 1].get_batch_size() * layers[num_dense - 1].get_output_size());

    // Free memory
    delete[] x;
}
