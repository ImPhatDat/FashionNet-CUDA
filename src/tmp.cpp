#include <iostream>
#include <string>
#include <random>
#include <cstring> // For memset
#include "dense.h"

// Example usage
int main() {
    int input_size = 128;
    int output_size = 64;
    int batch_size = 32;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a Dense layer
    Dense dense(input_size, output_size, batch_size, gen);

    // Input and output arrays
    float* input = new float[batch_size * input_size];
    float* output = new float[batch_size * output_size];

    // Fill input with random values (optional)
    initialize_1d_array(input, batch_size, input_size, gen);

    // Perform forward pass
    dense.forward(input, output, "relu");

    // Print the parameters
    dense.print_params();

    // Clean up
    delete[] input;
    delete[] output;

    return 0;
}
