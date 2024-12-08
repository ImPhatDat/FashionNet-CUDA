#include <iostream>
#include <string>
#include <random>
#include <cstring> // For memset
#include "dense.h"

int main() {
    // Random number generator for weight initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    // Network hyperparameters
    const int BATCH_SIZE = 32;
    const int INPUT_SIZE = 784;   // Example: MNIST image input size
    const int HIDDEN_SIZE_1 = 128; // First hidden layer size
    const int HIDDEN_SIZE_2 = 64;  // Second hidden layer size
    const int OUTPUT_SIZE = 10;    // Example: 10-class classification

    // Create dense layers
    Dense layer1(INPUT_SIZE, HIDDEN_SIZE_1, BATCH_SIZE, "relu", gen);
    Dense layer2(HIDDEN_SIZE_1, HIDDEN_SIZE_2, BATCH_SIZE, "relu", gen);
    Dense output_layer(HIDDEN_SIZE_2, OUTPUT_SIZE, BATCH_SIZE, "softmax", gen);

    // Array of layers for model_forward function
    Dense layers[] = {layer1, layer2, output_layer};
    int num_dense_layers = sizeof(layers) / sizeof(layers[0]);

    // Allocate input and output arrays
    float* input = new float[BATCH_SIZE * INPUT_SIZE];
    float* output = new float[BATCH_SIZE * OUTPUT_SIZE];

    // Initialize input (in a real scenario, this would be your training/test data)
    for (int i = 0; i < BATCH_SIZE * INPUT_SIZE; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform forward pass
    std::cout << output[100] << std::endl;
    model_forward(input, INPUT_SIZE, output, layers, num_dense_layers);
    std::cout << output[100] << std::endl;


    // Optional: Print layer details
    // std::cout << "Layer 1 details:\n";
    // layer1.print_params();
    // std::cout << "\nLayer 2 details:\n";
    // layer2.print_params();
    // std::cout << "\nOutput layer details:\n";
    // output_layer.print_params();

    // Cleanup
    delete[] input;
    delete[] output;

    return 0;
}