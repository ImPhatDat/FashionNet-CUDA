#include <iostream>
#include <string>
#include <random>
#include <cstring> // For memset
#include "dense.h"

std::mt19937 global_rng(1); // Random number generator

const int INPUT_SIZE = 784;   // Example: MNIST image input size
const int BATCH_SIZE = 64;
const int OUTPUT_SIZE = 10;

int main() {
    Dense layer1(INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, "softmax", global_rng);

    float* input = new float[INPUT_SIZE * BATCH_SIZE];
    float* output = new float[OUTPUT_SIZE * BATCH_SIZE];

    std::cout << layer1.get_biases()[1] << std::endl;

    // Cleanup
    delete[] input;
    delete[] output;

    return 0;
}