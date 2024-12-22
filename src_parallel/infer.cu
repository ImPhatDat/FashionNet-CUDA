#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <getopt.h>
#include "Model/Model.hh"
#include "layer/dense.hh"
#include "layer/relu.hh"
#include "layer/softmax.hh"
#include "helpers.hh"

const int INPUT_SIZE = 784; // Example: MNIST image input size
const int OUTPUT_SIZE = 10;
const int BATCH_SIZE = 1; // Single image for inference

int main(int argc, char **argv) {
    std::string checkpoint_path = "";
    std::string img_path = "";

    int opt;

    // Parsing command-line arguments
    while ((opt = getopt(argc, argv, "p:i:")) != -1)
    {
        switch (opt)
        {
        case 'p':
            checkpoint_path = optarg; // Store the checkpoint path
            break;
        case 'i':
            img_path = optarg; // Store the checkpoint path
            break;
        default:
            fprintf(stderr, "Usage: %s [-p checkpoint_path] [-i image_path]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    // Load a random PNG file
    std::ifstream file(img_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open image file." << std::endl;
        return -1;
    }

    // Read the image data
    uint8_t img[INPUT_SIZE];
    file.read(reinterpret_cast<char *>(img), INPUT_SIZE);
    file.close();

    // Normalize the image to [0, 1]
    float *input_image = new float[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; ++i) {
        input_image[i] = img[i] / 255.0f;
    }

    // Define the model layers
    std::mt19937 global_rng(1); // Random number generator
    Layer *layers[] = {
        new Dense(BATCH_SIZE, INPUT_SIZE, 128, global_rng),
        new ReLU(BATCH_SIZE, 128),
        new Dense(BATCH_SIZE, 128, 128, global_rng),
        new ReLU(BATCH_SIZE, 128),
        new Dense(BATCH_SIZE, 128, OUTPUT_SIZE, global_rng),
        new Softmax(BATCH_SIZE, OUTPUT_SIZE)
    };
    const int NUM_LAYERS = sizeof(layers) / sizeof(layers[0]);

    // Create the model
    Model model(layers, NUM_LAYERS, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);

    // Load the model weights
    model.load_weights(checkpoint_path);

    // Predict using your model
    float *output = new float[OUTPUT_SIZE];
    model.forward(input_image, output);

    // Find the class with the highest score
    int predicted_class = std::max_element(output, output + OUTPUT_SIZE) - output;
    std::cout << "Predicted class: " << predicted_class << std::endl;
    std::cout << "Class probabilities:" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << "Class " << i << ": " << output[i] * 100.0f << "%" << std::endl;
    }
    // Free memory
    delete[] input_image;
    delete[] output;

    return 0;
}