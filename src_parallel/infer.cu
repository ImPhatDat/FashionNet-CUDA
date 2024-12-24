#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <getopt.h>
#include "Model/Model.hh"
#include "layer/dense.hh"
#include "layer/relu.hh"
#include "layer/softmax.hh"
#include "utils/helpers.hh"

const int INPUT_SIZE = 784; // Example: MNIST image input size
const int OUTPUT_SIZE = 10;
const int BATCH_SIZE = 1; // Single image for inference

std::mt19937 global_rng(1); // Random number generator
int main(int argc, char **argv) {
    std::string checkpoint_path = "";
    std::string img_path = "";
    int blockSize1d = 256;
    int blockSize2d_x = 32;
    int blockSize2d_y = 32;
    int opt;

    // Parsing command-line arguments
    while ((opt = getopt(argc, argv, "p:i:k:x:y:")) != -1)
    {
        switch (opt)
        {
        case 'p':
            checkpoint_path = optarg; // Store the checkpoint path
            break;
        case 'i':
            img_path = optarg; // Store the checkpoint path
            break;
        case 'k':
            blockSize1d = atoi(optarg); // Convert argument to integer
            break;
        case 'x':
            blockSize2d_x = atoi(optarg); // Convert argument to integer
            break;
        case 'y':
            blockSize2d_y = atoi(optarg); // Convert argument to integer
            break;
        default:
            fprintf(stderr, "Usage: %s [-p checkpoint_path] [-i image_path] [-k blockSize1d] [-x blockSize2d_x] [-y blockSize2d_y]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "\nConfigurations:" << std::endl;
    std::cout << "\tCheckpoint: " << checkpoint_path << std::endl;
    std::cout << "\tImage: " << img_path << std::endl;
    std::cout << "\tBlockSize 1D: " << blockSize1d << std::endl;
    std::cout << "\tBlockSize 2D_x: " << blockSize2d_x << std::endl;
    std::cout << "\tBlockSize 2D_y: " << blockSize2d_y << std::endl;

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

    Layer *layers[] = {
        new Dense(BATCH_SIZE, INPUT_SIZE, 128, false, global_rng),
        new ReLU(BATCH_SIZE, 128),
        new Dense(BATCH_SIZE, 128, 128, false, global_rng),
        new ReLU(BATCH_SIZE, 128),
        new Dense(BATCH_SIZE, 128, OUTPUT_SIZE, false, global_rng),
        new Softmax(BATCH_SIZE, OUTPUT_SIZE)};

    dim3 blockSizes[] = {
        dim3(blockSize2d_x, blockSize2d_y),
        dim3(blockSize1d),
        dim3(blockSize2d_x, blockSize2d_y),
        dim3(blockSize1d),
        dim3(blockSize2d_x, blockSize2d_y),
        dim3(blockSize1d),
    };
    const int NUM_LAYERS = sizeof(layers) / sizeof(layers[0]);

    // Create the model
    Model model(layers, NUM_LAYERS, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);

    // Load the model weights
    model.load_weights(checkpoint_path);

    // Predict using your model

    float* input_d;
    float* output_d;
    CHECK(cudaMalloc(&input_d, INPUT_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&output_d, OUTPUT_SIZE * sizeof(float)));

    CHECK(cudaMemcpy(input_d, input_image, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    model.forward(input_d, output_d, blockSizes);
    float *output = new float[OUTPUT_SIZE];

    CHECK(cudaMemcpy(output, output_d, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));


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