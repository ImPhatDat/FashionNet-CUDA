#include <stdio.h>
#include <stdint.h>
#include <random>
#include "utils.h"
#include "framework.h"
#include "dense.h"

std::mt19937 global_rng(1); // Random number generator

// Model configurations
const int INPUT_SIZE = 784;   // Example: MNIST image input size
const int BATCH_SIZE = 64;
const int DENSE_OUTPUT[] = {128, 128, 10};
const int NUM_LAYERS = sizeof(DENSE_OUTPUT) / sizeof(DENSE_OUTPUT[0]);
const std::string ACTIVATION_TYPES[] = {"relu", "relu", "softmax"};

const std::string train_imageFilePath = "../data/fashion-mnist/train-images-idx3-ubyte";
const std::string train_labelFilePath = "../data/fashion-mnist/train-labels-idx1-ubyte";
const std::string test_imageFilePath = "../data/fashion-mnist/t10k-images-idx3-ubyte";
const std::string test_labelFilePath = "../data/fashion-mnist/t10k-labels-idx1-ubyte";


int main(int argc, char ** argv) {
    printDeviceInfo();
    
    // Create FashionMnist object
    FashionMnist train_set;
    train_set.loadDataset(train_imageFilePath, train_labelFilePath);
    FashionMnist test_set;
    test_set.loadDataset(test_imageFilePath, test_labelFilePath);

    // Print total number of images
    std::cout << "Total train images: " << train_set.getImageCount() << std::endl;
    std::cout << "Total test images: " << test_set.getImageCount() << std::endl;

    // Allocate dynamic array for Dense layers
    Dense* layers = new Dense[NUM_LAYERS];
    int previous_size = INPUT_SIZE;

    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        layers[i] = Dense(
            previous_size,       // input size 
            DENSE_OUTPUT[i],     // output size
            BATCH_SIZE,          // batch size
            ACTIVATION_TYPES[i], // activation type
            global_rng           // random number generator
        );
        previous_size = DENSE_OUTPUT[i]; // Update input size for next layer
    }

    // Assuming train_set is already loaded
    size_t total_images = train_set.getImageCount();
    size_t num_batches = total_images / BATCH_SIZE;

    // Prepare batches
    train_set.shuffle(global_rng);

    float** input_batches = prepareBatches(train_set, BATCH_SIZE, INPUT_SIZE);

    float* output = new float[BATCH_SIZE * DENSE_OUTPUT[NUM_LAYERS - 1]];

    model_forward(input_batches[0], INPUT_SIZE, output, layers, NUM_LAYERS);

    // When done
    cleanupBatches(input_batches, num_batches);
    delete[] output;
    delete[] layers;

    return 0;
}
