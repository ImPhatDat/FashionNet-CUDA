#include <stdio.h>
#include <stdint.h>
#include <random>
#include "fashion_mnist.h"
#include "Model/Model.h"
#include "layer/dense.h"
#include "layer/relu.h"
#include "layer/softmax.h"
#include "loss/categorical_crossentropy.h"
#include "cuda_utils.cu"


const std::string train_imageFilePath = "data/fashion-mnist/train-images-idx3-ubyte";
const std::string train_labelFilePath = "data/fashion-mnist/train-labels-idx1-ubyte";
const std::string test_imageFilePath = "data/fashion-mnist/t10k-images-idx3-ubyte";
const std::string test_labelFilePath = "data/fashion-mnist/t10k-labels-idx1-ubyte";

std::mt19937 global_rng(1); // Random number generator
// Model configurations
const int INPUT_SIZE = 784; // Example: MNIST image input size
const int BATCH_SIZE = 64;
const int OUTPUT_SIZE = 10;

const float LEARNING_RATE = 0.1;

Layer* layers[] = {
    new Dense(BATCH_SIZE, INPUT_SIZE, 128, global_rng),
    new ReLU(BATCH_SIZE, 128),
    new Dense(BATCH_SIZE, 128, 128, global_rng),
    new ReLU(BATCH_SIZE, 128),
    new Dense(BATCH_SIZE, 128, OUTPUT_SIZE, global_rng),
    new Softmax(BATCH_SIZE, OUTPUT_SIZE)
};

const int NUM_LAYERS = sizeof(layers) / sizeof(layers[0]);



int main(int argc, char **argv)
{
    printDeviceInfo();

    // Load dataset
    FashionMnist train_set;
    train_set.loadDataset(train_imageFilePath, train_labelFilePath);
    FashionMnist test_set;
    test_set.loadDataset(test_imageFilePath, test_labelFilePath);
    std::cout << "Total train images: " << train_set.getImageCount() << std::endl;
    std::cout << "Total test images: " << test_set.getImageCount() << std::endl;

    int total_images = train_set.getImageCount();
    int num_batches = total_images / BATCH_SIZE;

    // Prepare batches
    train_set.shuffle(global_rng);

    float **x_batches = nullptr;
    uint8_t **y_batches = nullptr;

    train_set.prepareBatchesWithLabels(BATCH_SIZE, INPUT_SIZE, x_batches, y_batches);

    float **y_pred_batches = new float *[num_batches];
    for (int bi = 0; bi < num_batches; ++bi)
    {
        y_pred_batches[bi] = new float[BATCH_SIZE * OUTPUT_SIZE];
    }

    Model model(layers, NUM_LAYERS, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);

    CategoricalCrossentropy loss_obj(1e-7);

    for (int bi = 0; bi < num_batches; ++bi) {
        model.forward(x_batches[bi], y_pred_batches[bi]);
        
        float loss = loss_obj.forward(y_batches[bi], y_pred_batches[bi], BATCH_SIZE, OUTPUT_SIZE);

        std::cout << "Loss for batch " << bi << ": " << loss << std::endl;

        model.backward(y_batches[bi], y_pred_batches[bi], &loss_obj);

        model.update_weights(LEARNING_RATE);
    } 

    // Deallocate
    for (size_t i = 0; i < num_batches; ++i)
    {
        delete[] x_batches[i];
        delete[] y_pred_batches[i];
        delete[] y_batches[i];
    }
    delete[] x_batches;
    delete[] y_pred_batches;
    delete[] y_batches;

    return 0;
}
