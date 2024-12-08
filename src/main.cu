#include <stdio.h>
#include <stdint.h>
#include <random>
#include "utils.h"
#include "framework.h"

std::mt19937 global_rng(1); // Random number generator

// Model configurations
const int DENSE_OUTPUT[] = {128, 128, 10};
const std::string ACTIVATION_TYPES[] = {"relu", "relu", "softmax"};
const int BATCH_SIZE = 64;


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

    

    return 0;
}
