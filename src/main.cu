#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include "framework.h"

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
