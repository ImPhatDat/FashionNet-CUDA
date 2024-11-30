#include "utils.h"

// Example usage
int main() {
    try {
        // Paths to your MNIST Fashion dataset files
        std::string imageFilePath = "../data/fashion-mnist/train-images-idx3-ubyte";
        std::string labelFilePath = "../data/fashion-mnist/train-labels-idx1-ubyte";

        // Read the dataset
        auto dataset = FashionMNISTReader::readDataset(imageFilePath, labelFilePath);

        

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}