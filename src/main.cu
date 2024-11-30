#include "utils.h"

// Example usage
int main() {
    try {
        // Create FashionMnist object
        FashionMnist fashionMnist;

        // Paths to your MNIST Fashion dataset files
        std::string imageFilePath = "../data/fashion-mnist/train-images-idx3-ubyte";
        std::string labelFilePath = "../data/fashion-mnist/train-labels-idx1-ubyte";

        // Load the dataset
        fashionMnist.loadDataset(imageFilePath, labelFilePath);

        // Print total number of images
        std::cout << "Total images: " << fashionMnist.getImageCount() << std::endl;

        // Print first 5 images
        for (int i = 0; i < 5; ++i) {
            std::cout << "Image " << i << ":" << std::endl;
            fashionMnist.getImage(i).print();
            std::cout << std::endl;
        }

        // Get images with label 0 (T-Shirt/Top)
        auto tshirtImages = fashionMnist.getImagesByLabel(0);
        std::cout << "Number of T-Shirt/Top images: " << tshirtImages.size() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}