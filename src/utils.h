#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

class FashionMNISTReader {
public:
    // Structure to hold image data
    struct Image {
        std::vector<uint8_t> pixels;
        int label;
    };

    // Read images from file
    static std::vector<Image> readDataset(
        const std::string& imageFilePath, 
        const std::string& labelFilePath
    ) {
        // Open image file
        std::ifstream imageFile(imageFilePath, std::ios::binary);
        if (!imageFile) {
            throw std::runtime_error("Cannot open image file: " + imageFilePath);
        }

        // Open label file
        std::ifstream labelFile(labelFilePath, std::ios::binary);
        if (!labelFile) {
            throw std::runtime_error("Cannot open label file: " + labelFilePath);
        }

        // Read magic numbers and metadata
        uint32_t imageMagic, labelMagic;
        uint32_t numImages, numLabels;
        uint32_t rows, cols;

        // Read image file header
        imageFile.read(reinterpret_cast<char*>(&imageMagic), 4);
        imageFile.read(reinterpret_cast<char*>(&numImages), 4);
        imageFile.read(reinterpret_cast<char*>(&rows), 4);
        imageFile.read(reinterpret_cast<char*>(&cols), 4);

        // Read label file header
        labelFile.read(reinterpret_cast<char*>(&labelMagic), 4);
        labelFile.read(reinterpret_cast<char*>(&numLabels), 4);

        // Convert from big-endian to host byte order
        imageMagic = __builtin_bswap32(imageMagic);
        labelMagic = __builtin_bswap32(labelMagic);
        numImages = __builtin_bswap32(numImages);
        numLabels = __builtin_bswap32(numLabels);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        // Validate headers
        if (imageMagic != 0x00000803 || labelMagic != 0x00000801) {
            throw std::runtime_error("Invalid file format");
        }

        if (numImages != numLabels) {
            throw std::runtime_error("Mismatch between image and label counts");
        }

        // Vector to store all images
        std::vector<Image> dataset;
        dataset.reserve(numImages);

        // Read images and labels
        for (uint32_t i = 0; i < numImages; ++i) {
            Image image;
            image.pixels.resize(rows * cols);
            
            // Read image pixels
            imageFile.read(reinterpret_cast<char*>(image.pixels.data()), rows * cols);
            
            // Read label
            uint8_t label;
            labelFile.read(reinterpret_cast<char*>(&label), 1);
            image.label = label;

            dataset.push_back(std::move(image));
        }

        return dataset;
    }

    // Utility function to print image details
    static void printImage(const Image& image, int rows = 28, int cols = 28) {
        std::cout << "Label: " << image.label << std::endl;
        
        // Print ASCII representation of the image
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                char pixel = image.pixels[i * cols + j] > 128 ? '#' : '.';
                std::cout << pixel;
            }
            std::cout << std::endl;
        }
    }
};