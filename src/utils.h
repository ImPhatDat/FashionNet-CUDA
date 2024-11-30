#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <algorithm>

class FashionMnist {
public:
    // Image struct to represent a single image
    struct Image {
        std::vector<uint8_t> pixels;  // Pixel data
        uint8_t label;                // Classification label
        int width;   // Image width
        int height;  // Image height

        // Constructor
        Image(int w = 28, int h = 28) : width(w), height(h), label(0) {
            pixels.resize(w * h);
        }

        // Method to get pixel at specific coordinate
        uint8_t getPixel(int x, int y) const {
            if (x < 0 || x >= width || y < 0 || y >= height) {
                throw std::out_of_range("Pixel coordinates out of bounds");
            }
            return pixels[y * width + x];
        }

        // Print ASCII representation of the image
        void print() const {
            std::cout << "Label: " << static_cast<int>(label) << std::endl;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    char pixel = getPixel(x, y) > 128 ? '#' : '.';
                    std::cout << pixel;
                }
                std::cout << std::endl;
            }
        }
    };

private:
    // Container for images
    std::vector<Image> images;
    int imageWidth;
    int imageHeight;

public:
    // Constructor
    FashionMnist(int width = 28, int height = 28) 
        : imageWidth(width), imageHeight(height) {}

    // Method to load dataset from files
    void loadDataset(
        const std::string& imageFilePath, 
        const std::string& labelFilePath
    ) {
        // Clear existing images
        images.clear();

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

        // Resize images vector
        images.reserve(numImages);

        // Read images and labels
        for (uint32_t i = 0; i < numImages; ++i) {
            Image image(imageWidth, imageHeight);
            
            // Read image pixels
            image.pixels.resize(rows * cols);
            imageFile.read(reinterpret_cast<char*>(image.pixels.data()), rows * cols);
            
            // Read label
            labelFile.read(reinterpret_cast<char*>(&image.label), 1);

            images.push_back(std::move(image));
        }
    }

    // Getters
    size_t getImageCount() const { return images.size(); }
    const Image& getImage(size_t index) const {
        if (index >= images.size()) {
            throw std::out_of_range("Image index out of range");
        }
        return images[index];
    }

    // Iterate through images
    const std::vector<Image>& getImages() const { return images; }

    // Method to get images with a specific label
    std::vector<Image> getImagesByLabel(uint8_t label) const {
        std::vector<Image> labeledImages;
        for (const auto& image : images) {
            if (image.label == label) {
                labeledImages.push_back(image);
            }
        }
        return labeledImages;
    }
};