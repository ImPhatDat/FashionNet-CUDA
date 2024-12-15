#include <stdio.h>
#include <stdint.h>
#include <random>

#include <iostream>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#pragma once

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

class FashionMnist
{
public:
    struct Image
    {
        uint8_t *pixels;
        uint8_t label;
        int width;
        int height;

        // Default constructor
        Image(int w = 28, int h = 28) : width(w), height(h), label(0)
        {
            pixels = new uint8_t[w * h];
            std::fill_n(pixels, w * h, 0);
        }

        // Destructor
        ~Image()
        {
            delete[] pixels;
        }

        // Copy constructor
        Image(const Image &other) : width(other.width), height(other.height), label(other.label)
        {
            pixels = new uint8_t[width * height];
            std::memcpy(pixels, other.pixels, width * height);
        }

        // Copy assignment operator
        Image &operator=(const Image &other)
        {
            if (this != &other)
            {
                delete[] pixels;
                width = other.width;
                height = other.height;
                label = other.label;
                pixels = new uint8_t[width * height];
                std::memcpy(pixels, other.pixels, width * height);
            }
            return *this;
        }

        // Move constructor
        Image(Image &&other) noexcept
            : pixels(other.pixels), label(other.label), width(other.width), height(other.height)
        {
            other.pixels = nullptr;
            other.width = other.height = 0;
            other.label = 0;
        }

        // Move assignment operator
        Image &operator=(Image &&other) noexcept
        {
            if (this != &other)
            {
                delete[] pixels;
                pixels = other.pixels;
                label = other.label;
                width = other.width;
                height = other.height;

                other.pixels = nullptr;
                other.width = other.height = 0;
                other.label = 0;
            }
            return *this;
        }

        uint8_t getPixel(int x, int y) const
        {
            if (x < 0 || x >= width || y < 0 || y >= height)
            {
                throw std::out_of_range("Pixel coordinates out of bounds");
            }
            return pixels[y * width + x];
        }

        void print() const
        {
            std::cout << "Label: " << static_cast<int>(label) << std::endl;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    char pixel = getPixel(x, y) > 128 ? '#' : '.';
                    std::cout << pixel;
                }
                std::cout << std::endl;
            }
        }
    };

private:
    Image *images;
    size_t imageCount;
    int imageWidth;
    int imageHeight;

public:
    // Constructor
    FashionMnist(int width = 28, int height = 28)
        : images(nullptr), imageWidth(width), imageHeight(height), imageCount(0) {}

    // Destructor
    ~FashionMnist()
    {
        delete[] images;
    }

    // Copy constructor
    FashionMnist(const FashionMnist &other)
        : imageWidth(other.imageWidth), imageHeight(other.imageHeight), imageCount(other.imageCount)
    {
        images = new Image[imageCount];
        for (size_t i = 0; i < imageCount; ++i)
        {
            images[i] = other.images[i];
        }
    }

    // Copy assignment operator
    FashionMnist &operator=(const FashionMnist &other)
    {
        if (this != &other)
        {
            delete[] images;
            imageWidth = other.imageWidth;
            imageHeight = other.imageHeight;
            imageCount = other.imageCount;

            images = new Image[imageCount];
            for (size_t i = 0; i < imageCount; ++i)
            {
                images[i] = other.images[i];
            }
        }
        return *this;
    }

    // Move constructor
    FashionMnist(FashionMnist &&other) noexcept
        : images(other.images), imageCount(other.imageCount),
          imageWidth(other.imageWidth), imageHeight(other.imageHeight)
    {
        other.images = nullptr;
        other.imageCount = 0;
        other.imageWidth = 0;
        other.imageHeight = 0;
    }

    // Move assignment operator
    FashionMnist &operator=(FashionMnist &&other) noexcept
    {
        if (this != &other)
        {
            delete[] images;
            images = other.images;
            imageCount = other.imageCount;
            imageWidth = other.imageWidth;
            imageHeight = other.imageHeight;

            other.images = nullptr;
            other.imageCount = 0;
            other.imageWidth = 0;
            other.imageHeight = 0;
        }
        return *this;
    }

    void loadDataset(
        const std::string &imageFilePath,
        const std::string &labelFilePath)
    {
        std::ifstream imageFile(imageFilePath, std::ios::binary);
        std::ifstream labelFile(labelFilePath, std::ios::binary);

        if (!imageFile)
        {
            throw std::runtime_error("Cannot open image file: " + imageFilePath);
        }

        if (!labelFile)
        {
            throw std::runtime_error("Cannot open label file: " + labelFilePath);
        }

        uint32_t imageMagic, labelMagic;
        uint32_t numImages, numLabels;
        uint32_t rows, cols;

        imageFile.read(reinterpret_cast<char *>(&imageMagic), 4);
        imageFile.read(reinterpret_cast<char *>(&numImages), 4);
        imageFile.read(reinterpret_cast<char *>(&rows), 4);
        imageFile.read(reinterpret_cast<char *>(&cols), 4);

        labelFile.read(reinterpret_cast<char *>(&labelMagic), 4);
        labelFile.read(reinterpret_cast<char *>(&numLabels), 4);

        imageMagic = __builtin_bswap32(imageMagic);
        labelMagic = __builtin_bswap32(labelMagic);
        numImages = __builtin_bswap32(numImages);
        numLabels = __builtin_bswap32(numLabels);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        if (imageMagic != 0x00000803 || labelMagic != 0x00000801)
        {
            throw std::runtime_error("Invalid file format");
        }

        if (numImages != numLabels)
        {
            throw std::runtime_error("Mismatch between image and label counts");
        }

        delete[] images;

        images = new Image[numImages];
        for (uint32_t i = 0; i < numImages; ++i)
        {
            Image &image = images[i];

            imageFile.read(reinterpret_cast<char *>(image.pixels), rows * cols);
            labelFile.read(reinterpret_cast<char *>(&image.label), 1);
        }
        imageCount = numImages;
    }

    size_t getImageCount() const { return imageCount; }

    const Image &getImage(size_t index) const
    {
        if (index >= imageCount)
        {
            throw std::out_of_range("Image index out of range");
        }
        return images[index];
    }

    // Method to shuffle the dataset
    void shuffle(std::mt19937 randomGenerator)
    {
        if (!images || imageCount == 0)
            return;
        std::shuffle(images, images + imageCount, randomGenerator);
    }
};

void prepareBatchesWithLabels(
    const FashionMnist& dataset, 
    int batch_size, 
    int input_size, 
    float**& batches, 
    uint8_t**& batch_labels
) {
    size_t total_images = dataset.getImageCount();
    size_t num_batches = total_images / batch_size;

    // Allocate array of batches
    batches = new float*[num_batches];
    batch_labels = new uint8_t*[num_batches];

    for (size_t batch = 0; batch < num_batches; ++batch) {
        // Allocate memory for current batch and labels
        batches[batch] = new float[batch_size * input_size];
        batch_labels[batch] = new uint8_t[batch_size];

        for (int i = 0; i < batch_size; ++i) {
            // Calculate the index of the current image in the dataset
            size_t image_index = batch * batch_size + i;

            // Get the current image
            const auto& image = dataset.getImage(image_index);

            // Copy pixel data to batch
            for (int j = 0; j < input_size; ++j) {
                // Normalize pixel values to [0, 1] range
                batches[batch][i * input_size + j] =
                    static_cast<float>(image.pixels[j]) / 255.0f;
            }

            // Store the label
            batch_labels[batch][i] = image.label;
        }
    }
}