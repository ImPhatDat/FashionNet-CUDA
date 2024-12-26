#include "fashion_mnist.hh"
#include <fstream>
#include <iostream>

// FashionMnist::Image Implementation
FashionMnist::Image::Image(int w, int h) : width(w), height(h), label(0)
{
    pixels = new uint8_t[w * h];
    std::fill_n(pixels, w * h, 0);
}

FashionMnist::Image::~Image()
{
    delete[] pixels;
}

FashionMnist::Image::Image(const Image &other) : width(other.width), height(other.height), label(other.label)
{
    pixels = new uint8_t[width * height];
    std::memcpy(pixels, other.pixels, width * height);
}

FashionMnist::Image &FashionMnist::Image::operator=(const Image &other)
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

FashionMnist::Image::Image(Image &&other) noexcept
    : pixels(other.pixels), label(other.label), width(other.width), height(other.height)
{
    other.pixels = nullptr;
    other.width = other.height = 0;
    other.label = 0;
}

FashionMnist::Image &FashionMnist::Image::operator=(Image &&other) noexcept
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

uint8_t FashionMnist::Image::getPixel(int x, int y) const
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        throw std::out_of_range("Pixel coordinates out of bounds");
    }
    return pixels[y * width + x];
}

void FashionMnist::Image::print() const
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

// FashionMnist Implementation
FashionMnist::FashionMnist(int width, int height)
    : images(nullptr), imageWidth(width), imageHeight(height), imageCount(0) {}

FashionMnist::~FashionMnist()
{
    delete[] images;
}

FashionMnist::FashionMnist(const FashionMnist &other)
    : imageWidth(other.imageWidth), imageHeight(other.imageHeight), imageCount(other.imageCount)
{
    images = new Image[imageCount];
    for (size_t i = 0; i < imageCount; ++i)
    {
        images[i] = other.images[i];
    }
}

FashionMnist &FashionMnist::operator=(const FashionMnist &other)
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

FashionMnist::FashionMnist(FashionMnist &&other) noexcept
    : images(other.images), imageCount(other.imageCount),
      imageWidth(other.imageWidth), imageHeight(other.imageHeight)
{
    other.images = nullptr;
    other.imageCount = 0;
    other.imageWidth = 0;
    other.imageHeight = 0;
}

FashionMnist &FashionMnist::operator=(FashionMnist &&other) noexcept
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

void FashionMnist::loadDataset(const std::string &imageFilePath, const std::string &labelFilePath)
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

size_t FashionMnist::getImageCount() const { return imageCount; }

const FashionMnist::Image &FashionMnist::getImage(size_t index) const
{
    if (index >= imageCount)
    {
        throw std::out_of_range("Image index out of range");
    }
    return images[index];
}

void FashionMnist::shuffle(std::mt19937 randomGenerator)
{
    if (!images || imageCount == 0)
        return;
    std::shuffle(images, images + imageCount, randomGenerator);
}

void FashionMnist::prepareBatchesWithLabels(int batch_size, int input_size, float **batches, uint8_t **batch_labels)
{
    size_t total_images = this->getImageCount();
    size_t num_batches = total_images / batch_size;

    for (size_t batch = 0; batch < num_batches; ++batch)
    {

        for (int i = 0; i < batch_size; ++i)
        {
            size_t image_index = batch * batch_size + i;

            const auto &image = this->getImage(image_index);

            for (int j = 0; j < input_size; ++j)
            {
                batches[batch][i * input_size + j] = static_cast<float>(image.pixels[j]) / 255.0f;
            }

            batch_labels[batch][i] = image.label;
        }
    }
}
