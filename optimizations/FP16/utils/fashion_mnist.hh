// FashionMnist.h
#ifndef FASHION_MNIST_H
#define FASHION_MNIST_H

#include "helpers.hh"
#include <stdint.h>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <iostream>

class FashionMnist
{
public:
    struct Image
    {
        uint8_t *pixels;
        uint8_t label;
        int width;
        int height;

        Image(int w = 28, int h = 28);
        ~Image();
        Image(const Image &other);
        Image &operator=(const Image &other);
        Image(Image &&other) noexcept;
        Image &operator=(Image &&other) noexcept;

        uint8_t getPixel(int x, int y) const;
        void print() const;
    };

private:
    Image *images;
    size_t imageCount;
    int imageWidth;
    int imageHeight;

public:
    FashionMnist(int width = 28, int height = 28);
    ~FashionMnist();
    FashionMnist(const FashionMnist &other);
    FashionMnist &operator=(const FashionMnist &other);
    FashionMnist(FashionMnist &&other) noexcept;
    FashionMnist &operator=(FashionMnist &&other) noexcept;

    void loadDataset(const std::string &imageFilePath, const std::string &labelFilePath);
    size_t getImageCount() const;
    const Image &getImage(size_t index) const;
    void shuffle(std::mt19937 randomGenerator);
    void prepareBatchesWithLabels(int batch_size, int input_size, __half **batches, uint8_t **batch_labels);
};

#endif // FASHION_MNIST_H