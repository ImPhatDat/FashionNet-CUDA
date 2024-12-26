#ifndef CATEGORICAL_CROSSENTROPY_H
#define CATEGORICAL_CROSSENTROPY_H

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#include "loss.hh"
#pragma once


class CategoricalCrossentropy: public Loss
{
private:
    __half epsilon;

public:
    CategoricalCrossentropy(__half epsilon = 1e-7);
    ~CategoricalCrossentropy();

    float forward(const uint8_t *y_true, const __half *y_pred, int batch_size, int num_classes, dim3 blockSize);
    void backward(const uint8_t *y_true, const __half *y_pred, int batch_size, int num_classes, __half *gradients, dim3 blockSize);
};

#endif
