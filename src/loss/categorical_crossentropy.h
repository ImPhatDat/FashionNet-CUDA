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
#include "loss.h"
#pragma once


class CategoricalCrossentropy: public Loss
{
private:
    float epsilon;

public:
    CategoricalCrossentropy(float epsilon = 1e-7);
    ~CategoricalCrossentropy();

    float forward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes);
    void backward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes, float *gradients);
};

#endif
