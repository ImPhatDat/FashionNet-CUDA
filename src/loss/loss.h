#ifndef LOSS_H
#define LOSS_H

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#pragma once


// Assumption:
// 1. sum over batch
// 2. y_true is sparse representation (1d array of [0-num_class - 1])
class Loss
{
protected:
    float loss_val;

public:
    Loss();
    virtual ~Loss();

    virtual float forward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes);
    virtual void backward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes, float *gradients);
};

#endif
