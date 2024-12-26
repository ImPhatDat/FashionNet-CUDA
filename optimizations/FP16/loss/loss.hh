#ifndef LOSS_H
#define LOSS_H
#include "../utils/helpers.hh"
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
public:
    float cumulative_loss;        // Sum of losses for all processed batches
    int batch_count;              // Number of processed batches
protected:
    float loss_val;

public:
    Loss();
    virtual ~Loss();

    virtual float forward(const uint8_t *y_true, const __half *y_pred, int batch_size, int num_classes, dim3 blockSize) = 0;
    virtual void backward(const uint8_t *y_true, const __half *y_pred, int batch_size, int num_classes, __half *gradients, dim3 blockSize) = 0;

    // Update state: Logs and accumulates loss for each batch
    void update_state(float batch_loss);

    // Reset state: Clears cumulative loss and batch count
    void reset_state();

    // Compute average loss
    float compute_average_loss() const;
};

#endif
