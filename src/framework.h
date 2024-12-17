#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <cmath>
#pragma once

// utils
bool nearlyEqual(double a, double b, double epsilon = 1e-6, double relativeTolerance = 1e-6);

void relu(float *input, int rows, int cols);

void softmax(float *input, int rows, int cols);

void matmul(const float *A, const float *B, float *C, int M, int K, int N);

// assume sum_over_batch
float categorical_crossentropy_loss(uint8_t *y_true, float *y_pred, int batch_size, int num_classes);
void categorical_crossentropy_gradient(uint8_t* y_true, float* y_pred, float* d_output, int batch_size, int output_size);



#endif 