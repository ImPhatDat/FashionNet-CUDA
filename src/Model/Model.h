#pragma once
#include "../layer/layer.h"
#include "../loss/loss.h"
#include <stdio.h>
#include <stdint.h>

class Model {
private:
	Layer** layers;
    int num_layers;
    
    int batch_size;
    int input_size;
    int num_classes;

public:
	Model();
	Model(Layer* layers[], int num_layers, int batch_size, int input_size, int num_classes);
	~Model();

	void forward(const float* batch_input, float* batch_output);
	void backward(const uint8_t* y_true, const float* y_pred, Loss* loss_func);
	void update_weights(const float learning_rate);
};