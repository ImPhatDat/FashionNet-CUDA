#include "Model.h"

FashionNet::FashionNet() {
    this->layers = nullptr;
    this->num_layers = 0;
}
FashionNet::FashionNet(Layer* layers[], int num_layers, int batch_size, int input_size, int num_classes) {
    this->num_layers = num_layers;
    this->batch_size = batch_size;
    this->input_size = input_size;
    this->num_classes = num_classes;

    for (int i = 0; i < num_layers; i++) {
        this->layers = layers;
    }
}
FashionNet::~FashionNet() {
    for (int i = 0; i < num_layers; i++) {
        delete this->layers[i];
    }
    delete[] this->layers;
}


void FashionNet::forward(const float* batch_input, float* batch_output) {
    float* x = new float[this->batch_size * this->input_size];
    float* tmp_x;
    std::memcpy(x, batch_input, sizeof(float) * this->batch_size * this->input_size);

    for (int i = 0; i < this->num_layers; i++) {
        int next_output_size = this->layers[i]->output_size;
        tmp_x = new float[this->batch_size * next_output_size];
        this->layers[i]->forward(x, tmp_x);

        delete[] x;
        x = tmp_x;
    }

    std::memcpy(batch_output, x, sizeof(float) * this->batch_size * this->num_classes);
    delete[] x;
}

void FashionNet::backward(const uint8_t* y_true, const float* y_pred, Loss* loss) {
    float* grad_x = new float[this->batch_size * this->num_classes];
    loss->backward(y_true, y_pred, this->batch_size, this->num_classes, grad_x);

    for (int i = num_classes - 1; i >= 0; i--) {
        float* grad_tmp = new float[this->batch_size * this->layers[i]->input_size];
        this->layers[i]->backward(grad_x, grad_tmp);
        
        delete[] grad_x;
        grad_x = grad_tmp;
    }

    delete[] grad_x;
}




