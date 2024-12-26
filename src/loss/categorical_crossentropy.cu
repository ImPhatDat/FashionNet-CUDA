#include "categorical_crossentropy.hh"

CategoricalCrossentropy::CategoricalCrossentropy(float epsilon) : epsilon(epsilon) {}

CategoricalCrossentropy::~CategoricalCrossentropy() {}

// Forward pass: Computes the loss
float CategoricalCrossentropy::forward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes)
{
    float loss = 0.0f;

    for (int b = 0; b < batch_size; ++b)
    {
        const float *pred_row = y_pred + b * num_classes;
        int label = y_true[b];

        // Prevent log(0) by adding a small constant
        float pred = pred_row[label];
        loss -= logf(pred + this->epsilon);
    }

    this->loss_val = loss / batch_size;
    return this->loss_val;
}

// Backward pass: Computes the gradient with respect to predictions
void CategoricalCrossentropy::backward(const uint8_t *y_true, const float *y_pred, int batch_size, int num_classes, float *gradients)
{
    for (int b = 0; b < batch_size; ++b)
    {
        const float *pred_row = y_pred + b * num_classes;
        float *grad_row = gradients + b * num_classes;
        int label = y_true[b];

        // Compute gradients
        for (int c = 0; c < num_classes; ++c)
        {
            float pred_round = std::max(pred_row[c], this->epsilon);
            grad_row[c] = (c == label ? -1.0f / pred_round : 0.0f);
            // grad_row[c] = (pred_row[c] - (c == label));
        }
    }
}