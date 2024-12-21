#include "accuracy.h"
#include <stdexcept>

// External argmax function definition
int argmax(const float* pred, int num_classes) {
    int max_index = 0;
    float max_value = pred[0];

    for (int i = 1; i < num_classes; ++i) {
        if (pred[i] > max_value) {
            max_value = pred[i];
            max_index = i;
        }
    }
    return max_index;
}

// Constructor
Accuracy::Accuracy() : correct_predictions(0), total_predictions(0) {}

// Update state
void Accuracy::update_state(const float* pred, const uint8_t* true_labels, int batch_size, int num_classes) {
    if (!pred || !true_labels || batch_size <= 0 || num_classes <= 0) {
        throw std::invalid_argument("Invalid input to update_state.");
    }

    for (int i = 0; i < batch_size; ++i) {
        int predicted_class = argmax(pred + i * num_classes, num_classes); // Argmax for each prediction
        if (predicted_class == true_labels[i]) {
            ++correct_predictions; // Increment correct predictions
        }
        ++total_predictions; // Increment total predictions
    }
}

// Compute accuracy
float Accuracy::compute() const {
    if (total_predictions == 0) {
        throw std::runtime_error("No predictions to compute accuracy.");
    }
    return static_cast<float>(correct_predictions) / total_predictions;
}

// Reset state
void Accuracy::reset_state() {
    correct_predictions = 0;
    total_predictions = 0;
}
