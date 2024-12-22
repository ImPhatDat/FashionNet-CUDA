#ifndef ACCURACY_METRIC_H
#define ACCURACY_METRIC_H

#include <cstdint>

// External argmax function declaration
int argmax(const float* pred, int num_classes);

// Accuracy metric class
class Accuracy {
public:
    // Constructor
    Accuracy();

    // Update the state with predictions and true labels
    void update_state(const float* pred, const uint8_t* true_labels, int batch_size, int num_classes);

    // Compute and return the accuracy
    float compute() const;

    // Reset the state
    void reset_state();

private:
    int correct_predictions; // Number of correct predictions
    int total_predictions;   // Total number of predictions
};

#endif // ACCURACY_METRIC_H
