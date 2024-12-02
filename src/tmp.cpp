#include <cmath>   // For std::log
#include <iostream> // For std::cout

float categorical_crossentropy_loss(int* y_true, float* y_pred, int batch_size, int num_classes) {
    float total_loss = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        int true_class = y_true[i];
        float predicted_prob = y_pred[i * num_classes + true_class];
        
        // Avoid log(0) by clamping probabilities to a small positive value
        const float epsilon = 1e-7f;
        predicted_prob = std::max(predicted_prob, epsilon);

        total_loss -= std::log(predicted_prob);
    }

    return total_loss / batch_size;
}

// Example usage
int main() {
    int y_true[] = {1, 2}; // Batch size = 3, True labels for each sample
    float y_pred[] = {0.05, 0.95, 0,   // Predicted probs for sample 1
                      0.1, 0.8, 0.1};  // Predicted probs for sample 3
    int batch_size = 2;
    int num_classes = 3;

    float loss = categorical_crossentropy_loss(y_true, y_pred, batch_size, num_classes);
    std::cout << "Loss: " << loss << std::endl;

    return 0;
}
