#include "loss.hh"
#include <iostream>
using namespace std;

Loss::Loss() {
    this->loss_val = 0;
    this->cumulative_loss = 0;
    this->batch_count = 0;
}

Loss::~Loss() {}

// Update state
void Loss::update_state(float batch_loss) {
    cumulative_loss += batch_loss;
    ++batch_count;
}

// Reset state
void Loss::reset_state() {
    cumulative_loss = 0.0f;
    batch_count = 0;
}

// Compute average loss
float Loss::compute_average_loss() const {
    if (batch_count == 0) {
        return 0.0f; // Avoid division by zero
    }
    return cumulative_loss / batch_count;
}