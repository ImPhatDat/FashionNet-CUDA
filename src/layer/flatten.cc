#include "./flatten.h"

Flatten::Flatten(int input_dim) {
  // Input dimension is the number of elements in the input (e.g., 28x28 for MNIST)
  input_dim_ = input_dim;
  // After flattening, the output dimension is the same size (i.e., 784 for MNIST images)
  output_dim_ = input_dim_;
}

void Flatten::forward(const Matrix& bottom) {
  // The input is a matrix (28x28 for MNIST images), which we flatten into a 1D vector
  top = bottom; // no transformation needed, we just need to reshape
  top.resize(1, input_dim_); // Reshaping to 1x784
}

void Flatten::backward(const Matrix& bottom, const Matrix& grad_top) {
  // During backpropagation, we propagate the gradient back through the flatten layer
  grad_bottom = grad_top; // The gradient with respect to the input is the same as the gradient with respect to the output
}

int Flatten::output_dim() {
  return output_dim_; // Return the flattened output dimension (e.g., 784 for MNIST)
}
