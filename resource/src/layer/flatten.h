#ifndef SRC_LAYER_FLATTEN_H_
#define SRC_LAYER_FLATTEN_H_

#include "../layer.h"

class Flatten : public Layer {
 public:
  // Constructor to initialize the input dimension.
  Flatten(int input_dim);

  // Forward pass
  void forward(const Matrix& bottom) override;

  // Backward pass
  void backward(const Matrix& bottom, const Matrix& grad_top) override;

  // Output dimension after flattening (1D vector size)
  int output_dim() override;

 private:
  int input_dim_; // input dimensions (before flattening)
  int output_dim_; // output dimension (after flattening)
};

#endif  // SRC_LAYER_FLATTEN_H_
