#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/fully_connected.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

int main() {
  // Load MNIST dataset
  MNIST dataset("../data/fashion-mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "MNIST train samples: " << n_train << std::endl;
  std::cout << "MNIST test samples: " << dataset.test_labels.cols() << std::endl;

  // Perform flattening directly in main
  int flatten_dim = dim_in; // Assume 28x28 images -> flatten to 784
  dataset.train_data.resize(flatten_dim, n_train); // Flatten training data
  dataset.test_data.resize(flatten_dim, dataset.test_data.cols()); // Flatten test data

  // Initialize the network
  Network dnn;

  // Define the architecture
  Layer* dense1 = new FullyConnected(flatten_dim, 128); // Dense layer with 128 units
  Layer* relu1 = new ReLU;                              // ReLU activation
  Layer* dense2 = new FullyConnected(128, 128);         // Dense layer with 128 units
  Layer* relu2 = new ReLU;                              // ReLU activation
  Layer* output = new FullyConnected(128, 10);          // Output layer with 10 units
  Layer* softmax = new Softmax;                         // Softmax activation

  // Add layers to the network
  dnn.add_layer(dense1);
  dnn.add_layer(relu1);
  dnn.add_layer(dense2);
  dnn.add_layer(relu2);
  dnn.add_layer(output);
  dnn.add_layer(softmax);

  // Define the loss function
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);

  // Optimizer
  SGD opt(0.001, 5e-4, 0.9, true);

  // Training configuration
  const int n_epoch = 5;
  const int batch_size = 128;

  // Training loop
  for (int epoch = 0; epoch < n_epoch; epoch++) {
    shuffle_data(dataset.train_data, dataset.train_labels);

    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, flatten_dim,
                                std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);

      // Forward and backward pass
      dnn.forward(x_batch);
      dnn.backward(x_batch, target_batch);

      // Display loss
      if (ith_batch % 50 == 0) {
        std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
                  << std::endl;
      }

      // Update parameters
      dnn.update(opt);
    }

    // Test the model
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test accuracy: " << acc << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
