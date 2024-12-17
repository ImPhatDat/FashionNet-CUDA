#include "dense.h"

// Default Constructor
Dense::Dense()
    : input_size(0), output_size(0), batch_size(0), activation_type("none"),
        weights(nullptr), biases(nullptr), grad_weights(nullptr), grad_biases(nullptr)
{
    // Empty: no memory allocation needed
}

Dense::Dense(int input_size, int output_size, int batch_size, std::string activation_type, std::mt19937 &gen)
        : input_size(input_size), output_size(output_size), batch_size(batch_size), activation_type(activation_type)
{
    // Allocate and initialize weights and biases
    weights = new float[input_size * output_size];
    biases = new float[output_size];

    grad_weights = new float[input_size * output_size];
    grad_biases = new float[output_size];

    initialize_dense(weights, biases, input_size, output_size, gen); // Initialize weights
}

Dense &Dense::operator=(const Dense &other)
{
    if (this == &other) // Self-assignment check
        return *this;

    // Free existing memory
    delete[] weights;
    delete[] biases;
    delete[] grad_weights;
    delete[] grad_biases;

    // Copy data members
    input_size = other.input_size;
    output_size = other.output_size;
    batch_size = other.batch_size;
    activation_type = other.activation_type;

    // Allocate new memory and copy contents
    weights = new float[input_size * output_size];
    biases = new float[output_size];
    grad_weights = new float[input_size * output_size];
    grad_biases = new float[output_size];

    std::copy(other.weights, other.weights + input_size * output_size, weights);
    std::copy(other.biases, other.biases + output_size, biases);
    std::copy(other.grad_weights, other.grad_weights + input_size * output_size, grad_weights);
    std::copy(other.grad_biases, other.grad_biases + output_size, grad_biases);

    return *this;
}

Dense::~Dense()
{
    delete[] weights;
    delete[] biases;
    delete[] grad_weights;
    delete[] grad_biases;
}

void initialize_dense(float *weights, float *biases, int rows, int cols, std::mt19937 &gen)
{
    std::uniform_real_distribution<float> dis(-1.0, 1.0); // Uniform distribution
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i * cols + j] = dis(gen); // Random value between -1 and 1
        }
    }

    for (int j = 0; j < cols; ++j)
    {
        biases[j] = 0; // Set biases to 0
    }   
}

// Forward pass
void Dense::forward(const float *input, float *output) const
{
    matmul(input, weights, output, this->batch_size, this->input_size, this->output_size);
    for (int i = 0; i < this->batch_size; ++i)
    {
        for (int j = 0; j < this->output_size; ++j)
        {
            output[i * output_size + j] += this->biases[j];
        }
    }

    // Apply activation function if specified
    if (activation_type == "relu")
        relu(output, this->batch_size, this->output_size);
    else if (activation_type == "softmax")
        softmax(output, this->batch_size, this->output_size);
    else if (activation_type != "none")
        std::cerr << "Error: Unsupported activation type \"" << activation_type << "\". Supported types are: relu, softmax, none.\n";
}


// Backward pass
void Dense::backward(const float *input, const float *grad_output, float *grad_input)
{
    // Reset gradients
    std::fill(grad_weights, grad_weights + input_size * output_size, 0);
    std::fill(grad_biases, grad_biases + output_size, 0);

    // Gradient of biases
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            grad_biases[j] += grad_output[i * output_size + j];
        }
    }

    // Gradient of weights
    if (input != nullptr)
    {
        for (int i = 0; i < batch_size; ++i)
        {
            for (int j = 0; j < output_size; ++j)
            {
                for (int k = 0; k < input_size; ++k)
                {
                    grad_weights[k * output_size + j] += input[i * input_size + k] * grad_output[i * output_size + j];
                }
            }
        }
    }

    // Apply activation derivative
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            int index = i * output_size + j;
            if (activation_type == "relu")
                grad_output_with_activation[index] = grad_output[index] * (output[index] > 0 ? 1.0f : 0.0f);
            else if (activation_type == "softmax")
                grad_output_with_activation[index] = /* Add proper softmax derivative */;
            else
                grad_output_with_activation[index] = grad_output[index];
        }
    }

    // Gradient of input
    matmul(grad_output_with_activation, weights, grad_input, batch_size, output_size, input_size);
}


void Dense::update_weights(float learning_rate) {
    for (int i = 0; i < input_size * output_size; ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }
    for (int i = 0; i < output_size; ++i) {
        biases[i] -= learning_rate * grad_biases[i];
    }
}

void model_forward(const float *input, float *output, Dense* layers, int num_dense)
{

    // Allocate a temporary array for intermediate results
    float *x = new float[layers[0].get_batch_size() * layers[0].get_output_size()];
    layers[0].forward(input, x);

    for (int i = 1; i < num_dense; ++i)
    {
        int batch_size = layers[i].get_batch_size();
        int output_size = layers[i].get_output_size();

        float *tmp_x = new float[batch_size * output_size];
        layers[i].forward(x, tmp_x);

        delete[] x;
        x = tmp_x;
    }

    // Copy the final result to the output array
    int final_batch_size = layers[num_dense - 1].get_batch_size();
    int final_output_size = layers[num_dense - 1].get_output_size();
    std::memcpy(output, x, sizeof(float) * final_batch_size * final_output_size);
    // Free memory
    delete[] x;
}

void model_backward(const float *input, int input_size, const float *output_grad, Dense* layers, int num_dense, float *input_grad)
{
    // Allocate memory for intermediate gradients
    float *current_grad = new float[layers[num_dense - 1].get_batch_size() * layers[num_dense - 1].get_output_size()];
    std::memcpy(current_grad, output_grad, sizeof(float) * layers[num_dense - 1].get_batch_size() * layers[num_dense - 1].get_output_size());

    for (int i = num_dense - 1; i >= 0; --i)
    {
        float *prev_grad = new float[layers[i].get_batch_size() * layers[i].get_input_size()];
        layers[i].backward((i == 0 ? input : nullptr), current_grad, prev_grad);

        delete[] current_grad;
        current_grad = prev_grad;
    }

    // Copy the final gradient to the input_grad array
    std::memcpy(input_grad, current_grad, sizeof(float) * layers[0].get_batch_size() * layers[0].get_input_size());

    // Free memory
    delete[] current_grad;
}
