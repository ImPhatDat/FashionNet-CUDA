#include "Model.hh"

Model::Model() {
    this->layers = nullptr;
    this->num_layers = 0;
}
Model::Model(Layer* layers[], int num_layers, int batch_size, int input_size, int num_classes) {
    this->num_layers = num_layers;
    this->batch_size = batch_size;
    this->input_size = input_size;
    this->num_classes = num_classes;

    this->layers = new Layer*[num_layers];
    for (int i = 0; i < num_layers; i++) {
        this->layers[i] = layers[i];
    }
}
Model::~Model() {
    for (int i = 0; i < num_layers; i++) {
        delete this->layers[i];
    }
    delete[] this->layers;
}


void Model::forward(const float* batch_input, float* batch_output) {
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

void Model::backward(const uint8_t* y_true, const float* y_pred, Loss* loss) {
    float* grad_x = new float[this->batch_size * this->num_classes];
    loss->backward(y_true, y_pred, this->batch_size, this->num_classes, grad_x);

    for (int i = num_layers - 1; i >= 0; i--) {
        float* grad_tmp = new float[this->batch_size * this->layers[i]->input_size];
        this->layers[i]->backward(grad_x, grad_tmp);
        
        delete[] grad_x;
        grad_x = grad_tmp;
    }

    delete[] grad_x;
}

void Model::update_weights(const float learning_rate) {
    for (int i = 0; i < this->num_layers; i++) {
        this->layers[i]->update_weights(learning_rate);
    }
}

void Model::save_weights(std::string path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    // Save the number of layers
    file.write(reinterpret_cast<const char*>(&this->num_layers), sizeof(this->num_layers));

    for (int l = 0; l < this->num_layers; l++) {
        float* weights = this->layers[l]->get_weights();
        float* biases = this->layers[l]->get_biases();
        int input_size = this->layers[l]->input_size;
        int output_size = this->layers[l]->output_size;

        // Save the dimensions of the weight matrix
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        // Indicate if weights are present
        bool has_weights = weights != nullptr;
        file.write(reinterpret_cast<const char*>(&has_weights), sizeof(has_weights));
        if (has_weights) {
            file.write(reinterpret_cast<const char*>(weights), input_size * output_size * sizeof(float));
        }

        // Indicate if biases are present
        bool has_biases = biases != nullptr;
        file.write(reinterpret_cast<const char*>(&has_biases), sizeof(has_biases));
        if (has_biases) {
            file.write(reinterpret_cast<const char*>(biases), output_size * sizeof(float));
        }
    }

    file.close();
    std::cout << "Model weights saved to " << path << std::endl;
}

void Model::load_weights(std::string path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    // Read the number of layers
    int num_layers_in_file;
    file.read(reinterpret_cast<char*>(&num_layers_in_file), sizeof(num_layers_in_file));

    if (num_layers_in_file != this->num_layers) {
        std::cerr << "Mismatch in number of layers: expected " 
                  << this->num_layers << ", found " 
                  << num_layers_in_file << std::endl;
        file.close();
        return;
    }

    for (int l = 0; l < this->num_layers; l++) {
        int input_size, output_size;
        file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

        if (input_size != this->layers[l]->input_size || output_size != this->layers[l]->output_size) {
            std::cerr << "Dimension mismatch in layer " << l << ": expected ("
                      << this->layers[l]->input_size << ", "
                      << this->layers[l]->output_size << "), found ("
                      << input_size << ", " << output_size << ")" << std::endl;
            file.close();
            return;
        }

        // Read if weights are present
        bool has_weights;
        file.read(reinterpret_cast<char*>(&has_weights), sizeof(has_weights));
        float* weights = nullptr;
        if (has_weights) {
            weights = new float[input_size * output_size];
            file.read(reinterpret_cast<char*>(weights), input_size * output_size * sizeof(float));
        }

        // Read if biases are present
        bool has_biases;
        file.read(reinterpret_cast<char*>(&has_biases), sizeof(has_biases));
        float* biases = nullptr;
        if (has_biases) {
            biases = new float[output_size];
            file.read(reinterpret_cast<char*>(biases), output_size * sizeof(float));
        }

        this->layers[l]->load_weights(weights, biases);

        delete[] weights; // Clean up temporary allocation
        delete[] biases; // Clean up temporary allocation
    }

    file.close();
    std::cout << "Model weights loaded successfully from " << path << std::endl;
}