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


void Model::forward(const __half* batch_input, __half* batch_output, dim3 blockSizes[]) {
    __half* x;
    CHECK(cudaMalloc(&x, sizeof(__half) * this->batch_size *  this->layers[0]->output_size));
    this->layers[0]->forward(batch_input, x, blockSizes[0]);

    __half* tmp_x;
    for (int i = 1; i < this->num_layers; i++) {
        CHECK(cudaMalloc(&tmp_x, sizeof(__half) * this->batch_size *  this->layers[i]->output_size));
        this->layers[i]->forward(x, tmp_x, blockSizes[i]);
        CHECK(cudaFree(x));
        x = tmp_x;
    }
    CHECK(cudaMemcpy(batch_output, x, sizeof(__half) * this->batch_size * this->num_classes, cudaMemcpyDeviceToDevice));
    CHECK(cudaFree(x));
}

void Model::backward(const uint8_t* y_true, const __half* y_pred, dim3 blockSizes[], Loss* loss, dim3 loss_blockSize) {
    __half* grad_x;
    CHECK(cudaMalloc(&grad_x, sizeof(__half) * this->batch_size * this->num_classes));
    loss->backward(y_true, y_pred, this->batch_size, this->num_classes, grad_x, loss_blockSize);

    __half* grad_tmp;
    for (int i = num_layers - 1; i >= 0; i--) {
        CHECK(cudaMalloc(&grad_tmp, sizeof(__half) * this->batch_size * this->layers[i]->input_size));
        this->layers[i]->backward(grad_x, grad_tmp, blockSizes[i]);
        
        // print gradient
        __half *input_d_host = new __half[batch_size * this->layers[i]->input_size];
        CHECK(cudaMemcpy(input_d_host, grad_tmp, batch_size * this->layers[i]->input_size * sizeof(__half), cudaMemcpyDeviceToHost));
        printf("=======================Layer %d\n", i);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                printf("%f ", __half2float(input_d_host[i * this->layers[i]->input_size + j]));
            }
            printf("\n");
        }
        delete[] input_d_host;
        // ends


        CHECK(cudaFree(grad_x));
        grad_x = grad_tmp;
    }

    CHECK(cudaFree(grad_x));
}

void Model::update_weights(const __half learning_rate, dim3 blockSizes[]) {
    for (int i = 0; i < this->num_layers; i++) {
        this->layers[i]->update_weights(learning_rate, dim3(256));
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
        int input_size = this->layers[l]->input_size;
        int output_size = this->layers[l]->output_size;
        // Save the dimensions of the weight matrix
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

        __half* half_weights = this->layers[l]->get_weights();
        __half* half_fbiases = this->layers[l]->get_biases();

        float* weights = new float[input_size * output_size];
        float* biases = new float[output_size];

        // Convert half-precision to float
        for (int i = 0; i < input_size * output_size; i++) {
            weights[i] = __half2float(half_weights[i]);
        }
        for (int i = 0; i < output_size; i++) {
            biases[i] = __half2float(half_fbiases[i]);
        }

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

        delete[] weights;
        delete[] biases;
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

        __half* half_weights = new __half[input_size * output_size];
        __half* half_biases = new __half[output_size];

        // Convert float to half-precision
        for (int i = 0; i < input_size * output_size; i++) {
            half_weights[i] = __float2half(weights[i]);
        }
        for (int i  =0; i < output_size; i++) {
            half_biases[i] = __float2half(biases[i]);
        }

        this->layers[l]->load_weights(half_weights, half_biases);

        delete[] weights; // Clean up temporary allocation
        delete[] biases; // Clean up temporary allocation
    }

    file.close();
    std::cout << "Model weights loaded successfully from " << path << std::endl;
}