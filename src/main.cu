#include <stdio.h>
#include <stdint.h>
#include <random>
#include <chrono>
#include <getopt.h>
#include "fashion_mnist.h"
#include "Model/Model.h"
#include "layer/dense.h"
#include "layer/relu.h"
#include "layer/softmax.h"
#include "loss/categorical_crossentropy.h"
#include "metrics/accuracy.h"
#include "helpers.cu"

const std::string train_imageFilePath = "data/fashion-mnist/train-images-idx3-ubyte";
const std::string train_labelFilePath = "data/fashion-mnist/train-labels-idx1-ubyte";
const std::string test_imageFilePath = "data/fashion-mnist/t10k-images-idx3-ubyte";
const std::string test_labelFilePath = "data/fashion-mnist/t10k-labels-idx1-ubyte";

std::mt19937 global_rng(1); // Random number generator
// Model configurations
const int INPUT_SIZE = 784; // Example: MNIST image input size
const int OUTPUT_SIZE = 10;

int main(int argc, char **argv)
{
    int num_epoch = 10;
    int batch_size = 64; // Default value
    float learning_rate = 0.001;
    int opt;

    // Parsing command-line arguments
    while ((opt = getopt(argc, argv, "e:b:l:")) != -1)
    {
        switch (opt)
        {
        case 'e':
            num_epoch = atoi(optarg); // Convert argument to integer
            break;
        case 'b':
            batch_size = atoi(optarg); // Convert argument to integer
            break;
        case 'l':
            learning_rate = atof(optarg); // Convert argument to integer
            break;
        default:
            fprintf(stderr, "Usage: %s [-e num_epoch] [-b batchsize] [-l learning_rate]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    printDeviceInfo();

    Layer *layers[] = {
        new Dense(batch_size, INPUT_SIZE, 128, global_rng),
        new ReLU(batch_size, 128),
        new Dense(batch_size, 128, 128, global_rng),
        new ReLU(batch_size, 128),
        new Dense(batch_size, 128, OUTPUT_SIZE, global_rng),
        new Softmax(batch_size, OUTPUT_SIZE)};
    const int NUM_LAYERS = sizeof(layers) / sizeof(layers[0]);

    // Load dataset
    FashionMnist train_set;
    train_set.loadDataset(train_imageFilePath, train_labelFilePath);
    FashionMnist test_set;
    test_set.loadDataset(test_imageFilePath, test_labelFilePath);
    std::cout << "Total train images: " << train_set.getImageCount() << std::endl;
    std::cout << "Total test images: " << test_set.getImageCount() << std::endl;

    int num_batches = train_set.getImageCount() / batch_size;
    float **x_batches = new float *[num_batches];
    uint8_t **y_batches = new uint8_t *[num_batches];
    float **y_pred_batches = new float *[num_batches];
    for (int bi = 0; bi < num_batches; ++bi)
    {
        x_batches[bi] = new float[batch_size * INPUT_SIZE];
        y_batches[bi] = new uint8_t[batch_size];
        y_pred_batches[bi] = new float[batch_size * OUTPUT_SIZE];
    }

    int test_num_batches = test_set.getImageCount() / batch_size;
    float **test_x_batches = new float *[test_num_batches];
    uint8_t **test_y_batches = new uint8_t *[test_num_batches];
    float **test_y_pred_batches = new float *[test_num_batches];
    for (int bi = 0; bi < test_num_batches; ++bi)
    {
        test_x_batches[bi] = new float[batch_size * INPUT_SIZE];
        test_y_batches[bi] = new uint8_t[batch_size];
        test_y_pred_batches[bi] = new float[batch_size * OUTPUT_SIZE];
    }
    test_set.prepareBatchesWithLabels(batch_size, INPUT_SIZE, test_x_batches, test_y_batches);

    Model model(layers, NUM_LAYERS, batch_size, INPUT_SIZE, OUTPUT_SIZE);
    CategoricalCrossentropy loss_obj(1e-7);
    Accuracy acc_obj;
    float loss_batch;

    std::cout << "\nConfigurations:" << std::endl;
    std::cout << "\tNum epoch: " << num_epoch << std::endl;
    std::cout << "\tBatch size: " << batch_size << std::endl;
    std::cout << "\tLearning rate: " << learning_rate << std::endl;

    HostTimer epoch_timer;
    for (int epoch = 0; epoch < num_epoch; epoch++)
    {
        // Start timing
        epoch_timer.Start();

        printf("====================Epoch (%d/%d)====================\n", epoch + 1, num_epoch);
        // reshuffle train after each epoch
        train_set.shuffle(global_rng);
        train_set.prepareBatchesWithLabels(batch_size, INPUT_SIZE, x_batches, y_batches);
        loss_obj.reset_state();
        acc_obj.reset_state();
        for (int bi = 0; bi < num_batches; ++bi)
        {
            model.forward(x_batches[bi], y_pred_batches[bi]);

            loss_batch = loss_obj.forward(y_batches[bi], y_pred_batches[bi], batch_size, OUTPUT_SIZE);
            loss_obj.update_state(loss_batch);
            acc_obj.update_state(y_pred_batches[bi], y_batches[bi], batch_size, OUTPUT_SIZE);

            model.backward(y_batches[bi], y_pred_batches[bi], &loss_obj);
            model.update_weights(learning_rate);

            if (bi % 100 == 0 || bi == num_batches - 1)
            {
                printf("Iter (%d/%d): loss - %f, acc - %f\n",
                       bi, num_batches - 1,
                       loss_obj.compute_average_loss(), acc_obj.compute());
            }
        }
        loss_obj.reset_state();
        acc_obj.reset_state();

        for (int bi = 0; bi < test_num_batches; ++bi)
        {
            model.forward(test_x_batches[bi], test_y_pred_batches[bi]);
            loss_batch = loss_obj.forward(test_y_batches[bi], test_y_pred_batches[bi], batch_size, OUTPUT_SIZE);
            loss_obj.update_state(loss_batch);
            acc_obj.update_state(test_y_pred_batches[bi], test_y_batches[bi], batch_size, OUTPUT_SIZE);
        }
        printf("Validation: loss - %f, acc - %f\n", loss_obj.compute_average_loss(), acc_obj.compute());

        // Stop timing
        epoch_timer.Stop();

        // Get and print the elapsed time
        printf("Epoch time: %f seconds\n", epoch_timer.Elapsed());
    }

    // Deallocate
    for (int i = 0; i < num_batches; ++i)
    {
        delete[] x_batches[i];
        delete[] y_pred_batches[i];
        delete[] y_batches[i];
    }
    delete[] x_batches;
    delete[] y_pred_batches;
    delete[] y_batches;

    for (int i = 0; i < test_num_batches; ++i)
    {
        delete[] test_x_batches[i];
        delete[] test_y_pred_batches[i];
        delete[] test_y_batches[i];
    }
    delete[] test_x_batches;
    delete[] test_y_pred_batches;
    delete[] test_y_batches;
    return 0;
}
