#include <stdio.h>
#include <stdint.h>
#include <random>
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
const int BATCH_SIZE = 64;
const int OUTPUT_SIZE = 10;

const float LEARNING_RATE = 0.001;

const int NUM_EPOCHS = 10;

Layer *layers[] = {
    new Dense(BATCH_SIZE, INPUT_SIZE, 128, global_rng),
    new ReLU(BATCH_SIZE, 128),
    new Dense(BATCH_SIZE, 128, 128, global_rng),
    new ReLU(BATCH_SIZE, 128),
    new Dense(BATCH_SIZE, 128, OUTPUT_SIZE, global_rng),
    new Softmax(BATCH_SIZE, OUTPUT_SIZE)};

const int NUM_LAYERS = sizeof(layers) / sizeof(layers[0]);

int main(int argc, char **argv)
{
    printDeviceInfo();

    // Load dataset
    FashionMnist train_set;
    train_set.loadDataset(train_imageFilePath, train_labelFilePath);
    FashionMnist test_set;
    test_set.loadDataset(test_imageFilePath, test_labelFilePath);
    std::cout << "Total train images: " << train_set.getImageCount() << std::endl;
    std::cout << "Total test images: " << test_set.getImageCount() << std::endl;

    int num_batches = train_set.getImageCount() / BATCH_SIZE;
    float **x_batches = new float*[num_batches];
    uint8_t **y_batches = new uint8_t*[num_batches];
    float **y_pred_batches = new float *[num_batches];
    for (int bi = 0; bi < num_batches; ++bi) {
        x_batches[bi] = new float[BATCH_SIZE * INPUT_SIZE];
        y_batches[bi] = new uint8_t[BATCH_SIZE];
        y_pred_batches[bi] = new float[BATCH_SIZE * OUTPUT_SIZE];
    }


    int test_num_batches = test_set.getImageCount() / BATCH_SIZE;
    float **test_x_batches = new float*[test_num_batches];
    uint8_t **test_y_batches = new uint8_t*[test_num_batches];
    float **test_y_pred_batches = new float *[test_num_batches];
    for (int bi = 0; bi < test_num_batches; ++bi) {
        test_x_batches[bi] = new float[BATCH_SIZE * INPUT_SIZE];
        test_y_batches[bi] = new uint8_t[BATCH_SIZE];
        test_y_pred_batches[bi] = new float[BATCH_SIZE * OUTPUT_SIZE];
    }
    test_set.prepareBatchesWithLabels(BATCH_SIZE, INPUT_SIZE, test_x_batches, test_y_batches);


    Model model(layers, NUM_LAYERS, BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE);
    CategoricalCrossentropy loss_obj(1e-7);
    Accuracy acc_obj;
    float loss_batch;
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        printf("====================Epoch (%d/%d)====================\n", epoch + 1, NUM_EPOCHS);
        // reshuffle train after each epoch
        train_set.shuffle(global_rng);
        train_set.prepareBatchesWithLabels(BATCH_SIZE, INPUT_SIZE, x_batches, y_batches);
        loss_obj.reset_state();
        acc_obj.reset_state();
        for (int bi = 0; bi < num_batches; ++bi)
        {
            model.forward(x_batches[bi], y_pred_batches[bi]);

            loss_batch = loss_obj.forward(y_batches[bi], y_pred_batches[bi], BATCH_SIZE, OUTPUT_SIZE);
            loss_obj.update_state(loss_batch);
            acc_obj.update_state(y_pred_batches[bi], y_batches[bi], BATCH_SIZE, OUTPUT_SIZE);

            model.backward(y_batches[bi], y_pred_batches[bi], &loss_obj);
            model.update_weights(LEARNING_RATE);

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
            loss_batch = loss_obj.forward(test_y_batches[bi], test_y_pred_batches[bi], BATCH_SIZE, OUTPUT_SIZE);
            loss_obj.update_state(loss_batch);
            acc_obj.update_state(test_y_pred_batches[bi], test_y_batches[bi], BATCH_SIZE, OUTPUT_SIZE);
        }
        printf("Validation: loss - %f, acc - %f\n", loss_obj.compute_average_loss(), acc_obj.compute());
    }

    // Deallocate
    for (size_t i = 0; i < num_batches; ++i)
    {
        delete[] x_batches[i];
        delete[] y_pred_batches[i];
        delete[] y_batches[i];
    }
    delete[] x_batches;
    delete[] y_pred_batches;
    delete[] y_batches;

    for (size_t i = 0; i < test_num_batches; ++i)
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
