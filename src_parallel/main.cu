#include <stdio.h>
#include <stdint.h>
#include <random>
#include <chrono>
#include <getopt.h>
#include "utils/fashion_mnist.hh"
#include "utils/helpers.hh"
#include "Model/Model.hh"
#include "layer/dense.hh"
#include "layer/relu.hh"
#include "layer/softmax.hh"
#include "loss/categorical_crossentropy.hh"
#include "metrics/accuracy.hh"

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

struct HostTimer
{
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point stop;

    // Start the timer
    void Start()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    // Stop the timer
    void Stop()
    {
        stop = std::chrono::high_resolution_clock::now();
    }

    // Get the elapsed time in milliseconds
    float Elapsed()
    {
        std::chrono::duration<float> duration = stop - start;
        return duration.count(); // Returns elapsed time in milliseconds
    }
};


unsigned long seed = 1;
std::mt19937 global_rng(1); // Random number generator

// Model configurations
const int INPUT_SIZE = 784; // Example: MNIST image input size
const int OUTPUT_SIZE = 10;

int main(int argc, char **argv)
{
    printDeviceInfo();
    std::string dataset_path = "";
    int num_epoch = 10;
    int batch_size = 64; // Default value
    float learning_rate = 0.001;
    std::string checkpoint_path = "";

    int blockSize1d = 256;
    int blockSize2d_x = 32;
    int blockSize2d_y = 32;

    int opt;
    
    // Parsing command-line arguments
    while ((opt = getopt(argc, argv, "d:e:b:l:p:k:x:y:")) != -1)
    {
        switch (opt)
        {
        case 'd':
            dataset_path = optarg; // Convert argument to integer
            break;
        case 'e':
            num_epoch = atoi(optarg); // Convert argument to integer
            break;
        case 'b':
            batch_size = atoi(optarg); // Convert argument to integer
            break;
        case 'l':
            learning_rate = atof(optarg); // Convert argument to float
            break;
        case 'p':
            checkpoint_path = optarg; // Store the checkpoint path
            break;
        case 'k':
            blockSize1d = atoi(optarg); // Convert argument to integer
            break;
        case 'x':
            blockSize2d_x = atoi(optarg); // Convert argument to integer
            break;
        case 'y':
            blockSize2d_y = atoi(optarg); // Convert argument to integer
            break;
        default:
            fprintf(stderr, "Usage: %s [-d dataset_path] [-e num_epoch] [-b batchsize] [-l learning_rate] [-p checkpoint_path] [-k blockSize1d] [-x blockSize2d_x] [-y blockSize2d_y]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "\nConfigurations:" << std::endl;
    std::cout << "\tDataset path: " << dataset_path << std::endl;
    std::cout << "\tNum epoch: " << num_epoch << std::endl;
    std::cout << "\tBatch size: " << batch_size << std::endl;
    std::cout << "\tLearning rate: " << learning_rate << std::endl;
    std::cout << "\tCheckpoint: " << checkpoint_path << std::endl;
    std::cout << "\tBlockSize 1D: " << blockSize1d << std::endl;
    std::cout << "\tBlockSize 2D_x: " << blockSize2d_x << std::endl;
    std::cout << "\tBlockSize 2D_y: " << blockSize2d_y << std::endl;


    const std::string train_imageFilePath = dataset_path + "train-images-idx3-ubyte";
    const std::string train_labelFilePath = dataset_path + "train-labels-idx1-ubyte";
    const std::string test_imageFilePath = dataset_path + "t10k-images-idx3-ubyte";
    const std::string test_labelFilePath = dataset_path + "t10k-labels-idx1-ubyte";

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
    for (int bi = 0; bi < num_batches; ++bi)
    {
        x_batches[bi] = new float[batch_size * INPUT_SIZE];
        y_batches[bi] = new uint8_t[batch_size];
    }

    int test_num_batches = test_set.getImageCount() / batch_size;
    float **test_x_batches = new float *[test_num_batches];
    uint8_t **test_y_batches = new uint8_t *[test_num_batches];
    for (int bi = 0; bi < test_num_batches; ++bi)
    {
        test_x_batches[bi] = new float[batch_size * INPUT_SIZE];
        test_y_batches[bi] = new uint8_t[batch_size];
    }
    test_set.prepareBatchesWithLabels(batch_size, INPUT_SIZE, test_x_batches, test_y_batches);

    Layer *layers[] = {
        new Dense(batch_size, INPUT_SIZE, 128, dim3(blockSize1d), true, seed),
        new ReLU(batch_size, 128),
        new Dense(batch_size, 128, 128, dim3(blockSize1d), true, seed),
        new ReLU(batch_size, 128),
        new Dense(batch_size, 128, OUTPUT_SIZE, dim3(blockSize1d), true, seed),
        new Softmax(batch_size, OUTPUT_SIZE)};

    dim3 blockSizes[] = {
        dim3(blockSize2d_x, blockSize2d_y),
        dim3(blockSize1d),
        dim3(blockSize2d_x, blockSize2d_y),
        dim3(blockSize1d),
        dim3(blockSize2d_x, blockSize2d_y),
        dim3(blockSize1d),
    };

    dim3 loss_blockSize = dim3(blockSize1d);

    const int NUM_LAYERS = sizeof(layers) / sizeof(layers[0]);

    Model model(layers, NUM_LAYERS, batch_size, INPUT_SIZE, OUTPUT_SIZE);
    CategoricalCrossentropy loss_obj(1e-7);
    Accuracy acc_obj;
    float loss_batch;

    HostTimer epoch_timer;
    HostTimer total_timer;
    total_timer.Start();

    //tmp malloc
    uint8_t * d_y_true;
    CHECK(cudaMalloc(&d_y_true, sizeof(uint8_t) * batch_size));
    float * d_y_pred;
    CHECK(cudaMalloc(&d_y_pred, sizeof(float) * batch_size * OUTPUT_SIZE));
    float* h_y_pred = new float[batch_size * OUTPUT_SIZE];

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
            model.forward(x_batches[bi], d_y_pred, blockSizes);
            CHECK(cudaMemcpy(d_y_true, y_batches[bi], sizeof(uint8_t) * batch_size, cudaMemcpyHostToDevice));
            
            loss_batch = loss_obj.forward(d_y_true, d_y_pred, batch_size, OUTPUT_SIZE, loss_blockSize);
            loss_obj.update_state(loss_batch);

            CHECK(cudaMemcpy(h_y_pred, d_y_pred, sizeof(float) * batch_size * OUTPUT_SIZE, cudaMemcpyDeviceToHost));
            acc_obj.update_state(h_y_pred, y_batches[bi], batch_size, OUTPUT_SIZE);

            model.backward(d_y_true, d_y_pred, blockSizes, &loss_obj, loss_blockSize);

            model.update_weights(learning_rate, blockSizes);

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
            model.forward(test_x_batches[bi], d_y_pred, blockSizes);

            CHECK(cudaMemcpy(d_y_true, test_y_batches[bi], sizeof(uint8_t) * batch_size, cudaMemcpyHostToDevice));
            loss_batch = loss_obj.forward(d_y_true, d_y_pred, batch_size, OUTPUT_SIZE, loss_blockSize);
            loss_obj.update_state(loss_batch);
            
            CHECK(cudaMemcpy(h_y_pred, d_y_pred, sizeof(float) * batch_size * OUTPUT_SIZE, cudaMemcpyDeviceToHost));
            acc_obj.update_state(h_y_pred, test_y_batches[bi], batch_size, OUTPUT_SIZE);
        }
        printf("Validation: loss - %f, acc - %f\n", loss_obj.compute_average_loss(), acc_obj.compute());

        // Stop timing
        epoch_timer.Stop();

        // Get and print the elapsed time
        printf("Epoch time: %f seconds\n", epoch_timer.Elapsed());
    }
    total_timer.Stop();
    printf("Total time: %f seconds\n", total_timer.Elapsed());

    if (checkpoint_path != "")
        model.save_weights(checkpoint_path);

    // DONT DELETE COMMENTED CODE BELOW  (for verify)

    // Layer *layers2[] = {
    // new Dense(batch_size, INPUT_SIZE, 128, global_rng),
    // new ReLU(batch_size, 128),
    // new Dense(batch_size, 128, 128, global_rng),
    // new ReLU(batch_size, 128),
    // new Dense(batch_size, 128, OUTPUT_SIZE, global_rng),
    // new Softmax(batch_size, OUTPUT_SIZE)};
    // Model model2(layers2, NUM_LAYERS, batch_size, INPUT_SIZE, OUTPUT_SIZE);
    // model2.load_weights("weight_ne.bin");
    // float* tmp_batch = new float[batch_size * OUTPUT_SIZE];
    // model2.forward(test_x_batches[0], tmp_batch);
    // std::cout << "Ori preds" << std::endl;
    // for (int ii = 0; ii < batch_size; ii++) {
    //     for (int jj = 0; jj < OUTPUT_SIZE; jj++) {
    //         std::cout << test_y_pred_batches[0][ii * OUTPUT_SIZE + jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Loaded preds" << std::endl;
    // for (int ii = 0; ii < batch_size; ii++) {
    //     for (int jj = 0; jj < OUTPUT_SIZE; jj++) {
    //         std::cout << tmp_batch[ii * OUTPUT_SIZE + jj] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // delete[] tmp_batch;

    // Deallocate
    CHECK(cudaFree(d_y_true));
    CHECK(cudaFree(d_y_pred));
    delete[] h_y_pred;
    for (int i = 0; i < num_batches; ++i)
    {
        delete[] x_batches[i];
        delete[] y_batches[i];
    }
    delete[] x_batches;
    delete[] y_batches;

    for (int i = 0; i < test_num_batches; ++i)
    {
        delete[] test_x_batches[i];
        delete[] test_y_batches[i];
    }
    delete[] test_x_batches;
    delete[] test_y_batches;
    return 0;
}
