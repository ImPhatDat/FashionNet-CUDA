#include <stdio.h>
#include <stdint.h>
#include <random>
#include "framework.cu"
#include "dense.cu"
#include "fashion_mnist.cu"

std::mt19937 global_rng(1); // Random number generator

// Model configurations
const int INPUT_SIZE = 784;   // Example: MNIST image input size
const int BATCH_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int DENSE_OUTPUT[] = {128, 128, OUTPUT_SIZE};
const int NUM_LAYERS = sizeof(DENSE_OUTPUT) / sizeof(DENSE_OUTPUT[0]);
const std::string ACTIVATION_TYPES[] = {"relu", "relu", "softmax"};

const std::string train_imageFilePath = "../data/fashion-mnist/train-images-idx3-ubyte";
const std::string train_labelFilePath = "../data/fashion-mnist/train-labels-idx1-ubyte";
const std::string test_imageFilePath = "../data/fashion-mnist/t10k-images-idx3-ubyte";
const std::string test_labelFilePath = "../data/fashion-mnist/t10k-labels-idx1-ubyte";


#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
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


int main(int argc, char ** argv) {
    printDeviceInfo();
    
    // Load dataset
    FashionMnist train_set;
    train_set.loadDataset(train_imageFilePath, train_labelFilePath);
    FashionMnist test_set;
    test_set.loadDataset(test_imageFilePath, test_labelFilePath);
    std::cout << "Total train images: " << train_set.getImageCount() << std::endl;
    std::cout << "Total test images: " << test_set.getImageCount() << std::endl;

    size_t total_images = train_set.getImageCount();
    size_t num_batches = total_images / BATCH_SIZE;

    // Prepare batches
    train_set.shuffle(global_rng);

    float** x_batches = nullptr;
    uint8_t** y_batches = nullptr;

    train_set.prepareBatchesWithLabels(BATCH_SIZE, INPUT_SIZE, x_batches, y_batches);

    float** output_batches = new float*[num_batches];
    for (int bi = 0; bi < num_batches; ++bi) {
        output_batches[bi] = new float[BATCH_SIZE * OUTPUT_SIZE];
    }

    // std::cout << y_batches[0] << std::endl;

    // Allocate for Dense layers
    Dense* layers = new Dense[NUM_LAYERS];
    int previous_size = INPUT_SIZE;

    for (size_t i = 0; i < NUM_LAYERS; ++i) {
        layers[i] = Dense(
            previous_size,               // input size 
            DENSE_OUTPUT[i],             // output size
            BATCH_SIZE,                  // batch size
            ACTIVATION_TYPES[i],         // activation type
            global_rng                   // random number generator
        );
        previous_size = DENSE_OUTPUT[i]; // Update input size for next layer
    }

    for (int bi = 0; bi < num_batches; ++bi) {
        model_forward(x_batches[bi], output_batches[bi], layers, NUM_LAYERS);
    }

    // Deallocate
    for (size_t i = 0; i < num_batches; ++i) {
        delete[] x_batches[i];
        delete[] output_batches[i];
        delete[] y_batches[i];
    }
    delete[] x_batches;
    delete[] output_batches;
    delete[] y_batches;


    delete[] layers;
    return 0;
}
