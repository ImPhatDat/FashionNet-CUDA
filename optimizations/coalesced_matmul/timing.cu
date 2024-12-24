#include <stdio.h>
#include <stdint.h>
#include <random>
#include <chrono>
#include <getopt.h>
#include "../../src_parallel/layer/dense.hh"

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

std::mt19937 global_rng(1); // Random number generator

bool checkCorrect(float* a, float* b, int shape1, int shape2) {
    float* tmpa = new float[shape1 * shape2];
    CHECK(cudaMemcpy(tmpa, a, sizeof(float) * shape1 * shape2, cudaMemcpyDeviceToHost));
    float* tmpb = new float[shape1 * shape2];
    CHECK(cudaMemcpy(tmpb, b, sizeof(float) * shape1 * shape2, cudaMemcpyDeviceToHost));

    bool res = true;
    for (int i = 0; i < shape1; i++) {
        for (int j = 0; j < shape2; j++) {
            float aa = tmpa[i * shape2 + j];
            float bb = tmpb[i * shape2 + j];
            if (aa != bb) {
                printf("Missmatch at %d,%d: %f vs %f\n", i, j, aa, bb);
                res = false;
            }
        }
    }

    delete[] tmpa;
    delete[] tmpb;
    return res;
}

// Model configurations
int main(int argc, char **argv)
{
    printDeviceInfo();
    
    int batch_size = 64;
    int input_size = 784;
    int output_size = 128;
    
    Dense layer_to_time(batch_size, input_size, 128, true, global_rng);
    
    float* random_input1 = new float[batch_size * input_size];
    for (int i = 0; i < batch_size * input_size; i++) {
        random_input1[i] = i;
    }

    float* random_input2 = new float[input_size * output_size];
    for (int i = 0; i < batch_size * input_size; i++) {
        random_input2[i] = i;
    }

    float* input_d;
    CHECK(cudaMalloc(&input_d, sizeof(float) * batch_size * input_size));
    CHECK(cudaMemcpy(input_d, random_input1, sizeof(float) * batch_size * input_size, cudaMemcpyHostToDevice));

    float* input_d2;
    CHECK(cudaMalloc(&input_d2, sizeof(float) *  input_size * output_size));
    CHECK(cudaMemcpy(input_d2, random_input1, sizeof(float) *  input_size * output_size, cudaMemcpyHostToDevice));

    GpuTimer timer;
    float* output_d_1;
    CHECK(cudaMalloc(&output_d_1, sizeof(float) * batch_size * output_size));
    timer.Start();
    layer_to_time.matmul(input_d, input_d2, output_d_1, batch_size, input_size, output_size, dim3(32, 32));
    timer.Stop();
    printf("Verion 0 time: %f ms\n", timer.Elapsed());

    layer_to_time.version = 1;
    float* output_d_2;
    CHECK(cudaMalloc(&output_d_2, sizeof(float) * batch_size * output_size));
    timer.Start();
    layer_to_time.matmul(input_d, input_d2, output_d_2, batch_size, input_size, output_size, dim3(32, 32));
    timer.Stop();
    printf("Verion 1 time: %f ms\n", timer.Elapsed());

    bool is_correct = checkCorrect(output_d_1, output_d_2, batch_size, output_size);
    printf("Correct? %s\n", is_correct ? "true" : "false");


    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d_1));
    CHECK(cudaFree(output_d_2));


    delete[] random_input1;
    delete[] random_input2;

    return 0;
}
