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


// check correct with host code
bool checkCorrect(float* a, float* b, int shape1, int shape2) {
    float* tmpa = new float[shape1 * shape2];
    CHECK(cudaMemcpy(tmpa, a, sizeof(float) * shape1 * shape2, cudaMemcpyDeviceToHost));
    // float* tmpb = new float[shape1 * shape2];
    // CHECK(cudaMemcpy(tmpb, b, sizeof(float) * shape1 * shape2, cudaMemcpyDeviceToHost));

    bool res = true;
    for (int i = 0; i < shape1; i++) {
        for (int j = 0; j < shape2; j++) {
            float aa = tmpa[i * shape2 + j];
            float bb = b[i * shape2 + j];
            if (std::abs(aa - bb) >= 1e-3) {
                printf("Missmatch at %d,%d: %f vs %f\n", i, j, aa, bb);
                res = false;
            }
        }
    }
    delete[] tmpa;
    return res;
}

void matmul_host(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


// Model configurations
int main(int argc, char **argv)
{
    printDeviceInfo();
    
    int batch_size = 1024;
    int input_size = 512;
    int output_size = 256;
    
    Dense layer_to_time(batch_size, input_size, output_size, true, global_rng);
    
    std::uniform_real_distribution<float> dis(-1, 1);

    float* random_input1 = new float[batch_size * input_size];
    for (int i = 0; i < batch_size * input_size; i++) {
        random_input1[i] = dis(global_rng);
    }

    float* random_input2 = new float[input_size * output_size];
    for (int i = 0; i < input_size * output_size; i++) {
        random_input2[i] = dis(global_rng);
    }

    // host matmul
    float * output_host = new float[batch_size * output_size];
    matmul_host(random_input1, random_input2, output_host, batch_size, input_size, output_size);

    float* input_d;
    CHECK(cudaMalloc(&input_d, sizeof(float) * batch_size * input_size));
    CHECK(cudaMemcpy(input_d, random_input1, sizeof(float) * batch_size * input_size, cudaMemcpyHostToDevice));

    float* input_d2;
    CHECK(cudaMalloc(&input_d2, sizeof(float) *  input_size * output_size));
    CHECK(cudaMemcpy(input_d2, random_input2, sizeof(float) *  input_size * output_size, cudaMemcpyHostToDevice));

    GpuTimer timer;


    layer_to_time.version = 0;
    float* output_d_0;
    CHECK(cudaMalloc(&output_d_0, sizeof(float) * batch_size * output_size));
    timer.Start();
    layer_to_time.matmul(input_d, input_d2, output_d_0, batch_size, input_size, output_size, dim3(32, 32));
    timer.Stop();
    // printf("Verion 0 time: %f ms\n", timer.Elapsed());

    layer_to_time.version = 1;
    float* output_d_1;
    CHECK(cudaMalloc(&output_d_1, sizeof(float) * batch_size * output_size));
    timer.Start();
    layer_to_time.matmul(input_d, input_d2, output_d_1, batch_size, input_size, output_size, dim3(32, 32));
    timer.Stop();
    // printf("Verion 1 time: %f ms\n", timer.Elapsed());


    layer_to_time.version = 2;
    float* output_d_2;
    CHECK(cudaMalloc(&output_d_2, sizeof(float) * batch_size * output_size));
    timer.Start();
    layer_to_time.matmul(input_d, input_d2, output_d_2, batch_size, input_size, output_size, dim3(32, 32));
    timer.Stop();
    // printf("Verion 2 time: %f ms\n", timer.Elapsed());

    float avg_time0 = 0;
    float avg_time1 = 0;
    float avg_time2 = 0;
    int num_runs = 50;
    for (int i = 0; i < num_runs; i++) {
        layer_to_time.version = 0;
        timer.Start();
        layer_to_time.matmul(input_d, input_d2, output_d_0, batch_size, input_size, output_size, dim3(32, 32));
        timer.Stop();
        avg_time0 += timer.Elapsed();

        layer_to_time.version = 1;
        timer.Start();
        layer_to_time.matmul(input_d, input_d2, output_d_1, batch_size, input_size, output_size, dim3(32, 32));
        timer.Stop();
        avg_time1 += timer.Elapsed();

        layer_to_time.version = 2;
        timer.Start();
        layer_to_time.matmul(input_d, input_d2, output_d_2, batch_size, input_size, output_size, dim3(32, 32));
        timer.Stop();
        avg_time2 += timer.Elapsed();
    }

    avg_time0 /= num_runs;
    avg_time1 /= num_runs;
    avg_time2 /= num_runs;
    printf("Num runs: %d\n", num_runs);
    printf("Average time version 0 (parallel): %f ms\n", avg_time0);
    printf("Average time version 1 (shared mem): %f ms\n", avg_time1);
    printf("Average time version 2 (shared mem + unrolling): %f ms\n", avg_time2);


    bool is_correct0 = checkCorrect(output_d_0, output_host, batch_size, output_size);
    bool is_correct1 = checkCorrect(output_d_1, output_host, batch_size, output_size);
    bool is_correct2 = checkCorrect(output_d_2, output_host, batch_size, output_size);
    printf("Correct 0? %s\n", is_correct0 ? "true" : "false");
    printf("Correct 1? %s\n", is_correct1 ? "true" : "false");
    printf("Correct 2? %s\n", is_correct2 ? "true" : "false");



    CHECK(cudaFree(input_d));
    CHECK(cudaFree(input_d2));
    CHECK(cudaFree(output_d_0));
    CHECK(cudaFree(output_d_1));
    CHECK(cudaFree(output_d_2));



    delete[] random_input1;
    delete[] random_input2;
    delete[] output_host;
    return 0;
}
