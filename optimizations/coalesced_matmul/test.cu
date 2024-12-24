#include <stdio.h>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#define TILE_WIDTH 32
// Original implementation
__global__ void matmul_kernel_original(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Corrected implementation
__global__ void matmul_kernel_corrected(const float *A, const float *B, float *C, int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_WIDTH + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU matrix multiplication for validation
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Utility to check if two matrices are equal within epsilon
bool compare_matrices(const float* A, const float* B, int size, float epsilon = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(A[i] - B[i]) > epsilon) {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// Test structure to hold matrix dimensions
struct TestCase {
    int M, K, N;
    const char* name;
};

// Function to run test case
void run_test_case(const TestCase& test) {
    printf("\nRunning test case: %s (%dx%d * %dx%d = %dx%d)\n", 
           test.name, test.M, test.K, test.K, test.N, test.M, test.N);

    size_t size_A = test.M * test.K;
    size_t size_B = test.K * test.N;
    size_t size_C = test.M * test.N;

    // Allocate host memory
    float *h_A = new float[size_A];
    float *h_B = new float[size_B];
    float *h_C_cpu = new float[size_C];
    float *h_C_original = new float[size_C];
    float *h_C_corrected = new float[size_C];

    // Initialize input matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < size_A; i++) h_A[i] = dis(gen);
    for (size_t i = 0; i < size_B; i++) h_B[i] = dis(gen);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((test.N + TILE_WIDTH - 1) / TILE_WIDTH, 
                  (test.M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Compute reference result on CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, test.M, test.K, test.N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

    // Test original implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel_original<<<gridSize, blockSize>>>(d_A, d_B, d_C, test.M, test.K, test.N);
    cudaEventRecord(stop);
    cudaMemcpy(h_C_original, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    
    float original_ms;
    cudaEventElapsedTime(&original_ms, start, stop);

    // Test corrected implementation
    cudaEventRecord(start);
    matmul_kernel_corrected<<<gridSize, blockSize>>>(d_A, d_B, d_C, test.M, test.K, test.N);
    cudaEventRecord(stop);
    cudaMemcpy(h_C_corrected, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    
    float corrected_ms;
    cudaEventElapsedTime(&corrected_ms, start, stop);

    // Compare results
    bool original_correct = compare_matrices(h_C_cpu, h_C_original, size_C);
    bool corrected_correct = compare_matrices(h_C_cpu, h_C_corrected, size_C);
    // Print results
    printf("CPU Time: %lld ms\n", cpu_duration.count());
    printf("Original GPU Time: %.2f ms (Correct: %s)\n", original_ms, original_correct ? "Yes" : "No");
    printf("Corrected GPU Time: %.2f ms (Correct: %s)\n", corrected_ms, corrected_correct ? "Yes" : "No");

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_original;
    delete[] h_C_corrected;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Define test cases
    TestCase tests[] = {
        {32, 32, 32, "Small square matrices"},
        {64, 64, 64, "Medium square matrices"},
        {128, 128, 128, "Large square matrices"},
        {256, 128, 64, "Rectangular matrices (M>N)"},
        {64, 128, 256, "Rectangular matrices (M<N)"},
        {31, 31, 31, "Non-tile-aligned matrices"},
        {1024, 1024, 1024, "Very large matrices"},
        {1, 1024, 1, "Vector-matrix edge case"},
        {33, 33, 33, "Slightly over tile size"}
    };

    // Run all test cases
    for (const auto& test : tests) {
        run_test_case(test);
    }

    return 0;
}