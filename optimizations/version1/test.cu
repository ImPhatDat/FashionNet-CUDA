#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Optimized for most GPU architectures
constexpr int TILE_DIM = 32;
constexpr int BLOCK_DIM = 32;

__global__ void matrixTransposeKernel(const float* input, float* output, 
                                     const int rows, const int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 prevents bank conflicts
    
    // Calculate global indices
    const int global_row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int global_col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Calculate local indices within the tile
    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    
    // Load phase - each thread loads one element into shared memory if within bounds
    for (int i = 0; i < TILE_DIM && global_row + i < rows && global_col < cols; i += BLOCK_DIM) {
        tile[local_row + i][local_col] = input[(global_row + i) * cols + global_col];
    }
    
    __syncthreads();
    
    // Calculate transposed global positions
    const int new_row = blockIdx.x * TILE_DIM + threadIdx.y;
    const int new_col = blockIdx.y * TILE_DIM + threadIdx.x;
    
    // Store phase - write data from shared memory to global memory
    for (int i = 0; i < TILE_DIM && new_row + i < cols && new_col < rows; i += BLOCK_DIM) {
        output[(new_row + i) * rows + new_col] = tile[threadIdx.x][threadIdx.y + i];
    }
}

cudaError_t transposeMatrix(const float* input, float* output, int rows, int cols) {
    // Input validation
    if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }
    
    float *d_input, *d_output;
    cudaError_t err;
    
    // Calculate memory sizes
    size_t input_size = rows * cols * sizeof(float);
    
    // Allocate device memory
    err = cudaMalloc(&d_input, input_size);
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&d_output, input_size);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        return err;
    }
    
    // Copy input data to device
    err = cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return err;
    }
    
    // Calculate grid dimensions to cover any input size
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(
        (cols + TILE_DIM - 1) / TILE_DIM,  // Ceiling division for width
        (rows + TILE_DIM - 1) / TILE_DIM   // Ceiling division for height
    );
    
    // Launch kernel
    matrixTransposeKernel<<<grid, block>>>(d_input, d_output, rows, cols);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return err;
    }
    
    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        return err;
    }
    
    // Copy result back to host
    err = cudaMemcpy(output, d_output, input_size, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    
    return err;
}

// Test function to verify transpose with different matrix sizes
void testTranspose(int rows, int cols) {
    printf("\nTesting transpose with matrix size: %d x %d\n", rows, cols);
    
    // Allocate host memory
    float* input = new float[rows * cols];
    float* output = new float[rows * cols];
    
    // Initialize input matrix
    for (int i = 0; i < rows * cols; i++) {
        input[i] = static_cast<float>(i);
    }
    
    // Perform transpose
    cudaError_t err = transposeMatrix(input, output, rows, cols);
    if (err != cudaSuccess) {
        fprintf(stderr, "Transpose failed: %s\n", cudaGetErrorString(err));
        delete[] input;
        delete[] output;
        return;
    }
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < rows && correct; i++) {
        for (int j = 0; j < cols && correct; j++) {
            if (input[i * cols + j] != output[j * rows + i]) {
                correct = false;
                printf("Mismatch at position [%d,%d]: %f != %f\n", 
                       i, j, input[i * cols + j], output[j * rows + i]);
                break;
            }
        }
    }
    
    printf("Matrix transpose %s\n", correct ? "successful!" : "failed!");
    
    delete[] input;
    delete[] output;
}

int main() {
    // Test various matrix sizes
    testTranspose(1000, 100);    // Tall matrix
    testTranspose(100, 1000);    // Wide matrix
    testTranspose(32, 32);       // Exactly one tile
    testTranspose(31, 31);       // Smaller than tile
    testTranspose(33, 33);       // Slightly larger than tile
    testTranspose(2048, 2048);   // Large square matrix
    testTranspose(1, 1000);      // Single row
    testTranspose(1000, 1);      // Single column
    testTranspose(17, 23);       // Prime numbers
    testTranspose(2, 3);       // Prime numbers
    testTranspose(3, 2);       // Prime numbers
    testTranspose(15, 61);       // Prime numbers
    
    return 0;
}