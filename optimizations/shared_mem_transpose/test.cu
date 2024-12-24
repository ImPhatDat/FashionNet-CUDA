#include <cuda_runtime.h>
#include <stdio.h>

// Define tile size for shared memory
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Kernel function for matrix transpose with shared memory
__global__ void transposeShared(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Avoid bank conflicts

    // Calculate input and output indices
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width_in = width;

    // Load data into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((x < width) && (y + j < height)) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width_in + x];
        }
    }
    __syncthreads(); // Ensure all threads have loaded data

    // Calculate transposed output indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Swap x and y for transpose
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed data from shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((x < height) && (y + j < width)) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Host code to execute the kernel
int main() {
    const int width = 1024;  // Matrix width
    const int height = 512;  // Matrix height
    const int size = width * height * sizeof(float);

    // Allocate memory on host
    float *h_idata = (float *)malloc(size);
    float *h_odata = (float *)malloc(size);

    // Initialize matrix with sample data
    for (int i = 0; i < width * height; ++i) {
        h_idata[i] = (float)i;
    }

    // Allocate memory on device
    float *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, size);
    cudaMalloc((void **)&d_odata, size);

    // Copy input data to device
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    transposeShared<<<dimGrid, dimBlock>>>(d_odata, d_idata, width, height);
    checkCUDAError("Kernel execution");

    // Copy results back to host
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    printf("Matrix transpose completed successfully.\n");
    return 0;
}
