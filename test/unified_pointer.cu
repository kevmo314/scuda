#include <iostream>
#include <math.h>

struct Operation {
    float *x;
    float *y;
    int n;
};

// CUDA kernel to add elements of two arrays
__global__ void add(Operation *op) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    printf("The X is: %x\n", op->x[0]);
    printf("The Y is: %x\n", op->y[0]);
    for (int i = index; i < op->n; i += stride)
    {
      op->y[i] = op->x[i] + op->y[i];
      printf("The value is: %f\n", op->y[i]);
    }
}

int main(void) {
    Operation *op;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&op, sizeof(Operation));
    op->n = 100;

    cudaMallocManaged(&op->x, op->n * sizeof(float));
    cudaMallocManaged(&op->y, op->n * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < op->n; i++) {
        op->x[i] = 1.0f;
        op->y[i] = 2.0f;
    }

    // Launch kernel on n elements on the GPU
    int blockSize = 256;
    int numBlocks = (op->n + blockSize - 1) / blockSize;

    std::cout << "numBlocks: " << numBlocks << std::endl;
    std::cout << "N: " << op->n << std::endl;

    add<<<numBlocks, blockSize>>>(op);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Log results for debugging
    std::cout << "Results (y = x + y):" << std::endl;
    for (int i = 0; i < op->n; i++) {
        std::cout << "y[" << i << "] = " << op->y[i] << " (expected: 3.0)" << std::endl;
    }

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < op->n; i++) {
        maxError = fmax(maxError, fabs(op->y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(op->x);
    cudaFree(op->y);
    cudaFree(op);

    return 0;
}
