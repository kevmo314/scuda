#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel to add elements of two arrays
// __global__ void addKernel(int *a, int *b, int *c, int size) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx < size) {
//         c[idx] = a[idx] * b[idx];
//     }
// }

__global__ void mulKernel(int *a, int *c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] * 100;
    }
}

int main() {

    // Define array size
    const int size = 10;
    const int bytes = size * sizeof(int);

    // Unified memory allocation
    int *a, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize arrays on the CPU
    for (int i = 0; i < size; ++i) {
        a[i] = i;
    }

    // Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "launching kernel..." << std::endl;

    for (int i = 0; i < size; ++i) {
        std::cout << "a[" << i << "] + b[" << i << "] = " << a[i] << "\n";
    }

    // Launch the kernel
    mulKernel<<<blocks, threadsPerBlock>>>(a, c, size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Display results
    std::cout << "Results:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << "a[" << i << "] + b[" << i << "] = " << c[i] << "\n";
    }

    // Free unified memory
    cudaFree(a);
    cudaFree(c);

    return 0;
}
