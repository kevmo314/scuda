#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel to add elements of two arrays
__global__ void addKernel(int *a, int *b, int *c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    // Define array size
    const int size = 10;
    const int bytes = size * sizeof(int);

    std::cout << "HELLO" << std::endl;

    // Unified memory allocation
    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize arrays on the CPU
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Define kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "launching kernel..." << std::endl;

    std::cout << "pointer a: " << a << std::endl;
    std::cout << "pointer b: " << b << std::endl;
    std::cout << "pointer c: " << c << std::endl;

    // Launch the kernel
    addKernel<<<blocks, threadsPerBlock>>>(a, b, c, size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Display results
    std::cout << "Results:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << "a[" << i << "] + b[" << i << "] = " << c[i] << "\n";
    }

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
