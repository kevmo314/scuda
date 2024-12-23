#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN with Unified Memory.
 **/
int main(int argc, char** argv)
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;

    // Create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

    int n = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = n * c * h * w;
    cudnnTensorDescriptor_t x_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    // Allocate unified memory for the tensor
    float *x;
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float));

    // Initialize the tensor
    for (int i = 0; i < NUM_ELEMENTS; i++) x[i] = i * 1.00f;

    std::cout << "Original array: ";
    for (int i = 0; i < NUM_ELEMENTS; i++) std::cout << x[i] << " ";
    std::cout << std::endl;

    // Synchronize to ensure data is accessible on the device
    cudaDeviceSynchronize();

    // Create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x
    );

    // Synchronize to ensure data is updated on the host
    cudaDeviceSynchronize();

    cudnnDestroy(handle_);
    std::cout << "Destroyed cuDNN handle." << std::endl;

    std::cout << "New array: ";
    for (int i = 0; i < NUM_ELEMENTS; i++) std::cout << x[i] << " ";
    std::cout << std::endl;

    cudaFree(x);

    return 0;
}
