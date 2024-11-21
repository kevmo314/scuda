#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

/**
 * Minimal example to apply sigmoid activation on a tensor 
 * using cuDNN with cudaMalloc instead of cudaMallocManaged.
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

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

    int n = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = n * c * h * w;
    cudnnTensorDescriptor_t x_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    // create and initialize the tensor
    float *h_x = new float[NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; i++) h_x[i] = i * 1.00f;
    
    float *d_x; // device mem
    cudaMalloc(&d_x, NUM_ELEMENTS * sizeof(float));
    cudaMemcpy(d_x, h_x, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Original array: "; 
    for (int i = 0; i < NUM_ELEMENTS; i++) std::cout << h_x[i] << " ";
    std::cout << std::endl;

    // create activation function descriptor
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
        d_x,
        beta,
        x_desc,
        d_x
    );

    // copy mem back to host
    cudaMemcpy(h_x, d_x, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    cudnnDestroy(handle_);
    std::cout << "Destroyed cuDNN handle." << std::endl;

    std::cout << "New array: ";
    for (int i = 0; i < NUM_ELEMENTS; i++) std::cout << h_x[i] << " ";
    std::cout << std::endl;

    cudaFree(d_x);

    return 0;
}
