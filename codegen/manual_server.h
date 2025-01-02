#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

int handle_cudaFree(void *conn);
int handle_cudaMemcpy(void *conn);
int handle_cudaMemcpyAsync(void *conn);
int handle_cudaLaunchKernel(void *conn);
int handle_cudaMallocManaged(void *conn);
int handle___cudaRegisterVar(void *conn);
int handle___cudaRegisterFunction(void *conn);
int handle___cudaRegisterFatBinary(void *conn);
int handle___cudaRegisterFatBinaryEnd(void *conn);
int handle___cudaPushCallConfiguration(void *conn);
int handle___cudaPopCallConfiguration(void *conn);
