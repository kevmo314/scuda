#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>

int handle_cudaMemcpy(void *conn);
int handle_cudaMemcpyAsync(void *conn);
int handle_cudaLaunchKernel(void *conn);
int handle___cudaRegisterFunction(void *conn);
