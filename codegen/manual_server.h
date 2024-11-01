#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>

int handle_nvmlShutdown(void *conn);
int handle_nvmlInit_v2(void *conn);
int handle_cudaMemcpy(void *conn);
int handle_cudaMemcpyAsync(void *conn);
