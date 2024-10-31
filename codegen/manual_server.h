#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>

int handle_cudaMemcpy(void *conn);
int handle_cudaMemcpyAsync(void *conn);

// cublas
int handle_cublasSgemm_v2(void *conn);
int handle_cublasCreate_v2(void *conn);
