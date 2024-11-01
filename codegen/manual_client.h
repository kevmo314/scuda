#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

nvmlReturn_t nvmlInit_v2();
nvmlReturn_t nvmlShutdown();
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
