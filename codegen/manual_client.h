#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
