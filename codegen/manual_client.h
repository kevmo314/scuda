#include <nvml.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
