#include <nvml.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
extern "C" void **__cudaRegisterFatBinary(void **fatCubin);
extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                   const char *hostFun,
                                   char *deviceFun,
                                   const char *deviceName,
                                   int thread_limit,
                                   uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream);
extern "C" unsigned __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                               size_t *sharedMem, void *stream);
extern "C" void __cudaInitModule(void **fatCubinHandle);
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global);

// cublas
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc);
