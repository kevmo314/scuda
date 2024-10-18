#include <nvml.h>
#include <cuda.h>
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
