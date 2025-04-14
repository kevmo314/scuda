#include <cublas_v2.h>
#include <cuda.h>
#include <nvml.h>
#include <cuda_runtime_api.h>

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, 
    const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams);
cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, 
    const cudaGraphNode_t* pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams* nodeParams);
cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr);
cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value);
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies,
    size_t numDependencies, unsigned int flags);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream);
extern "C" void **__cudaRegisterFatBinary(void **fatCubin);
extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize);
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem,
                                                struct CUstream_st *stream);
extern "C" unsigned __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                               size_t *sharedMem, void *stream);
extern "C" void __cudaInitModule(void **fatCubinHandle);
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global);
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const struct cudaHostNodeParams *pNodeParams);
cudaError_t
cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaMemcpy3DParms *pCopyParams);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                              size_t *numNodes);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
