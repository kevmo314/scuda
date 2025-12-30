#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <nvml.h>

#include <atomic>
#include <cstdio>
#include <cuda_runtime.h>
#include <list>
#include <mutex>
#include <vector>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"
#include "ptx_fatbin.hpp"

extern int rpc_read(const conn_t *conn, void *data, const std::size_t size);
extern int rpc_read_end(const conn_t *conn);
extern int rpc_write_start_response(const conn_t *conn, const int request_id);
extern int rpc_write(const conn_t *conn, const void *data,
                     const std::size_t size);
extern int rpc_write_end(const conn_t *conn);

void invoke_host_func(void *data);
void append_managed_ptr(const void *conn, void *srcPtr, void *dstPtr,
                        size_t size, cudaMemcpyKind kind, void *graph);
void maybe_destroy_graph_resources(void *graph);

// Stream callback infrastructure
struct PendingCallback {
  uint64_t callback_id;
  cudaStream_t stream;
  cudaError_t status;
};

struct CallbackProxyData {
  conn_t *conn;
  uint64_t callback_id;
};

static std::mutex g_callback_mutex;
static std::unordered_map<conn_t *, std::list<PendingCallback>> g_pending_callbacks;

// Proxy callback that gets called by CUDA runtime
static void CUDART_CB stream_callback_proxy(cudaStream_t stream,
                                            cudaError_t status, void *userData) {
  CallbackProxyData *data = static_cast<CallbackProxyData *>(userData);

  // Queue the callback for later retrieval by client
  {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_pending_callbacks[data->conn].push_back(
        {data->callback_id, stream, status});
  }

  delete data;
}

// Async D2H transfer infrastructure
struct PendingD2HTransfer {
  uint64_t transfer_id;
  void *host_data;      // Server-side buffer with the data
  size_t size;
  cudaError_t status;
};

struct D2HTransferProxyData {
  conn_t *conn;
  uint64_t transfer_id;
  void *host_data;
  size_t size;
};

static std::unordered_map<conn_t *, std::list<PendingD2HTransfer>> g_pending_d2h_transfers;
static std::atomic<uint64_t> g_d2h_transfer_id_counter{1};

// Proxy callback for async D2H transfers
static void CUDART_CB d2h_transfer_proxy(cudaStream_t stream,
                                         cudaError_t status, void *userData) {
  D2HTransferProxyData *data = static_cast<D2HTransferProxyData *>(userData);

  // Queue the completed transfer for later retrieval by client
  {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_pending_d2h_transfers[data->conn].push_back(
        {data->transfer_id, data->host_data, data->size, status});
  }

  delete data;
}

FILE *__cudart_trace_output_stream = stdout;

int handle_cudaMemcpy(conn_t *conn) {
  int request_id;
  cudaError_t result;
  void *src;
  void *dst;
  void *host_data = NULL;  // Must initialize - freed at end
  std::size_t count;
  enum cudaMemcpyKind kind;
  int ret = -1;

  if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
      (kind != cudaMemcpyHostToDevice &&
       rpc_read(conn, &src, sizeof(void *)) < 0) ||
      (kind != cudaMemcpyDeviceToHost &&
       rpc_read(conn, &dst, sizeof(void *)) < 0) ||
      rpc_read(conn, &count, sizeof(size_t)) < 0)
    goto ERROR_0;

  switch (kind) {
  case cudaMemcpyDeviceToHost:
    host_data = malloc(count);
    if (host_data == NULL)
      goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_1;

    result = cudaMemcpy(host_data, src, count, kind);
    break;
  case cudaMemcpyHostToDevice:
    host_data = malloc(count);
    if (host_data == NULL)
      goto ERROR_0;

    if (rpc_read(conn, host_data, count) < 0)
      goto ERROR_1;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_1;

    result = cudaMemcpy(dst, host_data, count, kind);
    break;
  case cudaMemcpyDeviceToDevice:
    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpy(dst, src, count, kind);
    break;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (kind == cudaMemcpyDeviceToHost &&
       rpc_write(conn, host_data, count) < 0) ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  ret = 0;
ERROR_1:
  if (host_data != NULL)
    free((void *)host_data);
ERROR_0:
  return ret;
}

int handle_cudaMemcpyAsync(conn_t *conn) {
  int request_id;
  cudaError_t result;
  void *src;
  void *dst;
  void *host_data = NULL;  // Must initialize
  std::size_t count;
  enum cudaMemcpyKind kind;
  int stream_null_check;
  cudaStream_t stream = 0;
  int ret = -1;
  uint64_t transfer_id = 0;

  if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
      rpc_read(conn, &stream_null_check, sizeof(int)) < 0 ||
      (stream_null_check == 0 &&
       rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0) ||
      (kind != cudaMemcpyHostToDevice &&
       rpc_read(conn, &src, sizeof(void *)) < 0) ||
      (kind != cudaMemcpyDeviceToHost &&
       rpc_read(conn, &dst, sizeof(void *)) < 0) ||
      rpc_read(conn, &count, sizeof(size_t)) < 0)
    goto ERROR_0;

  switch (kind) {
  case cudaMemcpyDeviceToHost: {
    host_data = malloc(count);
    if (host_data == NULL)
      goto ERROR_0;

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    // Queue the async copy
    result = cudaMemcpyAsync(host_data, src, count, kind, stream);

    if (result == cudaSuccess) {
      // Allocate transfer_id and register callback to queue data when ready
      transfer_id = g_d2h_transfer_id_counter.fetch_add(1);

      D2HTransferProxyData *proxy_data = new D2HTransferProxyData();
      proxy_data->conn = conn;
      proxy_data->transfer_id = transfer_id;
      proxy_data->host_data = host_data;
      proxy_data->size = count;

      result = cudaStreamAddCallback(stream, d2h_transfer_proxy, proxy_data, 0);
      if (result != cudaSuccess) {
        delete proxy_data;
        free(host_data);
        host_data = NULL;
      } else {
        // Don't free host_data here - callback will handle it
        host_data = NULL;
      }
    }
    break;
  }
  case cudaMemcpyHostToDevice:
    host_data = malloc(count);
    if (host_data == NULL)
      goto ERROR_0;

    if (rpc_read(conn, host_data, count) < 0) {
      goto ERROR_0;
    }

    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpyAsync(dst, host_data, count, kind, stream);
    break;
  case cudaMemcpyDeviceToDevice:
    request_id = rpc_read_end(conn);
    if (request_id < 0)
      goto ERROR_0;

    result = cudaMemcpyAsync(dst, src, count, kind, stream);
    break;
  }

  // For D2H, send transfer_id instead of data (data comes later via polling)
  if (rpc_write_start_response(conn, request_id) < 0 ||
      (kind == cudaMemcpyDeviceToHost &&
       rpc_write(conn, &transfer_id, sizeof(uint64_t)) < 0) ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  // For H2D, register callback to free host_data when copy completes
  if (kind == cudaMemcpyHostToDevice && host_data != NULL) {
    cudaStreamAddCallback(
        stream,
        [](cudaStream_t stream, cudaError_t status, void *ptr) {
          free(ptr);
        },
        host_data, 0);
    host_data = NULL;  // Callback owns it now
  }

  ret = 0;
ERROR_0:
  if (host_data != NULL)
    free(host_data);
  return ret;
}

int handle_cudaGraphAddKernelNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode = nullptr; // Initialize to nullptr
  cudaGraph_t graph;
  void **args;
  cudaKernelNodeParams pNodeParams = {0};
  std::vector<cudaGraphNode_t> dependencies;
  const cudaKernelNodeParams *pNodeParams_null_check;
  int request_id;
  int num_args;
  int arg_size;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0) {
    printf("Failed to read numDependencies\n");
    return -1;
  }

  if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0) {
    printf("Failed to read graph\n");
    return -1;
  }

  dependencies.resize(numDependencies);
  for (size_t i = 0; i < numDependencies; ++i) {
    if (rpc_read(conn, &dependencies[i], sizeof(cudaGraphNode_t)) < 0) {
      printf("Failed to read Dependency[%zu]\n", i);
      return -1;
    }
  }

  if (rpc_read(conn, &num_args, sizeof(int)) < 0) {
    printf("Failed to read arg count\n");
    return -1;
  }

  args = (void **)malloc(num_args * sizeof(void *));
  if (args == NULL)
    return -1;

  for (int i = 0; i < num_args; ++i) {
    if (rpc_read(conn, &arg_size, sizeof(int)) < 0)
      return -1;

    args[i] = malloc(arg_size);
    if (args[i] == NULL)
      return -1;

    if (rpc_read(conn, args[i], arg_size) < 0)
      return -1;
  }

  if (rpc_read(conn, &pNodeParams_null_check,
               sizeof(const cudaKernelNodeParams *)) < 0) {
    return -1;
  }

  if (pNodeParams_null_check) {
    if (rpc_read(conn, &pNodeParams,
                 sizeof(const struct cudaKernelNodeParams)) < 0) {
      return -1;
    }
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  // make sure we write our kernel args properly
  pNodeParams.kernelParams = args;

  scuda_intercept_result = cudaGraphAddKernelNode(
      &pGraphNode, graph, dependencies.data(), numDependencies, &pNodeParams);

  if (scuda_intercept_result != cudaSuccess) {
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }

  return 0;
}

int handle_cudaLaunchKernel(conn_t *conn) {
  int request_id;
  cudaError_t result;
  const void *func;
  void **args;
  dim3 gridDim, blockDim;
  size_t sharedMem;
  cudaStream_t stream;
  int num_args;
  int arg_size;

  if (rpc_read(conn, &func, sizeof(const void *)) < 0 ||
      rpc_read(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_read(conn, &num_args, sizeof(int)) < 0)
    goto ERROR_0;

  // Allocate memory for the arguments
  args = (void **)malloc(num_args * sizeof(void *));
  if (args == NULL)
    goto ERROR_0;

  for (int i = 0; i < num_args; ++i) {
    if (rpc_read(conn, &arg_size, sizeof(int)) < 0)
      goto ERROR_1;

    // Allocate memory for the argument
    args[i] = malloc(arg_size);
    if (args[i] == NULL)
      goto ERROR_1;

    // Read the actual argument data from the client
    if (rpc_read(conn, args[i], arg_size) < 0)
      goto ERROR_1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;

  result = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  return 0;
ERROR_1:
  for (int i = 0; i < num_args; ++i)
    if (args[i] != NULL)
      free(args[i]);
  free(args);
ERROR_0:
  return -1;
}

std::unordered_map<void **, __cudaFatCudaBinary2 *> fat_binary_map;

extern "C" void **__cudaRegisterFatBinary(void *fatCubin);

int handle___cudaRegisterFatBinary(conn_t *conn) {
  __cudaFatCudaBinary2 *fatCubin =
      (__cudaFatCudaBinary2 *)malloc(sizeof(__cudaFatCudaBinary2));
  unsigned long long size;

  if (rpc_read(conn, fatCubin, sizeof(__cudaFatCudaBinary2)) < 0 ||
      rpc_read(conn, &size, sizeof(unsigned long long)) < 0)
    return -1;

  void *cubin = malloc(size);
  if (rpc_read(conn, cubin, size) < 0)
    return -1;

  fatCubin->text = (uint64_t)cubin;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  void **p = __cudaRegisterFatBinary(fatCubin);

  fat_binary_map[p] = fatCubin;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &p, sizeof(void **)) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubin);

int handle___cudaUnregisterFatBinary(conn_t *conn) {
  void **fatCubin;
  if (rpc_read(conn, &fatCubin, sizeof(void **)) < 0)
    return -1;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  free((void *)fat_binary_map[fatCubin]->text);
  free(fat_binary_map[fatCubin]);
  fat_binary_map.erase(fatCubin);

  __cudaUnregisterFatBinary(fatCubin);

  return 0;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize);

int handle___cudaRegisterFunction(conn_t *conn) {
  void **fatCubinHandle;
  char *hostFun;
  size_t deviceFunLen;
  size_t deviceNameLen;
  char *deviceFun;
  char *deviceName;
  int thread_limit;
  uint8_t mask;
  uint3 tid, bid;
  dim3 bDim, gDim;
  int wSize;

  int request_id;
  int param_count = 0;
  size_t param_offset, param_size;
  int param_sizes[64];

  if (rpc_read(conn, &fatCubinHandle, sizeof(void **)) < 0 ||
      rpc_read(conn, &hostFun, sizeof(const char *)) < 0 ||
      rpc_read(conn, &deviceFunLen, sizeof(size_t)) < 0)
    goto ERROR_0;

  deviceFun = (char *)malloc(deviceFunLen);
  if (rpc_read(conn, deviceFun, deviceFunLen) < 0 ||
      rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0)
    goto ERROR_1;

  deviceName = (char *)malloc(deviceNameLen);
  if (rpc_read(conn, deviceName, deviceNameLen) < 0 ||
      rpc_read(conn, &thread_limit, sizeof(int)) < 0 ||
      rpc_read(conn, &mask, sizeof(uint8_t)) < 0 ||
      (mask & 1 << 0 && rpc_read(conn, &tid, sizeof(uint3)) < 0) ||
      (mask & 1 << 1 && rpc_read(conn, &bid, sizeof(uint3)) < 0) ||
      (mask & 1 << 2 && rpc_read(conn, &bDim, sizeof(dim3)) < 0) ||
      (mask & 1 << 3 && rpc_read(conn, &gDim, sizeof(dim3)) < 0) ||
      (mask & 1 << 4 && rpc_read(conn, &wSize, sizeof(int)) < 0))
    goto ERROR_2;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_2;

  __cudaRegisterFunction(
      fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
      mask & 1 << 0 ? &tid : nullptr, mask & 1 << 1 ? &bid : nullptr,
      mask & 1 << 2 ? &bDim : nullptr, mask & 1 << 3 ? &gDim : nullptr,
      mask & 1 << 4 ? &wSize : nullptr);

  // Query param info using cudaFuncGetParamInfo
  while (param_count < 64) {
    cudaError_t err = cudaFuncGetParamInfo(hostFun, param_count, &param_offset, &param_size);
    if (err != cudaSuccess) {
      // Clear the error state so it doesn't affect subsequent cudaGetLastError calls
      cudaGetLastError();
      break;
    }
    param_sizes[param_count] = (int)param_size;
    param_count++;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &param_count, sizeof(int)) < 0)
    return -1;

  for (int i = 0; i < param_count; i++) {
    if (rpc_write(conn, &param_sizes[i], sizeof(int)) < 0)
      return -1;
  }

  if (rpc_write_end(conn) < 0)
    return -1;

  return 0;
ERROR_2:
  free((void *)deviceName);
ERROR_1:
  free((void *)deviceFun);
ERROR_0:
  return -1;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);

int handle___cudaRegisterFatBinaryEnd(conn_t *conn) {
  void **fatCubinHandle;

  if (rpc_read(conn, &fatCubinHandle, sizeof(void **)) < 0)
    return -1;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  __cudaRegisterFatBinaryEnd(fatCubinHandle);

  if (rpc_write_start_response(conn, request_id) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

// Function pointer type for __cudaPushCallConfiguration
extern "C" unsigned int __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                    size_t sharedMem,
                                                    cudaStream_t stream);

int handle___cudaPushCallConfiguration(conn_t *conn) {
  dim3 gridDim, blockDim;
  size_t sharedMem;
  cudaStream_t stream;

  // Read the grid dimensions from the client
  if (rpc_read(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
    return -1;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  unsigned int result =
      __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                  size_t *sharedMem,
                                                  void *stream);

int handle___cudaPopCallConfiguration(conn_t *conn) {
  dim3 gridDim, blockDim;
  size_t sharedMem;
  cudaStream_t stream;

  int request_id = rpc_read_end(conn);
  if (request_id < 0)
    return -1;

  cudaError_t result =
      __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

// typedef void (*__cudaInitModule_type)(void **fatCubinHandle);

// void __cudaInitModule(void **fatCubinHandle) {
//   std::cerr << "calling __cudaInitModule" << std::endl;
// }

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global);

int handle___cudaRegisterVar(conn_t *conn) {
  void **fatCubinHandle;
  char *hostVar;
  char *deviceAddress;
  char *deviceName;
  int ext;
  size_t size;
  int constant;
  int global;

  // Read the fatCubinHandle
  if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0) {
    std::cerr << "Failed reading fatCubinHandle" << std::endl;
    return -1;
  }

  // Read hostVar as pointer value (not a string - it's a client-side address)
  if (rpc_read(conn, &hostVar, sizeof(void *)) < 0) {
    std::cerr << "Failed to read hostVar" << std::endl;
    return -1;
  }

  // Read deviceAddress as pointer value (not a string - it's a client-side address)
  if (rpc_read(conn, &deviceAddress, sizeof(void *)) < 0) {
    std::cerr << "Failed to read deviceAddress" << std::endl;
    return -1;
  }

  // Read deviceName (this IS a string)
  size_t deviceNameLen;
  if (rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to read deviceName length" << std::endl;
    return -1;
  }
  deviceName = (char *)malloc(deviceNameLen);
  if (rpc_read(conn, deviceName, deviceNameLen) < 0) {
    std::cerr << "Failed to read deviceName" << std::endl;
    free(deviceName);
    return -1;
  }

  // Read ext, size, constant, global
  if (rpc_read(conn, &ext, sizeof(int)) < 0) {
    std::cerr << "Failed reading ext" << std::endl;
    free(deviceName);
    return -1;
  }

  if (rpc_read(conn, &size, sizeof(size_t)) < 0) {
    std::cerr << "Failed reading size" << std::endl;
    free(deviceName);
    return -1;
  }

  if (rpc_read(conn, &constant, sizeof(int)) < 0) {
    std::cerr << "Failed reading constant" << std::endl;
    free(deviceName);
    return -1;
  }

  if (rpc_read(conn, &global, sizeof(int)) < 0) {
    std::cerr << "Failed reading global" << std::endl;
    free(deviceName);
    return -1;
  }

  // Note: We don't call __cudaRegisterVar here because hostVar and deviceAddress
  // are client-side pointers that are meaningless on the server. The symbol
  // registration is handled via cuModuleGetGlobal when cudaMemcpyToSymbol is called.

  free(deviceName);

  // End request phase
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    std::cerr << "rpc_read_end failed" << std::endl;
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 || rpc_write_end(conn) < 0)
    return -1;

  return 0;
}

int handle_cudaFree(conn_t *conn) {
  void *devPtr;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &devPtr, sizeof(void *)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaFree(devPtr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaMallocManaged(conn_t *conn) {
  void *devPtr;
  size_t size;
  unsigned int flags;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &devPtr, sizeof(void *)) < 0 ||
      rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaMallocManaged(&devPtr, size, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &devPtr, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaMallocHost(conn_t *conn) {
  void *ptr;
  size_t size;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &size, sizeof(size_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaMallocHost(&ptr, size);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ptr, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaHostAlloc(conn_t *conn) {
  std::cerr << "SERVER: handle_cudaHostAlloc entered" << std::endl;
  void *pHost;
  size_t size;
  unsigned int flags;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &size, sizeof(size_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0 || false)
    goto ERROR_0;
  std::cerr << "SERVER: cudaHostAlloc size=" << size << " flags=" << flags << std::endl;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaHostAlloc(&pHost, size, flags);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pHost, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaFreeHost(conn_t *conn) {
  void *ptr;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &ptr, sizeof(void *)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaFreeHost(ptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphGetNodes(conn_t *conn) {
  cudaGraph_t graph;
  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaGraphGetNodes(graph, nodes, &numNodes);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &nodes, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &numNodes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

typedef struct callBackData {
  conn_t *conn;
  void (*callback)(void *);
  void *data;
} callBackData_t;

int handle_cudaGraphAddHostNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> pDependencies;
  struct cudaHostNodeParams *pNodeParams_null_check;
  struct cudaHostNodeParams pNodeParams;
  int request_id;
  cudaError_t scuda_intercept_result;
  callBackData_t *hostFnData;

  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
          rpc_read(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
          rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
          [=, &pDependencies]()
          -> bool {
        pDependencies.resize(numDependencies); // Resize the dependencies vector
        for (size_t i = 0; i < numDependencies; ++i) {
          if (rpc_read(conn, &pDependencies[i], sizeof(const cudaGraphNode_t)) <
              0) {
            return false;
          }
        }
        return true;
      }() == false ||
                 rpc_read(conn, &pNodeParams_null_check,
                          sizeof(const struct cudaHostNodeParams *)) < 0 ||
                 (pNodeParams_null_check &&
                  rpc_read(conn, &pNodeParams,
                           sizeof(const struct cudaHostNodeParams)) < 0) ||
                 false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  hostFnData = (callBackData_t *)malloc(sizeof(callBackData_t));
  // assign the previous function pointer so we can map back to it
  hostFnData->callback = pNodeParams.fn;
  hostFnData->data = pNodeParams.userData;
  hostFnData->conn = conn;

  pNodeParams.fn = invoke_host_func;
  pNodeParams.userData = hostFnData;

  scuda_intercept_result = cudaGraphAddHostNode(
      &pGraphNode, graph, pDependencies.data(), numDependencies, &pNodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphAddMemcpyNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> pDependencies;
  struct cudaMemcpy3DParms *pCopyParams_null_check;
  struct cudaMemcpy3DParms pCopyParams;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
          rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
          [=, &pDependencies]()
          -> bool {
        pDependencies.resize(numDependencies); // Resize the dependencies vector
        for (size_t i = 0; i < numDependencies; ++i) {
          if (rpc_read(conn, &pDependencies[i], sizeof(const cudaGraphNode_t)) <
              0) {
            return false;
          }
        }
        return true;
      }() == false ||
                 rpc_read(conn, &pCopyParams_null_check,
                          sizeof(const struct cudaMemcpy3DParms *)) < 0 ||
                 (pCopyParams_null_check &&
                  rpc_read(conn, &pCopyParams,
                           sizeof(const struct cudaMemcpy3DParms)) < 0) ||
                 false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  append_managed_ptr(conn, (void *)pCopyParams.srcPtr.ptr,
                     (void *)pCopyParams.dstPtr.ptr, pCopyParams.extent.width,
                     pCopyParams.kind, graph);

  scuda_intercept_result = cudaGraphAddMemcpyNode(
      &pGraphNode, graph, pDependencies.data(), numDependencies, &pCopyParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphDestroy(conn_t *conn) {
  cudaGraph_t graph;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 || false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaGraphDestroy(graph);

  maybe_destroy_graph_resources(graph);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaGraphAddMemFreeNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraphNode_t pGraphNodeP;
  cudaGraph_t graph;
  cudaGraphNode_t *singleDependency = nullptr;
  std::vector<cudaGraphNode_t> pDependencies;
  void *dptr;
  int request_id;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_read(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
    goto ERROR_0;

  if (numDependencies == 1) {
    if (rpc_read(conn, &pGraphNodeP, sizeof(cudaGraphNode_t)) < 0) {
      goto ERROR_1;
    }
  } else if (numDependencies > 1) {
    pDependencies.resize(numDependencies);
    for (size_t i = 0; i < numDependencies; ++i) {
      if (rpc_read(conn, &pDependencies[i], sizeof(cudaGraphNode_t)) < 0) {
        goto ERROR_0;
      }
    }
  }

  if (rpc_read(conn, &dptr, sizeof(void *)) < 0)
    goto ERROR_1;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_1;

  scuda_intercept_result = cudaGraphAddMemFreeNode(
      &pGraphNode, graph,
      (numDependencies == 1) ? &pGraphNodeP : pDependencies.data(),
      numDependencies, dptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_1;

  if (singleDependency)
    free(singleDependency);

  return 0;

ERROR_1:
  if (singleDependency)
    free(singleDependency);
ERROR_0:
  return -1;
}

int handle_cudaGraphAddMemAllocNode(conn_t *conn) {
  size_t numDependencies;
  cudaGraphNode_t pGraphNode;
  cudaGraphNode_t pGraphNodeP;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> pDependencies;
  struct cudaMemAllocNodeParams *nodeParams_null_check;
  struct cudaMemAllocNodeParams nodeParams;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_read(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
      rpc_read(conn, &nodeParams_null_check,
               sizeof(struct cudaMemAllocNodeParams *)) < 0 ||
      (nodeParams_null_check &&
       rpc_read(conn, &nodeParams, sizeof(struct cudaMemAllocNodeParams)) <
           0) ||
      false)
    goto ERROR_0;

  if (numDependencies == 1) {
    if (rpc_read(conn, &pGraphNodeP, sizeof(cudaGraphNode_t)) < 0) {
      goto ERROR_0;
    }
  } else if (numDependencies > 1) {
    pDependencies.resize(numDependencies);
    for (size_t i = 0; i < numDependencies; ++i) {
      if (rpc_read(conn, &pDependencies[i], sizeof(cudaGraphNode_t)) < 0) {
        goto ERROR_0;
      }
    }
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  scuda_intercept_result = cudaGraphAddMemAllocNode(
      &pGraphNode, graph,
      (numDependencies == 1) ? &pGraphNodeP : pDependencies.data(),
      numDependencies, &nodeParams);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &nodeParams.dptr, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaDeviceGetGraphMemAttribute(conn_t *conn) {
  int device;
  enum cudaGraphMemAttributeType attr;
  void *value;
  int request_id;
  cudaError_t scuda_intercept_result;
  if (rpc_read(conn, &device, sizeof(int)) < 0 ||
      rpc_read(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0 ||
      false)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;
  scuda_intercept_result = cudaDeviceGetGraphMemAttribute(device, attr, &value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(void *)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaMemcpy3D(conn_t *conn) {
  struct cudaMemcpy3DParms p;
  size_t src_data_size;
  std::vector<char> src_buffer;
  int request_id;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &p, sizeof(struct cudaMemcpy3DParms)) < 0 ||
      rpc_read(conn, &src_data_size, sizeof(size_t)) < 0)
    goto ERROR_0;

  // If source data was sent, read it into a temporary buffer
  if (src_data_size > 0 &&
      (p.kind == cudaMemcpyHostToDevice || p.kind == cudaMemcpyHostToHost)) {
    src_buffer.resize(src_data_size);
    if (rpc_read(conn, src_buffer.data(), src_data_size) < 0)
      goto ERROR_0;
    // Point srcPtr to our local buffer
    p.srcPtr.ptr = src_buffer.data();
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  scuda_intercept_result = cudaMemcpy3D(&p);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0)
    goto ERROR_0;

  // If copying to host memory, send destination data back
  if (p.dstPtr.ptr != nullptr && scuda_intercept_result == cudaSuccess &&
      (p.kind == cudaMemcpyDeviceToHost || p.kind == cudaMemcpyHostToHost)) {
    size_t dst_data_size = p.dstPtr.pitch * p.extent.height * p.extent.depth;
    if (rpc_write(conn, p.dstPtr.ptr, dst_data_size) < 0)
      goto ERROR_0;
  }

  if (rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaCreateTextureObject(conn_t *conn) {
  const struct cudaResourceDesc *pResDescPtr;
  struct cudaResourceDesc resDesc;
  const struct cudaTextureDesc *pTexDescPtr;
  struct cudaTextureDesc texDesc;
  const struct cudaResourceViewDesc *pResViewDescPtr;
  struct cudaResourceViewDesc resViewDesc;
  cudaTextureObject_t texObject;
  int request_id;
  cudaError_t scuda_intercept_result;

  // Read pResDesc pointer to check if null
  if (rpc_read(conn, &pResDescPtr, sizeof(const struct cudaResourceDesc*)) < 0)
    goto ERROR_0;
  if (pResDescPtr != nullptr) {
    if (rpc_read(conn, &resDesc, sizeof(struct cudaResourceDesc)) < 0)
      goto ERROR_0;
  }

  // Read pTexDesc pointer to check if null
  if (rpc_read(conn, &pTexDescPtr, sizeof(const struct cudaTextureDesc*)) < 0)
    goto ERROR_0;
  if (pTexDescPtr != nullptr) {
    if (rpc_read(conn, &texDesc, sizeof(struct cudaTextureDesc)) < 0)
      goto ERROR_0;
  }

  // Read pResViewDesc pointer to check if null
  if (rpc_read(conn, &pResViewDescPtr, sizeof(const struct cudaResourceViewDesc*)) < 0)
    goto ERROR_0;
  if (pResViewDescPtr != nullptr) {
    if (rpc_read(conn, &resViewDesc, sizeof(struct cudaResourceViewDesc)) < 0)
      goto ERROR_0;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  // Call with NULL or pointer based on what client sent
  scuda_intercept_result = cudaCreateTextureObject(
      &texObject,
      pResDescPtr != nullptr ? &resDesc : nullptr,
      pTexDescPtr != nullptr ? &texDesc : nullptr,
      pResViewDescPtr != nullptr ? &resViewDesc : nullptr);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &texObject, sizeof(cudaTextureObject_t)) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaStreamAddCallback(conn_t *conn) {
  cudaStream_t stream;
  uint64_t callback_id;
  unsigned int flags;
  int request_id;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_read(conn, &callback_id, sizeof(uint64_t)) < 0 ||
      rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  {
    // Create proxy data that will be passed to our callback
    CallbackProxyData *proxy_data = new CallbackProxyData{conn, callback_id};

    // Register proxy callback with CUDA
    scuda_intercept_result =
        cudaStreamAddCallback(stream, stream_callback_proxy, proxy_data, flags);

    if (scuda_intercept_result != cudaSuccess) {
      delete proxy_data;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaStreamSynchronize(conn_t *conn) {
  cudaStream_t stream;
  int request_id;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  scuda_intercept_result = cudaStreamSynchronize(stream);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaDeviceSynchronize(conn_t *conn) {
  int request_id;
  cudaError_t scuda_intercept_result;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  scuda_intercept_result = cudaDeviceSynchronize();

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_cudaEventSynchronize(conn_t *conn) {
  cudaEvent_t event;
  int request_id;
  cudaError_t scuda_intercept_result;

  if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
    goto ERROR_0;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  scuda_intercept_result = cudaEventSynchronize(event);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &scuda_intercept_result, sizeof(cudaError_t)) < 0 ||
      rpc_write_end(conn) < 0)
    goto ERROR_0;

  return 0;
ERROR_0:
  return -1;
}

int handle_SCUDA_POLL_CALLBACKS(conn_t *conn) {
  int request_id;
  std::list<PendingCallback> callbacks;
  std::list<PendingD2HTransfer> d2h_transfers;

  request_id = rpc_read_end(conn);
  if (request_id < 0)
    goto ERROR_0;

  // Get pending callbacks and D2H transfers for this connection
  {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    auto it = g_pending_callbacks.find(conn);
    if (it != g_pending_callbacks.end()) {
      callbacks = std::move(it->second);
      it->second.clear();
    }

    auto d2h_it = g_pending_d2h_transfers.find(conn);
    if (d2h_it != g_pending_d2h_transfers.end()) {
      d2h_transfers = std::move(d2h_it->second);
      d2h_it->second.clear();
    }
  }

  {
    uint32_t num_callbacks = static_cast<uint32_t>(callbacks.size());
    uint32_t num_d2h_transfers = static_cast<uint32_t>(d2h_transfers.size());

    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &num_callbacks, sizeof(uint32_t)) < 0)
      goto ERROR_0;

    for (const auto &cb : callbacks) {
      if (rpc_write(conn, &cb.callback_id, sizeof(uint64_t)) < 0 ||
          rpc_write(conn, &cb.stream, sizeof(cudaStream_t)) < 0 ||
          rpc_write(conn, &cb.status, sizeof(cudaError_t)) < 0)
        goto ERROR_0;
    }

    // Send D2H transfer completions
    if (rpc_write(conn, &num_d2h_transfers, sizeof(uint32_t)) < 0)
      goto ERROR_0;

    for (auto &transfer : d2h_transfers) {
      if (rpc_write(conn, &transfer.transfer_id, sizeof(uint64_t)) < 0 ||
          rpc_write(conn, &transfer.size, sizeof(size_t)) < 0 ||
          rpc_write(conn, &transfer.status, sizeof(cudaError_t)) < 0 ||
          (transfer.status == cudaSuccess && transfer.size > 0 &&
           rpc_write(conn, transfer.host_data, transfer.size) < 0))
        goto ERROR_1;
    }

    if (rpc_write_end(conn) < 0)
      goto ERROR_1;
  }

  // Free D2H transfer host buffers
  for (auto &transfer : d2h_transfers) {
    if (transfer.host_data != NULL) {
      free(transfer.host_data);
    }
  }

  return 0;
ERROR_1:
  // Free D2H transfer host buffers on error
  for (auto &transfer : d2h_transfers) {
    if (transfer.host_data != NULL) {
      free(transfer.host_data);
    }
  }
ERROR_0:
  return -1;
}
