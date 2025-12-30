#include <cassert>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <nvml.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gen_api.h"
#include "ptx_fatbin.hpp"
#include "rpc.h"

// SCUDA internal operation codes (not from codegen)
#define RPC_SCUDA_POLL_CALLBACKS 1100

// Forward declarations needed for callback infrastructure
extern conn_t *rpc_client_get_connection(unsigned int index);

// Stream callback infrastructure
struct StreamCallbackInfo {
  cudaStreamCallback_t callback;
  void *userData;
};

// Pending D2H transfer tracking
struct PendingD2HInfo {
  void *dst;       // Client-side destination pointer
  size_t size;     // Size of the transfer
};

static std::mutex g_callback_mutex;
static std::unordered_map<uint64_t, StreamCallbackInfo> g_pending_callbacks;
static std::unordered_map<uint64_t, PendingD2HInfo> g_pending_d2h_transfers;
static std::atomic<uint64_t> g_callback_id_counter{1};
static std::atomic<bool> g_callback_thread_running{false};
static std::thread g_callback_thread;

// Poll for and invoke pending stream callbacks
static cudaError_t poll_stream_callbacks(conn_t *conn) {
  if (rpc_write_start_request(conn, RPC_SCUDA_POLL_CALLBACKS) < 0 ||
      rpc_wait_for_response(conn) < 0)
    return cudaErrorDevicesUnavailable;

  uint32_t num_callbacks;
  if (rpc_read(conn, &num_callbacks, sizeof(uint32_t)) < 0)
    return cudaErrorDevicesUnavailable;

  // Collect callbacks to invoke (do RPC reads first, invoke later)
  std::vector<std::pair<StreamCallbackInfo, std::pair<cudaStream_t, cudaError_t>>> callbacks_to_invoke;

  for (uint32_t i = 0; i < num_callbacks; i++) {
    uint64_t callback_id;
    cudaStream_t stream;
    cudaError_t status;

    if (rpc_read(conn, &callback_id, sizeof(uint64_t)) < 0 ||
        rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_read(conn, &status, sizeof(cudaError_t)) < 0)
      return cudaErrorDevicesUnavailable;

    // Find the callback info
    StreamCallbackInfo info = {nullptr, nullptr};
    {
      std::lock_guard<std::mutex> lock(g_callback_mutex);
      auto it = g_pending_callbacks.find(callback_id);
      if (it != g_pending_callbacks.end()) {
        info = it->second;
        g_pending_callbacks.erase(it);
      }
    }

    if (info.callback) {
      callbacks_to_invoke.push_back({info, {stream, status}});
    }
  }

  // Read D2H transfer completions
  uint32_t num_d2h_transfers;
  if (rpc_read(conn, &num_d2h_transfers, sizeof(uint32_t)) < 0)
    return cudaErrorDevicesUnavailable;

  for (uint32_t i = 0; i < num_d2h_transfers; i++) {
    uint64_t transfer_id;
    size_t size;
    cudaError_t status;

    if (rpc_read(conn, &transfer_id, sizeof(uint64_t)) < 0 ||
        rpc_read(conn, &size, sizeof(size_t)) < 0 ||
        rpc_read(conn, &status, sizeof(cudaError_t)) < 0)
      return cudaErrorDevicesUnavailable;

    // Find the pending D2H transfer
    PendingD2HInfo d2h_info = {nullptr, 0};
    {
      std::lock_guard<std::mutex> lock(g_callback_mutex);
      auto it = g_pending_d2h_transfers.find(transfer_id);
      if (it != g_pending_d2h_transfers.end()) {
        d2h_info = it->second;
        g_pending_d2h_transfers.erase(it);
      }
    }

    // Read the data if transfer succeeded
    if (status == cudaSuccess && size > 0) {
      if (d2h_info.dst != nullptr) {
        // Read directly into destination buffer
        if (rpc_read(conn, d2h_info.dst, size) < 0)
          return cudaErrorDevicesUnavailable;
      } else {
        // Destination not found - read and discard
        std::vector<char> discard(size);
        if (rpc_read(conn, discard.data(), size) < 0)
          return cudaErrorDevicesUnavailable;
      }
    }
  }

  if (rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  // Now invoke callbacks outside of RPC transaction
  for (auto &cb_pair : callbacks_to_invoke) {
    cb_pair.first.callback(cb_pair.second.first, cb_pair.second.second,
                           cb_pair.first.userData);
  }

  return cudaSuccess;
}

// Background thread that polls for callbacks
static void callback_polling_thread() {
  while (g_callback_thread_running.load()) {
    // Check if there are any pending callbacks or D2H transfers to wait for
    {
      std::lock_guard<std::mutex> lock(g_callback_mutex);
      if (g_pending_callbacks.empty() && g_pending_d2h_transfers.empty()) {
        // Nothing pending, exit thread
        g_callback_thread_running.store(false);
        return;
      }
    }

    // Poll for callbacks and D2H transfers
    conn_t *conn = rpc_client_get_connection(0);
    if (conn) {
      poll_stream_callbacks(conn);
    }

    // Sleep for a bit before polling again
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

static void start_callback_thread() {
  bool expected = false;
  if (g_callback_thread_running.compare_exchange_strong(expected, true)) {
    // Detach any previous thread
    if (g_callback_thread.joinable()) {
      g_callback_thread.detach();
    }
    g_callback_thread = std::thread(callback_polling_thread);
    g_callback_thread.detach();
  }
}

extern void add_host_node(void *, void *);

size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output,
                  size_t output_size);

extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);
int is_unified_pointer(conn_t *conn, void *arg);
int maybe_copy_unified_arg(conn_t *conn, void *arg, enum cudaMemcpyKind kind);
extern void rpc_close(conn_t *conn);
extern cudaError_t cuda_memcpy_unified_ptrs(conn_t *conn, cudaMemcpyKind kind);
extern void maybe_free_unified_mem(conn_t *conn, void *ptr);
extern void allocate_unified_mem_pointer(conn_t *conn, void *dev_ptr,
                                         size_t size);
extern void allocate_pinned_host_pointer(conn_t *conn, void *host_ptr,
                                          size_t size);
extern void maybe_free_pinned_host(conn_t *conn, void *ptr);
extern bool ensure_unified_ptr_mapped(conn_t *conn, void *ptr);
extern int is_unified_pointer(const int index, void *arg);
int maybe_copy_unified_arg(const int index, void *arg,
                           enum cudaMemcpyKind kind);

#define MAX_FUNCTION_NAME 1024
#define MAX_ARGS 128

#define FATBIN_FLAG_COMPRESS 0x0000000000002000LL

size_t decompress(const uint8_t *input, size_t input_size, uint8_t *output,
                  size_t output_size) {
  size_t ipos = 0, opos = 0;
  uint64_t next_nclen;  // length of next non-compressed segment
  uint64_t next_clen;   // length of next compressed segment
  uint64_t back_offset; // negative offset where redudant data is located,
                        // relative to current opos

  while (ipos < input_size) {
    next_nclen = (input[ipos] & 0xf0) >> 4;
    next_clen = 4 + (input[ipos] & 0xf);
    if (next_nclen == 0xf) {
      do {
        next_nclen += input[++ipos];
      } while (input[ipos] == 0xff);
    }

    if (memcpy(output + opos, input + (++ipos), next_nclen) == NULL) {
      fprintf(stderr, "Error copying data");
      return 0;
    }

    ipos += next_nclen;
    opos += next_nclen;
    if (ipos >= input_size || opos >= output_size) {
      break;
    }
    back_offset = input[ipos] + (input[ipos + 1] << 8);
    ipos += 2;
    if (next_clen == 0xf + 4) {
      do {
        next_clen += input[ipos++];
      } while (input[ipos - 1] == 0xff);
    }

    if (next_clen <= back_offset) {
      if (memcpy(output + opos, output + opos - back_offset, next_clen) ==
          NULL) {
        fprintf(stderr, "Error copying data");
        return 0;
      }
    } else {
      if (memcpy(output + opos, output + opos - back_offset, back_offset) ==
          NULL) {
        fprintf(stderr, "Error copying data");
        return 0;
      }
      for (size_t i = back_offset; i < next_clen; i++) {
        output[opos + i] = output[opos + i - back_offset];
      }
    }

    opos += next_clen;
  }
  return opos;
}

static ssize_t
decompress_single_section(const uint8_t *input, uint8_t **output,
                          size_t *output_size,
                          struct __cudaFatCudaBinary2HeaderRec *eh,
                          struct __cudaFatCudaBinary2EntryRec *th) {
  size_t padding;
  size_t input_read = 0;
  size_t output_written = 0;
  size_t decompress_ret = 0;
  const uint8_t zeroes[8] = {0};

  if (input == NULL || output == NULL || eh == NULL || th == NULL) {
    return 1;
  }

  uint8_t *mal = (uint8_t *)malloc(th->uncompressedBinarySize + 7);

  // add max padding of 7 bytes
  if ((*output = mal) == NULL) {
    goto error;
  }

  decompress_ret =
      decompress(input, th->binarySize, *output, th->uncompressedBinarySize);

  // @brodey - keeping this temporarily so that we can compare the compression
  // returns
  printf("decompressed return::: : %lx \n", decompress_ret);
  printf("compared return::: : %llx \n", th->uncompressedBinarySize);

  if (decompress_ret != th->uncompressedBinarySize) {
    std::cout << "failed actual decompress..." << std::endl;
    goto error;
  }
  input_read += th->binarySize;
  output_written += th->uncompressedBinarySize;

  padding = ((8 - (size_t)(input + input_read)) % 8);
  if (memcmp(input + input_read, zeroes, padding) != 0) {
    goto error;
  }
  input_read += padding;

  padding = ((8 - (size_t)th->uncompressedBinarySize) % 8);
  // Because we always allocated enough memory for one more elf_header and this
  // is smaller than the maximal padding of 7, we do not have to reallocate
  // here.
  memset(*output, 0, padding);
  output_written += padding;

  *output_size = output_written;
  return input_read;
error:
  free(*output);
  *output = NULL;
  return -1;
}

struct Function {
  char *name;
  void *fat_cubin;       // the fat cubin that this function is a part of.
  const char *host_func; // if registered, points at the host function.
  int *arg_sizes;
  int arg_count;
};

std::vector<Function> functions;

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
  cudaError_t return_value;

  conn_t *conn = rpc_client_get_connection(0);
  if (conn == NULL)
    return cudaErrorDevicesUnavailable;

  int request_id = rpc_write_start_request(conn, RPC_cudaMemcpy);
  if (request_id < 0 || rpc_write(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
    return cudaErrorDevicesUnavailable;

  // we need to swap device directions in this case
  switch (kind) {
  case cudaMemcpyDeviceToHost:
    // Ensure destination is mapped if it's a unified pointer
    ensure_unified_ptr_mapped(conn, dst);
    if (rpc_write(conn, &src, sizeof(void *)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 || rpc_read(conn, dst, count) < 0)
      return cudaErrorDevicesUnavailable;
    break;
  case cudaMemcpyHostToDevice:
    if (rpc_write(conn, &dst, sizeof(void *)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_write(conn, src, count) < 0 || rpc_wait_for_response(conn) < 0)
      return cudaErrorDevicesUnavailable;
    break;
  case cudaMemcpyDeviceToDevice:
    if (rpc_write(conn, &dst, sizeof(void *)) < 0 ||
        rpc_write(conn, &src, sizeof(void *)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0)
      return cudaErrorDevicesUnavailable;
    break;
  }

  if (rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
  cudaError_t return_value;

  conn_t *conn = rpc_client_get_connection(0);

  int request_id = rpc_write_start_request(conn, RPC_cudaMemcpyAsync);
  int stream_null_check = stream == 0 ? 1 : 0;
  if (request_id < 0 ||
      rpc_write(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
      rpc_write(conn, &stream_null_check, sizeof(int)) < 0 ||
      (stream_null_check == 0 &&
       rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0))
    return cudaErrorDevicesUnavailable;

  // we need to swap device directions in this case
  switch (kind) {
  case cudaMemcpyDeviceToHost: {
    // Ensure destination is mapped if it's a unified pointer
    ensure_unified_ptr_mapped(conn, dst);
    // Server will return a transfer_id; data comes asynchronously via polling
    uint64_t transfer_id;
    if (rpc_write(conn, &src, sizeof(void *)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &transfer_id, sizeof(uint64_t)) < 0)
      return cudaErrorDevicesUnavailable;

    if (rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
        rpc_read_end(conn) < 0)
      return cudaErrorDevicesUnavailable;

    // If successful, register the pending D2H transfer
    if (return_value == cudaSuccess && transfer_id != 0) {
      std::lock_guard<std::mutex> lock(g_callback_mutex);
      g_pending_d2h_transfers[transfer_id] = {dst, count};
      // Start polling thread to receive the data when ready
      start_callback_thread();
    }

    return return_value;
  }
  case cudaMemcpyHostToDevice:
    if (rpc_write(conn, &dst, sizeof(void *)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_write(conn, src, count) < 0 || rpc_wait_for_response(conn) < 0)
      return cudaErrorDevicesUnavailable;
    break;
  case cudaMemcpyDeviceToDevice:
    if (rpc_write(conn, &dst, sizeof(void *)) < 0 ||
        rpc_write(conn, &src, sizeof(void *)) < 0 ||
        rpc_write(conn, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(conn) < 0)
      return cudaErrorDevicesUnavailable;
    break;
  }

  if (rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "cudaSuccess: No errors";
  case cudaErrorInvalidValue:
    return "cudaErrorInvalidValue: Invalid value";
  case cudaErrorMemoryAllocation:
    return "cudaErrorMemoryAllocation: Out of memory";
  case cudaErrorInitializationError:
    return "cudaErrorInitializationError: Initialization error";
  case cudaErrorLaunchFailure:
    return "cudaErrorLaunchFailure: Launch failure";
  case cudaErrorPriorLaunchFailure:
    return "cudaErrorPriorLaunchFailure: Launch failure of a previous kernel";
  case cudaErrorLaunchTimeout:
    return "cudaErrorLaunchTimeout: Launch timed out";
  case cudaErrorLaunchOutOfResources:
    return "cudaErrorLaunchOutOfResources: Launch exceeded resources";
  case cudaErrorInvalidDeviceFunction:
    return "cudaErrorInvalidDeviceFunction: Invalid device function";
  case cudaErrorInvalidConfiguration:
    return "cudaErrorInvalidConfiguration: Invalid configuration";
  case cudaErrorInvalidDevice:
    return "cudaErrorInvalidDevice: Invalid device";
  case cudaErrorInvalidMemcpyDirection:
    return "cudaErrorInvalidMemcpyDirection: Invalid memory copy direction";
  case cudaErrorInsufficientDriver:
    return "cudaErrorInsufficientDriver: CUDA driver is insufficient for the "
           "runtime version";
  case cudaErrorMissingConfiguration:
    return "cudaErrorMissingConfiguration: Missing configuration";
  case cudaErrorNoDevice:
    return "cudaErrorNoDevice: No CUDA-capable device is detected";
  case cudaErrorArrayIsMapped:
    return "cudaErrorArrayIsMapped: Array is already mapped";
  case cudaErrorAlreadyMapped:
    return "cudaErrorAlreadyMapped: Resource is already mapped";
  case cudaErrorNoKernelImageForDevice:
    return "cudaErrorNoKernelImageForDevice: No kernel image is available for "
           "the device";
  case cudaErrorECCUncorrectable:
    return "cudaErrorECCUncorrectable: Uncorrectable ECC error detected";
  case cudaErrorSharedObjectSymbolNotFound:
    return "cudaErrorSharedObjectSymbolNotFound: Shared object symbol not "
           "found";
  case cudaErrorSharedObjectInitFailed:
    return "cudaErrorSharedObjectInitFailed: Shared object initialization "
           "failed";
  case cudaErrorUnsupportedLimit:
    return "cudaErrorUnsupportedLimit: Unsupported limit";
  case cudaErrorDuplicateVariableName:
    return "cudaErrorDuplicateVariableName: Duplicate global variable name";
  case cudaErrorDuplicateTextureName:
    return "cudaErrorDuplicateTextureName: Duplicate texture name";
  case cudaErrorDuplicateSurfaceName:
    return "cudaErrorDuplicateSurfaceName: Duplicate surface name";
  case cudaErrorDevicesUnavailable:
    return "cudaErrorDevicesUnavailable: All devices are busy or unavailable";
  case cudaErrorInvalidKernelImage:
    return "cudaErrorInvalidKernelImage: The kernel image is invalid";
  case cudaErrorInvalidSource:
    return "cudaErrorInvalidSource: The device kernel source is invalid";
  case cudaErrorFileNotFound:
    return "cudaErrorFileNotFound: File not found";
  case cudaErrorInvalidPtx:
    return "cudaErrorInvalidPtx: The PTX is invalid";
  case cudaErrorInvalidGraphicsContext:
    return "cudaErrorInvalidGraphicsContext: Invalid OpenGL or DirectX context";
  case cudaErrorInvalidResourceHandle:
    return "cudaErrorInvalidResourceHandle: Invalid resource handle";
  case cudaErrorNotReady:
    return "cudaErrorNotReady: CUDA operations are not ready";
  case cudaErrorIllegalAddress:
    return "cudaErrorIllegalAddress: An illegal memory access occurred";
  case cudaErrorInvalidPitchValue:
    return "cudaErrorInvalidPitchValue: Invalid pitch value";
  case cudaErrorInvalidSymbol:
    return "cudaErrorInvalidSymbol: Invalid symbol";
  case cudaErrorUnknown:
    return "cudaErrorUnknown: Unknown error";
  // Add any other CUDA error codes that are missing
  default:
    return "Unknown CUDA error";
  }
}

cudaError_t
cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaKernelNodeParams *pNodeParams) {
  conn_t *conn = rpc_client_get_connection(0);
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyHostToDevice) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;

  for (size_t i = 0;
       i < numDependencies && is_unified_pointer(conn, (void *)pDependencies);
       i++) {
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (maybe_copy_unified_arg(conn, (void *)pNodeParams,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;

  cudaError_t return_value;

  if (rpc_write_start_request(conn, RPC_cudaGraphAddKernelNode) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &graph, sizeof(cudaGraph_t)) < 0)
    return cudaErrorDevicesUnavailable;

  // Send dependencies individually
  for (size_t i = 0; i < numDependencies; ++i) {
    if (rpc_write(conn, &pDependencies[i], sizeof(cudaGraphNode_t)) < 0) {
      printf("Failed to write Dependency[%zu]\n", i);
      return cudaErrorDevicesUnavailable;
    }
  }

  std::cout << "callin cudaGraphAddKernelNode" << std::endl;

  Function *f = nullptr;
  for (auto &function : functions)
    if (function.host_func == pNodeParams->func) {
      f = &function;
    }

  if (f == nullptr || rpc_write(conn, &f->arg_count, sizeof(int)) < 0)
    return cudaErrorDevicesUnavailable;

  for (int i = 0; i < f->arg_count; ++i) {
    if (rpc_write(conn, &f->arg_sizes[i], sizeof(int)) < 0 ||
        rpc_write(conn, pNodeParams->kernelParams[i], f->arg_sizes[i]) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (rpc_write(conn, &pNodeParams,
                sizeof(const struct cudaKernelNodeParams *)) < 0 ||
      (pNodeParams != nullptr &&
       rpc_write(conn, pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  std::cout << "!!!! " << return_value << std::endl;

  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;

  for (size_t i = 0;
       i < numDependencies && is_unified_pointer(conn, (void *)pDependencies);
       i++) {
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (maybe_copy_unified_arg(conn, (void *)pNodeParams,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

#include <iostream>

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes,
                              size_t *numNodes) {
  cudaError_t return_value;
  conn_t *conn = rpc_client_get_connection(0);

  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)nodes, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)numNodes, cudaMemcpyHostToDevice) <
      0)
    return cudaErrorDevicesUnavailable;

  rpc_write_start_request(conn, RPC_cudaGraphGetNodes);
  rpc_write(conn, &graph, sizeof(cudaGraph_t));
  rpc_wait_for_response(conn);

  // Read values
  rpc_read(conn, &nodes, sizeof(cudaGraphNode_t));
  rpc_read(conn, numNodes, sizeof(size_t));
  rpc_read(conn, &return_value, sizeof(cudaError_t));
  rpc_read_end(conn);

  // Log values after reading
  std::cout << "cudaGraphGetNodes response:" << std::endl;
  std::cout << "  - Graph: " << graph << std::endl;
  std::cout << "  - Nodes pointer: " << nodes << std::endl;
  std::cout << "  - Number of nodes: " << *numNodes << std::endl;
  std::cout << "  - Return value: " << return_value << std::endl;

  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)nodes, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)numNodes, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  cudaError_t return_value;
  cudaError_t memcpy_return;

  // First check if function is registered before making any RPC calls
  Function *f = nullptr;
  for (auto &function : functions) {
    if (function.host_func == func)
      f = &function;
  }

  if (f == nullptr) {
    // Return error without making RPC call - function not registered
    return cudaErrorInvalidDeviceFunction;
  }

  conn_t *conn = rpc_client_get_connection(0);

  memcpy_return = cuda_memcpy_unified_ptrs(conn, cudaMemcpyHostToDevice);
  if (memcpy_return != cudaSuccess)
    return memcpy_return;

  // Start the RPC request
  int request_id = rpc_write_start_request(conn, RPC_cudaLaunchKernel);
  if (request_id < 0 || rpc_write(conn, &func, sizeof(const void *)) < 0 ||
      rpc_write(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0)
    return cudaErrorDevicesUnavailable;

  if (rpc_write(conn, &f->arg_count, sizeof(int)) < 0)
    return cudaErrorDevicesUnavailable;

  for (int i = 0; i < f->arg_count; ++i) {
    if (rpc_write(conn, &f->arg_sizes[i], sizeof(int)) < 0 ||
        rpc_write(conn, args[i], f->arg_sizes[i]) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (rpc_wait_for_response(conn) < 0) {
    return cudaErrorDevicesUnavailable;
  }

  if (rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  memcpy_return = cuda_memcpy_unified_ptrs(conn, cudaMemcpyDeviceToHost);
  if (memcpy_return != cudaSuccess) {
    return memcpy_return;
  }

  return return_value;
}

// Wrapper for __cudaLaunchKernel which is used by CUDA 13.x compiled binaries
extern "C" cudaError_t __cudaLaunchKernel(
    const void *kernel, dim3 gridDim, dim3 blockDim,
    void **args, size_t sharedMem, cudaStream_t stream) {
  return cudaLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
}

// Function to calculate byte size based on PTX data type
int get_type_size(const char *type) {
  if (*type == 'u' || *type == 's' || *type == 'f')
    type++;
  else
    return 0; // Unknown type
  if (*type == '8')
    return 1;
  if (*type == '1' && *(type + 1) == '6')
    return 2;
  if (*type == '3' && *(type + 1) == '2')
    return 4;
  if (*type == '6' && *(type + 1) == '4')
    return 8;
  return 0; // Unknown type
}

// Function to parse a PTX string and fill functions into a dynamically
// allocated array
void parse_ptx_string(void *fatCubin, const char *ptx_string,
                      unsigned long long ptx_len) {
  // for this entire function we work with offsets to avoid risky pointer stuff.
  for (unsigned long long i = 0; i < ptx_len; i++) {
    // check if this token is an entry.
    if (ptx_string[i] != '.' || i + 5 >= ptx_len ||
        strncmp(ptx_string + i + 1, "entry", strlen("entry")) != 0)
      continue;

    char *name = new char[MAX_FUNCTION_NAME];
    int *arg_sizes = new int[MAX_ARGS];
    int arg_count = 0;

    // find the next non a-zA-Z0-9_ character
    i += strlen(".entry");
    while (i < ptx_len && !isalnum(ptx_string[i]) && ptx_string[i] != '_')
      i++;

    // now we're pointing at the start of the name, copy the name to the
    // function
    int j = 0;
    for (; j < MAX_FUNCTION_NAME - 1 && i < ptx_len &&
           (isalnum(ptx_string[i]) || ptx_string[i] == '_');)
      name[j++] = ptx_string[i++];
    name[j] = '\0';

    // find the next ( character to demarcate the arg start or { to demarcate
    // the function body
    while (i < ptx_len && ptx_string[i] != '(' && ptx_string[i] != '{')
      i++;

    if (ptx_string[i] == '(') {
      // parse out the args-list
      for (; arg_count < MAX_ARGS; arg_count++) {
        int arg_size = 0;

        // read until a . is found or )
        while (i < ptx_len && (ptx_string[i] != '.' && ptx_string[i] != ')'))
          i++;

        if (ptx_string[i] == ')')
          break;

        // assert that the next token is "param"
        if (i + 5 >= ptx_len ||
            strncmp(ptx_string + i, ".param", strlen(".param")) != 0)
          continue;

        while (true) {
          // read the arguments list

          // read until a . , ) or [
          while (i < ptx_len && (ptx_string[i] != '.' && ptx_string[i] != ',' &&
                                 ptx_string[i] != ')' && ptx_string[i] != '['))
            i++;

          if (ptx_string[i] == '.') {
            // read the type, ignoring if it's not a valid type
            int type_size = get_type_size(ptx_string + (++i));
            if (type_size == 0)
              continue;
            arg_size = type_size;
          } else if (ptx_string[i] == '[') {
            // this is an array type. read until the ]
            int start = i + 1;
            while (i < ptx_len && ptx_string[i] != ']')
              i++;

            // parse the int value
            int n = 0;
            for (int j = start; j < i; j++)
              n = n * 10 + ptx_string[j] - '0';
            arg_size *= n;
          } else if (ptx_string[i] == ',' || ptx_string[i] == ')')
            // end of this argument
            break;
        }

        arg_sizes[arg_count] = arg_size;
      }
    }

    // add the function to the list
    functions.push_back(Function{
        .name = name,
        .fat_cubin = fatCubin,
        .host_func = nullptr,
        .arg_sizes = arg_sizes,
        .arg_count = arg_count,
    });
  }
}

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
  void **p;
  int return_value;

  conn_t *conn = rpc_client_get_connection(0);

  if (rpc_write_start_request(conn, RPC___cudaRegisterFatBinary) < 0)
    return nullptr;

  if (*(unsigned *)fatCubin == __cudaFatMAGIC2) {
    __cudaFatCudaBinary2 *binary = (__cudaFatCudaBinary2 *)fatCubin;

    if (rpc_write(conn, binary, sizeof(__cudaFatCudaBinary2)) < 0)
      return nullptr;

    __cudaFatCudaBinary2Header *header =
        (__cudaFatCudaBinary2Header *)binary->text;

    unsigned long long size = sizeof(__cudaFatCudaBinary2Header) + header->size;

    if (rpc_write(conn, &size, sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, header, size) < 0)
      return nullptr;

    // also parse the ptx file from the fatbin to store the parameter sizes for
    // the assorted functions
    char *base = (char *)(header + 1);
    long long unsigned int offset = 0;
    __cudaFatCudaBinary2EntryRec *entry =
        (__cudaFatCudaBinary2EntryRec *)(base);

    while (offset < header->size) {
      entry = (__cudaFatCudaBinary2EntryRec *)(base + offset);
      offset += entry->binary + entry->binarySize;

      if (!(entry->type & FATBIN_2_PTX))
        continue;

      // if compress flag exists, we should decompress before parsing the ptx
      if (entry->flags & FATBIN_FLAG_COMPRESS) {
        uint8_t *text_data = NULL;
        size_t text_data_size = 0;

        std::cout << "decompression required; starting decompress..."
                  << std::endl;

        if (decompress_single_section((const uint8_t *)entry + entry->binary,
                                      &text_data, &text_data_size, header,
                                      entry) < 0) {
          std::cout << "decompressing failed..." << std::endl;
          return nullptr;
        }

        parse_ptx_string(fatCubin, (char *)text_data, text_data_size);
      } else {
        parse_ptx_string(fatCubin, (char *)entry + entry->binary,
                         entry->binarySize);
      }
    }
  }

  if (rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &p, sizeof(void **)) < 0 || rpc_read_end(conn) < 0)
    return nullptr;

  return p;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  fprintf(stderr, "SCUDA: __cudaRegisterFatBinaryEnd called\n");
  void *return_value;

  conn_t *conn = rpc_client_get_connection(0);

  int request_id =
      rpc_write_start_request(conn, RPC___cudaRegisterFatBinaryEnd);
  if (request_id < 0 ||
      rpc_write(conn, &fatCubinHandle, sizeof(const void *)) < 0 ||
      rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0)
    return;
  fprintf(stderr, "SCUDA: __cudaRegisterFatBinaryEnd done\n");
}

extern "C" void __cudaInitModule(void **fatCubinHandle) {
  // No-op - CUDA runtime initializes modules internally
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  //   std::cout << "__cudaUnregisterFatBinary writing data..." << std::endl;
}

extern "C" cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  cudaError_t res;

  // Debug: Calling __cudaPushCallConfiguration

  conn_t *conn = rpc_client_get_connection(0);

  if (rpc_write_start_request(conn, RPC___cudaPushCallConfiguration) < 0 ||
      rpc_write(conn, &gridDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &blockDim, sizeof(dim3)) < 0 ||
      rpc_write(conn, &sharedMem, sizeof(size_t)) < 0 ||
      rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &res, sizeof(cudaError_t)) < 0 || rpc_read_end(conn) < 0) {
    return cudaErrorDevicesUnavailable;
  }

  return res;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                  size_t *sharedMem,
                                                  cudaStream_t *stream) {
  cudaError_t res;

  conn_t *conn = rpc_client_get_connection(0);

  if (rpc_write_start_request(conn, RPC___cudaPopCallConfiguration) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, gridDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, blockDim, sizeof(dim3)) < 0 ||
      rpc_read(conn, sharedMem, sizeof(size_t)) < 0 ||
      rpc_read(conn, stream, sizeof(cudaStream_t)) < 0 ||
      rpc_read(conn, &res, sizeof(cudaError_t)) < 0 || rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  return res;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize) {
  void *return_value;

  size_t deviceFunLen = strlen(deviceFun) + 1;
  size_t deviceNameLen = strlen(deviceName) + 1;

  uint8_t mask = 0;
  if (tid != nullptr)
    mask |= 1 << 0;
  if (bid != nullptr)
    mask |= 1 << 1;
  if (bDim != nullptr)
    mask |= 1 << 2;
  if (gDim != nullptr)
    mask |= 1 << 3;
  if (wSize != nullptr)
    mask |= 1 << 4;

  conn_t *conn = rpc_client_get_connection(0);

  if (rpc_write_start_request(conn, RPC___cudaRegisterFunction) < 0 ||
      rpc_write(conn, &fatCubinHandle, sizeof(void **)) < 0 ||
      rpc_write(conn, &hostFun, sizeof(const char *)) < 0 ||
      rpc_write(conn, &deviceFunLen, sizeof(size_t)) < 0 ||
      rpc_write(conn, deviceFun, deviceFunLen) < 0 ||
      rpc_write(conn, &deviceNameLen, sizeof(size_t)) < 0 ||
      rpc_write(conn, deviceName, deviceNameLen) < 0 ||
      rpc_write(conn, &thread_limit, sizeof(int)) < 0 ||
      rpc_write(conn, &mask, sizeof(uint8_t)) < 0 ||
      (tid != nullptr && rpc_write(conn, tid, sizeof(uint3)) < 0) ||
      (bid != nullptr && rpc_write(conn, bid, sizeof(uint3)) < 0) ||
      (bDim != nullptr && rpc_write(conn, bDim, sizeof(dim3)) < 0) ||
      (gDim != nullptr && rpc_write(conn, gDim, sizeof(dim3)) < 0) ||
      (wSize != nullptr && rpc_write(conn, wSize, sizeof(int)) < 0) ||
      rpc_wait_for_response(conn) < 0)
    return;

  // Receive param count and sizes from server
  int param_count = 0;
  if (rpc_read(conn, &param_count, sizeof(int)) < 0) {
    rpc_read_end(conn);
    return;
  }

  int *arg_sizes = new int[param_count > 0 ? param_count : 1];
  for (int i = 0; i < param_count; i++) {
    if (rpc_read(conn, &arg_sizes[i], sizeof(int)) < 0) {
      delete[] arg_sizes;
      rpc_read_end(conn);
      return;
    }
  }

  if (rpc_read_end(conn) < 0) {
    delete[] arg_sizes;
    return;
  }

  // Store the function info received from server
  char *name = new char[strlen(deviceName) + 1];
  strcpy(name, deviceName);

  functions.push_back(Function{
      .name = name,
      .fat_cubin = fatCubinHandle,
      .host_func = hostFun,
      .arg_sizes = arg_sizes,
      .arg_count = param_count,
  });
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  void *return_value;

  conn_t *conn = rpc_client_get_connection(0);

  int request_id = rpc_write_start_request(conn, RPC___cudaRegisterVar);
  if (request_id < 0) {
    std::cerr << "Failed to start RPC request" << std::endl;
    return;
  }

  if (rpc_write(conn, &fatCubinHandle, sizeof(void *)) < 0) {
    std::cerr << "Failed writing fatCubinHandle" << std::endl;
    return;
  }

  // Send hostVar as pointer value (not as string - it's a memory address)
  if (rpc_write(conn, &hostVar, sizeof(void *)) < 0) {
    std::cerr << "Failed writing hostVar" << std::endl;
    return;
  }

  // Send deviceAddress as pointer value (not as string - it's a memory address)
  if (rpc_write(conn, &deviceAddress, sizeof(void *)) < 0) {
    std::cerr << "Failed writing deviceAddress" << std::endl;
    return;
  }

  // Send deviceName length and data (this IS a string)
  size_t deviceNameLen = strlen(deviceName) + 1;
  if (rpc_write(conn, &deviceNameLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to send deviceName length" << std::endl;
    return;
  }
  if (rpc_write(conn, deviceName, deviceNameLen) < 0) {
    std::cerr << "Failed writing deviceName" << std::endl;
    return;
  }

  // Write the rest of the arguments
  if (rpc_write(conn, &ext, sizeof(int)) < 0) {
    std::cerr << "Failed writing ext" << std::endl;
    return;
  }

  if (rpc_write(conn, &size, sizeof(size_t)) < 0) {
    std::cerr << "Failed writing size" << std::endl;
    return;
  }

  if (rpc_write(conn, &constant, sizeof(int)) < 0) {
    std::cerr << "Failed writing constant" << std::endl;
    return;
  }

  if (rpc_write(conn, &global, sizeof(int)) < 0) {
    std::cerr << "Failed writing global" << std::endl;
    return;
  }

  // Wait for a response from the server
  if (rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0) {
    std::cerr << "Failed waiting for response" << std::endl;
    return;
  }
}

cudaError_t cudaFree(void *devPtr) {
  cudaError_t return_value;

  conn_t *conn = rpc_client_get_connection(0);

  maybe_free_unified_mem(conn, devPtr);

  if (rpc_write_start_request(conn, RPC_cudaFree) < 0 ||
      rpc_write(conn, &devPtr, sizeof(void *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
  void *d_mem;

  conn_t *conn = rpc_client_get_connection(0);

  cudaError_t err = cudaMalloc((void **)&d_mem, size);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
    return err;
  }

  std::cout << "allocated unified device mem " << d_mem << " size: " << size
            << std::endl;

  allocate_unified_mem_pointer(conn, d_mem, size);

  *devPtr = d_mem;

  return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *ptr) {
  /**
   * The benefit of page-locked mem is to:
   *  "automatically accelerate calls to functions such as cudaMemcpy().
   *  Since the memory can be accessed directly by the device, it can be read or
   * written with much higher bandwidth than pageable memory that has not been
   * registered".
   *
   * Given that Scuda memcpy's happen over the wire, so there's no real benefit
   * to creating page-locked memory on scuda clients.
   * @TODO: ponder this more down the road.
   */
  return cudaSuccess;
}

cudaError_t cudaHostRegister(void *devPtr, size_t size, unsigned int flags) {
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
    conn_t *conn = rpc_client_get_connection(0);
    if (maybe_copy_unified_arg(conn, (void*)pHost, cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
    if (maybe_copy_unified_arg(conn, (void*)&size, cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
    if (maybe_copy_unified_arg(conn, (void*)&flags, cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
    cudaError_t return_value;
    if (rpc_write_start_request(conn, RPC_cudaHostAlloc) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, pHost, sizeof(void*)) < 0 ||
        rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
        rpc_read_end(conn) < 0)
        return cudaErrorDevicesUnavailable;
    if (maybe_copy_unified_arg(conn, (void*)pHost, cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
    if (maybe_copy_unified_arg(conn, (void*)&size, cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
    if (maybe_copy_unified_arg(conn, (void*)&flags, cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;

    // Register with pinned host memory tracking for segfault handling.
    // This allows the client to access the memory on-demand via segfault handler.
    // IMPORTANT: This is NOT unified memory - do NOT sync during kernel launches.
    if (return_value == cudaSuccess) {
      allocate_pinned_host_pointer(conn, *pHost, size);
    }

    return return_value;
}

cudaError_t cudaMallocHost(void** ptr, size_t size)
{
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

cudaError_t cudaFreeHost(void* ptr)
{
    conn_t *conn = rpc_client_get_connection(0);

    // Clean up memory tracking (both unified and pinned host)
    maybe_free_unified_mem(conn, ptr);
    maybe_free_pinned_host(conn, ptr);

    if (maybe_copy_unified_arg(conn, (void*)&ptr, cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
    cudaError_t return_value;
    if (rpc_write_start_request(conn, RPC_cudaFreeHost) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
        rpc_read_end(conn) < 0)
        return cudaErrorDevicesUnavailable;
    if (maybe_copy_unified_arg(conn, (void*)&ptr, cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
    return return_value;
}


cudaError_t
cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                       const cudaGraphNode_t *pDependencies,
                       size_t numDependencies,
                       const struct cudaMemcpy3DParms *pCopyParams) {
  conn_t *conn = rpc_client_get_connection(0);
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyHostToDevice) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++)
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pCopyParams,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  cudaError_t return_value;
  if (rpc_write_start_request(conn, RPC_cudaGraphAddMemcpyNode) < 0 ||
          rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
          rpc_write(conn, &graph, sizeof(cudaGraph_t)) < 0 || [=]()
          -> bool {
        for (size_t i = 0; i < numDependencies; ++i) {
          if (rpc_write(conn, &pDependencies[i],
                        sizeof(const cudaGraphNode_t)) < 0) {
            printf("Failed to write Dependency[%zu]\n", i);
            return false;
          }
        }
        return true;
      }() == false ||
                 rpc_write(conn, &pCopyParams,
                           sizeof(const struct cudaMemcpy3DParms *)) < 0 ||
                 (pCopyParams != nullptr &&
                  rpc_write(conn, pCopyParams,
                            sizeof(const struct cudaMemcpy3DParms)) < 0) ||
                 rpc_wait_for_response(conn) < 0 ||
                 rpc_read(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
                 rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
                 rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++)
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pCopyParams,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  printf("done calling cudaGraphAddMemcpyNode\n");
  return return_value;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                 const cudaGraphNode_t *pDependencies,
                                 size_t numDependencies,
                                 const struct cudaHostNodeParams *pNodeParams) {
  conn_t *conn = rpc_client_get_connection(0);
  add_host_node((void *)pNodeParams->fn, (void *)pNodeParams->userData);
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyHostToDevice) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++)
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pNodeParams,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  cudaError_t return_value;
  if (rpc_write_start_request(conn, RPC_cudaGraphAddHostNode) < 0 ||
          rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
          rpc_write(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
          rpc_write(conn, &graph, sizeof(cudaGraph_t)) < 0 || [=]()
          -> bool {
        for (size_t i = 0; i < numDependencies; ++i) {
          if (rpc_write(conn, &pDependencies[i],
                        sizeof(const cudaGraphNode_t)) < 0) {
            printf("Failed to write Dependency[%zu]\n", i);
            return false;
          }
        }
        return true;
      }() == false ||
                 rpc_write(conn, &pNodeParams,
                           sizeof(const struct cudaHostNodeParams *)) < 0 ||
                 (pNodeParams != nullptr &&
                  rpc_write(conn, pNodeParams,
                            sizeof(const struct cudaHostNodeParams)) < 0) ||
                 rpc_wait_for_response(conn) < 0 ||
                 rpc_read(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
                 rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
                 rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++)
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pNodeParams,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  return return_value;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  conn_t *conn = rpc_client_get_connection(0);
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  cudaError_t return_value;
  if (rpc_write_start_request(conn, RPC_cudaGraphDestroy) < 0 ||
      rpc_write(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  return return_value;
}

cudaError_t
cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                         const cudaGraphNode_t *pDependencies,
                         size_t numDependencies,
                         struct cudaMemAllocNodeParams *nodeParams) {
  void *dptr;
  conn_t *conn = rpc_client_get_connection(0);
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyHostToDevice) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++)
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)nodeParams, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;

  cudaError_t return_value;

  if (rpc_write_start_request(conn, RPC_cudaGraphAddMemAllocNode) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &graph, sizeof(cudaGraph_t)) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(struct cudaMemAllocNodeParams *)) <
          0 ||
      (nodeParams != nullptr &&
       rpc_write(conn, nodeParams, sizeof(struct cudaMemAllocNodeParams)) <
           0)) {
    return cudaErrorDevicesUnavailable;
  }

  if (numDependencies == 1) {
    if (rpc_write(conn, pDependencies, sizeof(cudaGraphNode_t)) < 0) {
      printf("Failed to write single dependency!\n");
      return cudaErrorDevicesUnavailable;
    }
  } else {
    for (size_t i = 0; i < numDependencies; ++i) {
      if (rpc_write(conn, pDependencies + i, sizeof(cudaGraphNode_t)) < 0) {
        printf("Failed to write Dependency[%zu]\n", i);
        return cudaErrorDevicesUnavailable;
      }
    }
  }

  if (rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &dptr, sizeof(void *)) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++)
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyDeviceToHost) < 0)
      return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)nodeParams, cudaMemcpyDeviceToHost) <
      0)
    return cudaErrorDevicesUnavailable;

  nodeParams->dptr = dptr;
  return return_value;
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode,
                                    cudaGraph_t graph,
                                    const cudaGraphNode_t *pDependencies,
                                    size_t numDependencies, void *dptr) {
  conn_t *conn = rpc_client_get_connection(0);

  if (numDependencies > 0 && pDependencies == nullptr) {
    printf("Error: pDependencies is NULL when numDependencies > 0\n");
    return cudaErrorInvalidValue;
  }

  if (maybe_copy_unified_arg(conn, (void *)&numDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pGraphNode, cudaMemcpyHostToDevice) <
      0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&graph, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)pDependencies,
                             cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;

  for (int i = 0; i < static_cast<int>(numDependencies) &&
                  is_unified_pointer(conn, (void *)pDependencies);
       i++) {
    if (maybe_copy_unified_arg(conn, (void *)&pDependencies[i],
                               cudaMemcpyHostToDevice) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (maybe_copy_unified_arg(conn, (void *)dptr, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;

  cudaError_t return_value;

  if (rpc_write_start_request(conn, RPC_cudaGraphAddMemFreeNode) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &graph, sizeof(cudaGraph_t)) < 0) {
    printf("RPC Write failed before dependencies!\n");
    return cudaErrorDevicesUnavailable;
  }

  if (numDependencies == 1) {
    if (rpc_write(conn, pDependencies, sizeof(cudaGraphNode_t)) < 0) {
      printf("Failed to write single dependency!\n");
      return cudaErrorDevicesUnavailable;
    }
  } else {
    printf("Writing %zu dependencies...\n", numDependencies);
    for (size_t i = 0; i < numDependencies; ++i) {
      if (rpc_write(conn, pDependencies + i, sizeof(cudaGraphNode_t)) < 0) {
        printf("Failed to write Dependency[%zu]\n", i);
        return cudaErrorDevicesUnavailable;
      }
    }
  }

  if (rpc_write(conn, &dptr, sizeof(void *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0) {
    return cudaErrorDevicesUnavailable;
  }

  return return_value;
}

cudaError_t cudaDeviceGetGraphMemAttribute(int device,
                                           enum cudaGraphMemAttributeType attr,
                                           void *value) {
  conn_t *conn = rpc_client_get_connection(0);
  if (maybe_copy_unified_arg(conn, (void *)&device, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&attr, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)value, cudaMemcpyHostToDevice) < 0)
    return cudaErrorDevicesUnavailable;
  cudaError_t return_value;
  if (rpc_write_start_request(conn, RPC_cudaDeviceGetGraphMemAttribute) < 0 ||
      rpc_write(conn, &device, sizeof(int)) < 0 ||
      rpc_write(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, sizeof(void *)) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&device, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)&attr, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  if (maybe_copy_unified_arg(conn, (void *)value, cudaMemcpyDeviceToHost) < 0)
    return cudaErrorDevicesUnavailable;
  return return_value;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                     const struct cudaResourceDesc *pResDesc,
                                     const struct cudaTextureDesc *pTexDesc,
                                     const struct cudaResourceViewDesc *pResViewDesc) {
  conn_t *conn = rpc_client_get_connection(0);
  cudaError_t return_value;

  if (rpc_write_start_request(conn, RPC_cudaCreateTextureObject) < 0 ||
      rpc_write(conn, &pResDesc, sizeof(const struct cudaResourceDesc*)) < 0 ||
      (pResDesc != nullptr && rpc_write(conn, pResDesc, sizeof(struct cudaResourceDesc)) < 0) ||
      rpc_write(conn, &pTexDesc, sizeof(const struct cudaTextureDesc*)) < 0 ||
      (pTexDesc != nullptr && rpc_write(conn, pTexDesc, sizeof(struct cudaTextureDesc)) < 0) ||
      rpc_write(conn, &pResViewDesc, sizeof(const struct cudaResourceViewDesc*)) < 0 ||
      (pResViewDesc != nullptr && rpc_write(conn, pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pTexObject, sizeof(cudaTextureObject_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
  conn_t *conn = rpc_client_get_connection(0);
  cudaError_t return_value;

  // Calculate data size for srcPtr if it has data
  size_t src_data_size = 0;
  if (p->srcPtr.ptr != nullptr) {
    // Data size = pitch * height * depth
    src_data_size = p->srcPtr.pitch * p->extent.height * p->extent.depth;
  }

  // Calculate data size for dstPtr if it will receive data
  size_t dst_data_size = 0;
  if (p->dstPtr.ptr != nullptr) {
    dst_data_size = p->dstPtr.pitch * p->extent.height * p->extent.depth;
  }

  if (rpc_write_start_request(conn, RPC_cudaMemcpy3D) < 0 ||
      rpc_write(conn, p, sizeof(struct cudaMemcpy3DParms)) < 0 ||
      rpc_write(conn, &src_data_size, sizeof(size_t)) < 0)
    return cudaErrorDevicesUnavailable;

  // If copying from host memory, send the source data
  if (src_data_size > 0 &&
      (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyHostToHost)) {
    if (rpc_write(conn, p->srcPtr.ptr, src_data_size) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0)
    return cudaErrorDevicesUnavailable;

  // If copying to host memory, receive the destination data
  if (dst_data_size > 0 && return_value == cudaSuccess &&
      (p->kind == cudaMemcpyDeviceToHost || p->kind == cudaMemcpyHostToHost)) {
    if (rpc_read(conn, p->dstPtr.ptr, dst_data_size) < 0)
      return cudaErrorDevicesUnavailable;
  }

  if (rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  return return_value;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData, unsigned int flags) {
  fprintf(stderr, "SCUDA: cudaStreamAddCallback called stream=%p callback=%p\n",
          (void *)stream, (void *)callback);
  fflush(stderr);

  conn_t *conn = rpc_client_get_connection(0);
  cudaError_t return_value;

  // Generate a unique callback ID
  uint64_t callback_id = g_callback_id_counter.fetch_add(1);

  // Store the callback info locally
  {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_pending_callbacks[callback_id] = {callback, userData};
  }

  // Send to server: stream, callback_id, flags
  if (rpc_write_start_request(conn, RPC_cudaStreamAddCallback) < 0 ||
      rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_write(conn, &callback_id, sizeof(uint64_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0) {
    // Remove callback on failure
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_pending_callbacks.erase(callback_id);
    return cudaErrorDevicesUnavailable;
  }

  if (return_value != cudaSuccess) {
    // Remove callback on failure
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    g_pending_callbacks.erase(callback_id);
  } else {
    // Start background polling thread to handle async callbacks
    start_callback_thread();
  }

  return return_value;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  fprintf(stderr, "SCUDA: cudaStreamSynchronize called stream=%p\n", (void *)stream);
  fflush(stderr);

  conn_t *conn = rpc_client_get_connection(0);
  cudaError_t return_value;

  // First synchronize the stream on the server
  if (rpc_write_start_request(conn, RPC_cudaStreamSynchronize) < 0 ||
      rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  // After synchronization, poll for any pending callbacks and D2H transfers
  cudaError_t poll_result = poll_stream_callbacks(conn);
  if (poll_result != cudaSuccess)
    return poll_result;

  return return_value;
}

cudaError_t cudaDeviceSynchronize() {
  conn_t *conn = rpc_client_get_connection(0);
  cudaError_t return_value;

  // First synchronize the device on the server
  if (rpc_write_start_request(conn, RPC_cudaDeviceSynchronize) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  // After synchronization, poll for any pending callbacks and D2H transfers
  cudaError_t poll_result = poll_stream_callbacks(conn);
  if (poll_result != cudaSuccess)
    return poll_result;

  return return_value;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  conn_t *conn = rpc_client_get_connection(0);
  cudaError_t return_value;

  // First synchronize the event on the server
  if (rpc_write_start_request(conn, RPC_cudaEventSynchronize) < 0 ||
      rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(cudaError_t)) < 0 ||
      rpc_read_end(conn) < 0)
    return cudaErrorDevicesUnavailable;

  // After synchronization, poll for any pending callbacks and D2H transfers
  cudaError_t poll_result = poll_stream_callbacks(conn);
  if (poll_result != cudaSuccess)
    return poll_result;

  return return_value;
}
