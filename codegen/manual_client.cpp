#include <cassert>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <nvml.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "gen_api.h"
#include "ptx_fatbin.hpp"
#include "rpc.h"
#include "parse_elf.h"

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
  printf("decompressed return::: : %x \n", decompress_ret);
  printf("compared return::: : %x \n", th->uncompressedBinarySize);

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
  case cudaMemcpyDeviceToHost:
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

  Function *f = nullptr;
  for (auto &function : functions)
    if (function.host_func == func)
      f = &function;

  if (f == nullptr || rpc_write(conn, &f->arg_count, sizeof(int)) < 0)
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
  if (memcpy_return != cudaSuccess)
    return memcpy_return;

  return return_value;
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
      if (entry->type & 2) {
        process_elf((const unsigned char *)entry + entry->binary,
                    entry->binarySize, entry->flags & FATBIN_FLAG_COMPRESS, 
                    functions, MAX_FUNCTION_NAME, MAX_ARGS);
      }
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

        // verify the decompressed output looks right; we should run
        // --no-compress with nvcc before running this decompression logic to
        // compare outputs.
        for (int i = 0; i < text_data_size; i++)
          std::cout << *(char *)((char *)text_data + i);
        std::cout << std::endl;

        parse_ptx_string(fatCubin, (char *)text_data, text_data_size);
      } else {
        // print the entire ptx file for debugging
        for (int i = 0; i < entry->binarySize; i++)
          std::cout << *(char *)((char *)entry + entry->binary + i);
        std::cout << std::endl;

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
  void *return_value;

  conn_t *conn = rpc_client_get_connection(0);

  int request_id =
      rpc_write_start_request(conn, RPC___cudaRegisterFatBinaryEnd);
  if (request_id < 0 ||
      rpc_write(conn, &fatCubinHandle, sizeof(const void *)) < 0 ||
      rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0)
    return;
}

extern "C" void __cudaInitModule(void **fatCubinHandle) {
  std::cout << "__cudaInitModule writing data..." << std::endl;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  //   std::cout << "__cudaUnregisterFatBinary writing data..." << std::endl;
}

extern "C" cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  cudaError_t res;

  std::cout << "Calling __cudaPushCallConfiguration" << std::endl;

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
      rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0)
    return;

  // also memorize the host pointer function
  for (auto &function : functions)
    if (strcmp(function.name, deviceName) == 0)
      function.host_func = hostFun;
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

  // Send hostVar length and data
  size_t hostVarLen = strlen(hostVar) + 1;
  if (rpc_write(conn, &hostVarLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to send hostVar length" << std::endl;
    return;
  }
  if (rpc_write(conn, hostVar, hostVarLen) < 0) {
    std::cerr << "Failed writing hostVar" << std::endl;
    return;
  }

  // Send deviceAddress length and data
  size_t deviceAddressLen = strlen(deviceAddress) + 1;
  if (rpc_write(conn, &deviceAddressLen, sizeof(size_t)) < 0) {
    std::cerr << "Failed to send deviceAddress length" << std::endl;
    return;
  }
  if (rpc_write(conn, deviceAddress, deviceAddressLen) < 0) {
    std::cerr << "Failed writing deviceAddress" << std::endl;
    return;
  }

  // Send deviceName length and data
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

cudaError_t cudaMallocHost(void **ptr, size_t size) {
  *ptr = (void *)malloc(size);

  return cudaSuccess;
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
