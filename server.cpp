#include <arpa/inet.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include <dlfcn.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <vector>

#include <list>
#include <map>

#include "codegen/gen_server.h"
#include "codegen/gen_api.h"
#include "rpc.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx,
                                           unsigned int flags, CUdevice dev);

struct scuda_fatbin_wrapper {
  uint32_t magic;
  uint32_t version;
  const void *data;
  const void *filename_or_fatbins;
};

static constexpr uint32_t SCUDA_FATBINC_MAGIC = 0x466243b1;
static constexpr uint32_t SCUDA_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t SCUDA_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t SCUDA_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr int SCUDA_RPC_cuFuncGetParamLayout = 1000001;
static constexpr int SCUDA_RPC_cuCtxCreate_v2 = 1000002;
static constexpr int SCUDA_RPC_cuMemPoolSetAttribute = 1000003;
static constexpr int SCUDA_RPC_cuMemPoolGetAttribute = 1000004;
static constexpr int SCUDA_RPC_cuLaunchHostFunc = 1000005;
static constexpr int SCUDA_RPC_cuPointerGetAttribute = 1000006;
static constexpr int SCUDA_RPC_cuGetExportTableMetadata = 1000007;
static constexpr int SCUDA_RPC_cuPrivateGetModuleNode = 1000008;
static constexpr int SCUDA_RPC_cuPointerGetAttributes = 1000009;
static constexpr int SCUDA_RPC_cuStreamAddCallback = 1000010;
static constexpr int SCUDA_RPC_cuGraphConditionalHandleCreate = 1000011;
static constexpr int SCUDA_RPC_cuGraphAddNode_v2 = 1000012;
static constexpr int SCUDA_RPC_cuStreamBeginCaptureToGraph = 1000013;
static constexpr int SCUDA_RPC_cuStreamUpdateCaptureDependencies_v2 = 1000014;
static constexpr int SCUDA_RPC_cuStreamGetCaptureInfo_v3 = 1000015;
static constexpr int SCUDA_RPC_cuDeviceGetGraphMemAttribute = 1000016;
static constexpr int SCUDA_RPC_cuDeviceSetGraphMemAttribute = 1000017;
static constexpr int SCUDA_RPC_cuLinkAddData_v2 = 1000018;
static constexpr uint32_t SCUDA_PRIVATE_EXPORT_MAX_SLOTS = 256;

struct scuda_kernel_param_layout {
  uint32_t count = 0;
  size_t offsets[64] = {};
  size_t sizes[64] = {};
};

struct scuda_private_module_node_capture {
  void *node = nullptr;
  uint64_t owner = 0;
  uint64_t count = 0;
};

static bool scuda_pointer_attribute_size(CUpointer_attribute attr,
                                         size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_POINTER_ATTRIBUTE_CONTEXT:
    *size = sizeof(CUcontext);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    *size = sizeof(void *);
    return true;
  case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
    *size = sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS);
    return true;
  case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
  case CU_POINTER_ATTRIBUTE_IS_MANAGED:
  case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
  case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
  case CU_POINTER_ATTRIBUTE_MAPPED:
  case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
  case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
  case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
    *size = sizeof(int);
    return true;
  case CU_POINTER_ATTRIBUTE_BUFFER_ID:
  case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
    *size = sizeof(unsigned long long);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
  case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
  case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
    *size = sizeof(size_t);
    return true;
  case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
    *size = sizeof(CUmemoryPool);
    return true;
  default:
    return false;
  }
}

static std::unordered_map<CUmodule, CUlibrary> &scuda_module_libraries() {
  static std::unordered_map<CUmodule, CUlibrary> libraries;
  return libraries;
}

struct scuda_graph_host_copy {
  void *client_dst = nullptr;
  void *server_src = nullptr;
  size_t bytes = 0;
};

struct scuda_graph_resources;

struct scuda_host_callback_data {
  conn_t *conn = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  std::shared_ptr<scuda_graph_resources> resources;
};

struct scuda_stream_callback_data {
  conn_t *conn = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
};

struct scuda_graph_resources {
  std::vector<CUdeviceptr> device_allocs;
  std::vector<void *> host_allocs;
  std::vector<void *> pageable_allocs;
  std::vector<void *> callback_allocs;
  std::vector<scuda_graph_host_copy> dtoh_copies;
  void *capture_scratch = nullptr;
  size_t capture_scratch_size = 0;
  size_t capture_scratch_offset = 0;

  ~scuda_graph_resources() {
    for (CUdeviceptr ptr : device_allocs) {
      if (ptr != 0) {
        cuMemFree_v2(ptr);
      }
    }
    for (void *ptr : host_allocs) {
      if (ptr != nullptr) {
        cuMemFreeHost(ptr);
      }
    }
    for (void *ptr : pageable_allocs) {
      free(ptr);
    }
    for (void *ptr : callback_allocs) {
      delete static_cast<scuda_host_callback_data *>(ptr);
    }
  }
};

static std::unordered_map<CUgraph, std::shared_ptr<scuda_graph_resources>>
    &scuda_graph_resource_map() {
  static std::unordered_map<CUgraph, std::shared_ptr<scuda_graph_resources>>
      resources;
  return resources;
}

static std::unordered_map<CUgraphExec, std::shared_ptr<scuda_graph_resources>>
    &scuda_graph_exec_resource_map() {
  static std::unordered_map<CUgraphExec, std::shared_ptr<scuda_graph_resources>>
      resources;
  return resources;
}

static std::unordered_map<CUstream, std::shared_ptr<scuda_graph_resources>>
    &scuda_stream_capture_resource_map() {
  static std::unordered_map<CUstream, std::shared_ptr<scuda_graph_resources>>
      resources;
  return resources;
}

static std::unordered_map<CUevent, std::shared_ptr<scuda_graph_resources>>
    &scuda_event_capture_resource_map() {
  static std::unordered_map<CUevent, std::shared_ptr<scuda_graph_resources>>
      resources;
  return resources;
}

static std::shared_ptr<scuda_graph_resources>
scuda_get_graph_resources(CUgraph graph) {
  auto &resources = scuda_graph_resource_map();
  auto it = resources.find(graph);
  if (it != resources.end()) {
    return it->second;
  }
  auto created = std::make_shared<scuda_graph_resources>();
  resources[graph] = created;
  return created;
}

static bool scuda_mem_pool_attribute_size(CUmemPool_attribute attr,
                                          size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
    *size = sizeof(int);
    return true;
  case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
  case CU_MEMPOOL_ATTR_USED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
    *size = sizeof(cuuint64_t);
    return true;
  default:
    return false;
  }
}

static bool scuda_server_trace_enabled() {
  static bool enabled = [] {
    const char *value = getenv("SCUDA_SERVER_TRACE");
    return value != nullptr && strcmp(value, "0") != 0;
  }();
  return enabled;
}

static void scuda_log_manual_handler_error(const char *name) {
  fprintf(stderr, "Error handling manual %s request.\n", name);
}

static uint64_t scuda_fnv1a64(const void *data, size_t size) {
  static constexpr uint64_t kOffset = 14695981039346656037ull;
  static constexpr uint64_t kPrime = 1099511628211ull;
  const auto *bytes = static_cast<const unsigned char *>(data);
  uint64_t hash = kOffset;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

static uint64_t scuda_export_slot_hash(const void *fn) {
  if (fn == nullptr) {
    return 0;
  }
  Dl_info info = {};
  if (dladdr(fn, &info) == 0 || info.dli_fname == nullptr) {
    return 0;
  }
  return scuda_fnv1a64(fn, 32);
}

int handle_manual_cuGetExportTableMetadata(conn_t *conn) {
  CUuuid uuid = {};
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  uint64_t byte_size = 0;
  uint32_t slot_count = 0;
  uint32_t trusted = 0;
  uint64_t hashes[SCUDA_PRIVATE_EXPORT_MAX_SLOTS] = {};

  if (rpc_read(conn, uuid.bytes, sizeof(uuid.bytes)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  const void *export_table = nullptr;
  cuInit(0);
  result = cuGetExportTable(&export_table, &uuid);
  if (result == CUDA_SUCCESS && export_table != nullptr) {
    const auto *slots = static_cast<const void *const *>(export_table);
    byte_size = reinterpret_cast<uintptr_t>(slots[0]);
    if (byte_size >= sizeof(void *) &&
        byte_size % sizeof(void *) == 0 &&
        byte_size / sizeof(void *) <= SCUDA_PRIVATE_EXPORT_MAX_SLOTS) {
      trusted = 1;
      slot_count = static_cast<uint32_t>(byte_size / sizeof(void *));
      for (uint32_t i = 1; i < slot_count; ++i) {
        hashes[i] = scuda_export_slot_hash(slots[i]);
      }
    }
  }

  if (scuda_server_trace_enabled()) {
    std::cerr << "SCUDA server cuGetExportTable metadata result=" << result
              << " bytes=" << byte_size << " slots=" << slot_count
              << " trusted=" << trusted << std::endl;
  }

  size_t hash_bytes = static_cast<size_t>(slot_count) * sizeof(uint64_t);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &byte_size, sizeof(byte_size)) < 0 ||
      rpc_write(conn, &slot_count, sizeof(slot_count)) < 0 ||
      rpc_write(conn, &trusted, sizeof(trusted)) < 0 ||
      (hash_bytes != 0 && rpc_write(conn, hashes, hash_bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static void scuda_private_module_node_callback(void *opaque, void *node,
                                               uint64_t owner) {
  auto *capture =
      static_cast<scuda_private_module_node_capture *>(opaque);
  if (capture == nullptr || capture->node != nullptr) {
    return;
  }
  capture->node = node;
  capture->owner = owner;
  capture->count = 1;
}

int handle_manual_cuPrivateGetModuleNode(conn_t *conn) {
  static constexpr unsigned char PRIVATE_MODULE_ITERATOR_UUID[16] = {
      0x6e, 0x16, 0x3f, 0xbe, 0xb9, 0x58, 0x44, 0x4d,
      0x83, 0x5c, 0xe1, 0x82, 0xaf, 0xf1, 0x99, 0x1e};

  CUcontext context = nullptr;
  CUmodule module = nullptr;
  int request_id;
  CUfunction node = nullptr;
  uint64_t owner = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &context, sizeof(context)) < 0 ||
      rpc_read(conn, &module, sizeof(module)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUuuid uuid = {};
  memcpy(uuid.bytes, PRIVATE_MODULE_ITERATOR_UUID, sizeof(uuid.bytes));
  const void *export_table = nullptr;
  result = cuGetExportTable(&export_table, &uuid);
  if (result == CUDA_SUCCESS && export_table != nullptr) {
    const auto *slots = static_cast<const void *const *>(export_table);
    size_t byte_size = reinterpret_cast<uintptr_t>(slots[0]);
    if (byte_size <= 7 * sizeof(void *) || slots[7] == nullptr) {
      result = CUDA_ERROR_NOT_FOUND;
    } else {
      using private_module_iterator = uint64_t (*)(
          uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
      auto iterator = reinterpret_cast<private_module_iterator>(
          const_cast<void *>(slots[7]));
      scuda_private_module_node_capture capture;
      CUcontext previous = nullptr;
      cuCtxGetCurrent(&previous);
      if (context != nullptr) {
        cuCtxSetCurrent(context);
      }
      uint64_t count = iterator(
          reinterpret_cast<uint64_t>(context),
          reinterpret_cast<uint64_t>(module),
          reinterpret_cast<uint64_t>(&scuda_private_module_node_callback),
          reinterpret_cast<uint64_t>(&capture),
          reinterpret_cast<uint64_t>(module), 0);
      if (previous != context) {
        cuCtxSetCurrent(previous);
      }
      if (capture.node != nullptr) {
        node = reinterpret_cast<CUfunction>(capture.node);
        owner = capture.owner;
        result = CUDA_SUCCESS;
      } else {
        result = count == 0 ? CUDA_ERROR_NOT_FOUND : CUDA_ERROR_UNKNOWN;
      }
      if (scuda_server_trace_enabled()) {
        std::cerr << "SCUDA server private module node module=" << module
                  << " context=" << context << " count=" << count
                  << " node=" << node
                  << " owner=" << reinterpret_cast<void *>(owner)
                  << " result=" << static_cast<int>(result) << std::endl;
      }
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &node, sizeof(node)) < 0 ||
      rpc_write(conn, &owner, sizeof(owner)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static size_t scuda_memcpy3d_host_span_bytes(const CUDA_MEMCPY3D &params,
                                             bool source) {
  size_t width = params.WidthInBytes;
  size_t height = params.Height == 0 ? 1 : params.Height;
  size_t depth = params.Depth == 0 ? 1 : params.Depth;
  size_t pitch = source ? params.srcPitch : params.dstPitch;
  if (pitch == 0) {
    pitch = width;
  }
  return pitch * height * depth;
}

static int scuda_read_graph_dependencies(conn_t *conn,
                                         std::vector<CUgraphNode> *deps) {
  size_t count = 0;
  if (deps == nullptr || rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  deps->resize(count);
  if (count != 0 &&
      rpc_read(conn, deps->data(), count * sizeof(CUgraphNode)) < 0) {
    return -1;
  }
  return 0;
}

static void *scuda_alloc_host_resource(
    const std::shared_ptr<scuda_graph_resources> &resources, size_t bytes) {
  void *ptr = nullptr;
  if (bytes == 0) {
    return nullptr;
  }
  if (cuMemAllocHost(&ptr, bytes) != CUDA_SUCCESS) {
    ptr = malloc(bytes);
    if (ptr != nullptr) {
      resources->pageable_allocs.push_back(ptr);
    }
    return ptr;
  }
  if (ptr != nullptr) {
    resources->host_allocs.push_back(ptr);
  }
  return ptr;
}

static int scuda_write_graph_dtoh_copies(
    conn_t *conn, const std::shared_ptr<scuda_graph_resources> &resources,
    uint32_t *count) {
  if (count == nullptr) {
    return -1;
  }
  *count = resources == nullptr
               ? 0
               : static_cast<uint32_t>(resources->dtoh_copies.size());
  if (rpc_write(conn, count, sizeof(*count)) < 0) {
    return -1;
  }
  if (resources == nullptr) {
    return 0;
  }
  for (const auto &copy : resources->dtoh_copies) {
    if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
        rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
        (copy.bytes != 0 && rpc_write(conn, copy.server_src, copy.bytes) < 0)) {
      return -1;
    }
  }
  return 0;
}

static void *scuda_alloc_capture_scratch(
    const std::shared_ptr<scuda_graph_resources> &resources, size_t bytes) {
  if (!resources || bytes == 0) {
    return nullptr;
  }
  size_t offset = (resources->capture_scratch_offset + 255) & ~size_t(255);
  if (offset + bytes > resources->capture_scratch_size) {
    return nullptr;
  }
  auto *ptr = static_cast<unsigned char *>(resources->capture_scratch) + offset;
  resources->capture_scratch_offset = offset + bytes;
  return ptr;
}

int rpc_write(const void *conn, const void *data, const size_t size) {
  ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] =
      (struct iovec){(void *)data, size};
  return 0;
}

int handle_manual_cuModuleLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUmodule module = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 ||
      rpc_read(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (kind == SCUDA_MODULE_IMAGE_FATBINC_V1 ||
      kind == SCUDA_MODULE_IMAGE_FATBINC_V2) {
    scuda_fatbin_wrapper wrapper = {
        SCUDA_FATBINC_MAGIC,
        kind == SCUDA_MODULE_IMAGE_FATBINC_V2 ? 2U : 1U,
        image.data(),
        nullptr,
    };
    result = cuModuleLoadData(&module, &wrapper);
  } else if (kind == SCUDA_MODULE_IMAGE_FATBIN_RAW) {
    result = cuModuleLoadData(&module, image.data());
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUlibrary library = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 ||
      rpc_read(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (kind == SCUDA_MODULE_IMAGE_FATBINC_V1 ||
      kind == SCUDA_MODULE_IMAGE_FATBINC_V2) {
    scuda_fatbin_wrapper wrapper = {
        SCUDA_FATBINC_MAGIC,
        kind == SCUDA_MODULE_IMAGE_FATBINC_V2 ? 2U : 1U,
        image.data(),
        nullptr,
    };
    result = cuLibraryLoadData(&library, &wrapper, nullptr, nullptr, 0,
                               nullptr, nullptr, 0);
  } else if (kind == SCUDA_MODULE_IMAGE_FATBIN_RAW) {
    result = cuLibraryLoadData(&library, image.data(), nullptr, nullptr, 0,
                               nullptr, nullptr, 0);
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuCtxCreate_v2(conn_t *conn) {
  unsigned int flags = 0;
  CUdevice dev = 0;
  int request_id;
  CUcontext ctx = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &dev, sizeof(dev)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuCtxCreate_v2(&ctx, flags, dev);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolSetAttribute(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemPool_attribute attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!scuda_mem_pool_attribute_size(attr, &expected_size) ||
      value_size != expected_size) {
    return -1;
  }

  std::vector<unsigned char> value(value_size);
  if (rpc_read(conn, value.data(), value_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuMemPoolSetAttribute(pool, attr, value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolGetAttribute(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemPool_attribute attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!scuda_mem_pool_attribute_size(attr, &expected_size) ||
      value_size != expected_size) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> value(value_size);
  result = cuMemPoolGetAttribute(pool, attr, value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, value.data(), value_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerGetAttribute(conn_t *conn) {
  CUpointer_attribute attribute;
  CUdeviceptr ptr = 0;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  unsigned char value[64] = {};

  if (rpc_read(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_read(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!scuda_pointer_attribute_size(attribute, &expected_size) ||
      value_size != expected_size || value_size > sizeof(value)) {
    result = CUDA_ERROR_NOT_SUPPORTED;
    value_size = 0;
  } else {
    result = cuPointerGetAttribute(value, attribute, ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, value, value_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerGetAttributes(conn_t *conn) {
  unsigned int num_attributes = 0;
  CUdeviceptr ptr = 0;
  int request_id;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &num_attributes, sizeof(num_attributes)) < 0) {
    return -1;
  }
  std::vector<CUpointer_attribute> attributes(num_attributes);
  if (num_attributes != 0 &&
      rpc_read(conn, attributes.data(),
               num_attributes * sizeof(CUpointer_attribute)) < 0) {
    return -1;
  }
  if (rpc_read(conn, &ptr, sizeof(ptr)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<size_t> value_sizes(num_attributes, 0);
  std::vector<std::vector<unsigned char>> values(num_attributes);
  std::vector<void *> data(num_attributes, nullptr);
  for (unsigned int i = 0; i < num_attributes; ++i) {
    size_t value_size = 0;
    if (!scuda_pointer_attribute_size(attributes[i], &value_size) ||
        value_size > 64) {
      result = CUDA_ERROR_NOT_SUPPORTED;
      break;
    }
    value_sizes[i] = value_size;
    values[i].resize(value_size);
    data[i] = values[i].data();
  }

  if (result == CUDA_SUCCESS) {
    result =
        cuPointerGetAttributes(num_attributes, attributes.data(), data.data(), ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_attributes; ++i) {
    size_t value_size = result == CUDA_SUCCESS ? value_sizes[i] : 0;
    if (rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
        (value_size != 0 &&
         rpc_write(conn, values[i].data(), value_size) < 0)) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuArrayCreate_v2(conn_t *conn) {
  CUDA_ARRAY_DESCRIPTOR desc = {};
  CUarray handle = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &desc, sizeof(desc)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuArrayCreate_v2(&handle, &desc);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuArray3DCreate_v2(conn_t *conn) {
  CUDA_ARRAY3D_DESCRIPTOR desc = {};
  CUarray handle = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &desc, sizeof(desc)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuArray3DCreate_v2(&handle, &desc);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkCreate_v2(conn_t *conn) {
  unsigned int numOptions = 0;
  CUlinkState state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &numOptions, sizeof(numOptions)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  if (numOptions == 0) {
    result = cuLinkCreate_v2(0, nullptr, nullptr, &state);
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkAddData_v2(conn_t *conn) {
  CUlinkState state = nullptr;
  CUjitInputType type = CU_JIT_INPUT_PTX;
  size_t size = 0;
  size_t name_len = 0;
  unsigned int numOptions = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> data(size);
  if ((size != 0 && rpc_read(conn, data.data(), size) < 0) ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0) {
    return -1;
  }
  std::vector<char> name(name_len == 0 ? 1 : name_len, '\0');
  if ((name_len != 0 && rpc_read(conn, name.data(), name_len) < 0) ||
      rpc_read(conn, &numOptions, sizeof(numOptions)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  if (numOptions == 0) {
    result = cuLinkAddData_v2(state, type, data.data(), data.size(),
                              name_len == 0 ? nullptr : name.data(), 0,
                              nullptr, nullptr);
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkComplete(conn_t *conn) {
  CUlinkState state = nullptr;
  void *cubin = nullptr;
  size_t size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkComplete(state, &cubin, &size);
  size_t returned_size = result == CUDA_SUCCESS ? size : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_size, sizeof(returned_size)) < 0 ||
      (returned_size != 0 && rpc_write(conn, cubin, returned_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpy3D_v2(conn_t *conn) {
  CUDA_MEMCPY3D copy = {};
  size_t src_host_size = 0;
  size_t dst_host_size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &src_host_size, sizeof(src_host_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> src_host(src_host_size);
  if (src_host_size != 0 && rpc_read(conn, src_host.data(), src_host_size) < 0) {
    return -1;
  }
  if (rpc_read(conn, &dst_host_size, sizeof(dst_host_size)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> dst_host(dst_host_size);
  if (copy.srcMemoryType == CU_MEMORYTYPE_HOST) {
    copy.srcHost = src_host.empty() ? nullptr : src_host.data();
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = dst_host.empty() ? nullptr : dst_host.data();
  }

  result = cuMemcpy3D_v2(&copy);
  size_t returned_dst_size = result == CUDA_SUCCESS ? dst_host.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0 ||
      (returned_dst_size != 0 &&
       rpc_write(conn, dst_host.data(), returned_dst_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int handle_manual_cuMemcpy2D_common(conn_t *conn, bool async,
                                           bool unaligned) {
  CUDA_MEMCPY2D copy = {};
  size_t src_host_size = 0;
  size_t dst_host_size = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &src_host_size, sizeof(src_host_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> src_host(src_host_size);
  if (src_host_size != 0 && rpc_read(conn, src_host.data(), src_host_size) < 0) {
    return -1;
  }
  if (rpc_read(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> dst_host(dst_host_size);
  if (copy.srcMemoryType == CU_MEMORYTYPE_HOST) {
    copy.srcHost = src_host.empty() ? nullptr : src_host.data();
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = dst_host.empty() ? nullptr : dst_host.data();
  }

  if (async) {
    result = cuMemcpy2DAsync_v2(&copy, stream);
    if (result == CUDA_SUCCESS) {
      result = cuStreamSynchronize(stream);
    }
  } else if (unaligned) {
    result = cuMemcpy2DUnaligned_v2(&copy);
  } else {
    result = cuMemcpy2D_v2(&copy);
  }

  size_t returned_dst_size = result == CUDA_SUCCESS ? dst_host.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0 ||
      (returned_dst_size != 0 &&
       rpc_write(conn, dst_host.data(), returned_dst_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpy2D_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, false, false);
}

int handle_manual_cuMemcpy2DUnaligned_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, false, true);
}

int handle_manual_cuMemcpy2DAsync_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, true, false);
}

int handle_manual_cuGraphAddMemAllocNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEM_ALLOC_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemAllocNode(&graphNode, hGraph,
                                  deps.empty() ? nullptr : deps.data(),
                                  deps.size(), &nodeParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemFreeNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUdeviceptr dptr = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &dptr, sizeof(dptr)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemFreeNode(&graphNode, hGraph,
                                 deps.empty() ? nullptr : deps.data(),
                                 deps.size(), dptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDeviceGetGraphMemAttribute(conn_t *conn) {
  CUdevice device = 0;
  CUgraphMem_attribute attr = CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
  cuuint64_t value = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuDeviceGetGraphMemAttribute(device, attr, &value);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDeviceSetGraphMemAttribute(conn_t *conn) {
  CUdevice device = 0;
  CUgraphMem_attribute attr = CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
  cuuint64_t value = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value, sizeof(value)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuDeviceSetGraphMemAttribute(device, attr, &value);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuTexObjectCreate(conn_t *conn) {
  CUDA_RESOURCE_DESC resDesc = {};
  CUDA_TEXTURE_DESC texDesc = {};
  CUDA_RESOURCE_VIEW_DESC resViewDesc = {};
  uint32_t hasTexDesc = 0;
  uint32_t hasResViewDesc = 0;
  CUtexObject texObject = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &resDesc, sizeof(resDesc)) < 0 ||
      rpc_read(conn, &hasTexDesc, sizeof(hasTexDesc)) < 0 ||
      (hasTexDesc != 0 && rpc_read(conn, &texDesc, sizeof(texDesc)) < 0) ||
      rpc_read(conn, &hasResViewDesc, sizeof(hasResViewDesc)) < 0 ||
      (hasResViewDesc != 0 &&
       rpc_read(conn, &resViewDesc, sizeof(resViewDesc)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuTexObjectCreate(
      &texObject, &resDesc, hasTexDesc != 0 ? &texDesc : nullptr,
      hasResViewDesc != 0 ? &resViewDesc : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &texObject, sizeof(texObject)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryGetModule(conn_t *conn) {
  CUlibrary library = nullptr;
  CUmodule module = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuLibraryGetModule(&module, library);
  if (result == CUDA_SUCCESS) {
    scuda_module_libraries()[module] = library;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryUnload(conn_t *conn) {
  CUlibrary library = nullptr;
  int request_id;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  (void)library;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuModuleGetGlobal_v2(conn_t *conn) {
  CUdeviceptr dptr = 0;
  size_t bytes = 0;
  CUmodule module = nullptr;
  std::size_t name_len = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &module, sizeof(module)) < 0 ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0) {
    return -1;
  }
  std::vector<char> name(name_len);
  if (name_len == 0 || rpc_read(conn, name.data(), name_len) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuModuleGetGlobal_v2(&dptr, &bytes, module, name.data());
  if (result != CUDA_SUCCESS) {
    auto library_it = scuda_module_libraries().find(module);
    if (library_it != scuda_module_libraries().end()) {
      CUdeviceptr library_dptr = 0;
      size_t library_bytes = 0;
      CUresult library_result =
          cuLibraryGetGlobal(&library_dptr, &library_bytes,
                             library_it->second, name.data());
      if (library_result == CUDA_SUCCESS) {
        dptr = library_dptr;
        bytes = library_bytes;
        result = library_result;
      }
    }
  }
  if (scuda_server_trace_enabled()) {
    std::cerr << "SCUDA cuModuleGetGlobal name=" << name.data()
              << " result=" << static_cast<int>(result)
              << " bytes=" << bytes << std::endl;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(dptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

CUresult scuda_get_kernel_param_layout(CUfunction f,
                                       scuda_kernel_param_layout *layout) {
  if (layout == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *layout = {};
  bool use_kernel_info = false;
  for (uint32_t i = 0; i < 64; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result =
        use_kernel_info
            ? cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f), i, &offset,
                                   &size)
            : cuFuncGetParamInfo(f, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      return CUDA_SUCCESS;
    }
    if (i == 0 && result == CUDA_ERROR_INVALID_HANDLE) {
      use_kernel_info = true;
      result = cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f), i, &offset,
                                    &size);
      if (result == CUDA_ERROR_INVALID_VALUE) {
        return CUDA_SUCCESS;
      }
    }
    if (result != CUDA_SUCCESS) {
      return i == 0 ? result : CUDA_SUCCESS;
    }
    layout->offsets[i] = offset;
    layout->sizes[i] = size;
    layout->count = i + 1;
  }
  return CUDA_SUCCESS;
}

int handle_manual_cuFuncGetParamLayout(conn_t *conn) {
  CUfunction f = nullptr;
  int request_id;
  scuda_kernel_param_layout layout;
  CUresult result;

  if (rpc_read(conn, &f, sizeof(f)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = scuda_get_kernel_param_layout(f, &layout);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, layout.offsets, sizeof(layout.offsets)) < 0 ||
      rpc_write(conn, layout.sizes, sizeof(layout.sizes)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchKernel(conn_t *conn) {
  CUfunction f = nullptr;
  unsigned int gridDimX = 0;
  unsigned int gridDimY = 0;
  unsigned int gridDimZ = 0;
  unsigned int blockDimX = 0;
  unsigned int blockDimY = 0;
  unsigned int blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  uint32_t param_count = 0;
  size_t packed_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  scuda_kernel_param_layout layout;
  result = scuda_get_kernel_param_layout(f, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      result = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                              blockDimY, blockDimZ, sharedMemBytes, hStream,
                              param_count == 0 ? nullptr : params.data(),
                              nullptr);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchCooperativeKernel(conn_t *conn) {
  CUfunction f = nullptr;
  unsigned int gridDimX = 0;
  unsigned int gridDimY = 0;
  unsigned int gridDimZ = 0;
  unsigned int blockDimX = 0;
  unsigned int blockDimY = 0;
  unsigned int blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  uint32_t param_count = 0;
  size_t packed_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  scuda_kernel_param_layout layout;
  result = scuda_get_kernel_param_layout(f, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      result = cuLaunchCooperativeKernel(
          f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
          sharedMemBytes, hStream, param_count == 0 ? nullptr : params.data());
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static void CUDA_CB scuda_graph_host_callback(void *userData) {
  auto *callback = static_cast<scuda_host_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr) {
    return;
  }

  int transfer_count = 0;
  if (callback->resources) {
    transfer_count =
        static_cast<int>(callback->resources->dtoh_copies.size());
  }

  conn_t *conn = callback->conn;
  if (rpc_write_start_request(conn, 1) < 0 ||
      rpc_write(conn, &transfer_count, sizeof(transfer_count)) < 0) {
    return;
  }
  if (callback->resources) {
    for (const auto &copy : callback->resources->dtoh_copies) {
      if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
          rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
          rpc_write(conn, copy.server_src, copy.bytes) < 0) {
        return;
      }
    }
  }
  void *fn = reinterpret_cast<void *>(callback->fn);
  void *response = nullptr;
  if (rpc_write(conn, &fn, sizeof(fn)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &response, sizeof(response)) < 0 ||
      rpc_read_end(conn) < 0) {
    return;
  }
}

static void CUDA_CB scuda_stream_callback(CUstream stream, CUresult status,
                                         void *userData) {
  auto *callback = static_cast<scuda_stream_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr ||
      callback->callback == nullptr) {
    delete callback;
    return;
  }

  conn_t *conn = callback->conn;
  void *fn = reinterpret_cast<void *>(callback->callback);
  void *client_user_data = callback->userData;
  void *response = nullptr;
  if (rpc_write_start_request(conn, 2) >= 0 &&
      rpc_write(conn, &stream, sizeof(stream)) >= 0 &&
      rpc_write(conn, &status, sizeof(status)) >= 0 &&
      rpc_write(conn, &fn, sizeof(fn)) >= 0 &&
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) >= 0 &&
      rpc_wait_for_response(conn) >= 0) {
    rpc_read(conn, &response, sizeof(response));
    rpc_read_end(conn);
  }
  delete callback;
}

int handle_manual_cuGraphAddKernelNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t packed_size = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUfunction func = nodeParams.func != nullptr
                        ? nodeParams.func
                        : reinterpret_cast<CUfunction>(nodeParams.kern);
  scuda_kernel_param_layout layout;
  result = scuda_get_kernel_param_layout(func, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      nodeParams.kernelParams = params.data();
      nodeParams.extra = nullptr;
      result = cuGraphAddKernelNode_v2(
          &graphNode, hGraph, deps.empty() ? nullptr : deps.data(), deps.size(),
          &nodeParams);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemcpyNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEMCPY3D copyParams = {};
  CUcontext ctx = nullptr;
  size_t host_src_bytes = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &copyParams, sizeof(copyParams)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &host_src_bytes, sizeof(host_src_bytes)) < 0) {
    return -1;
  }

  auto resources = scuda_get_graph_resources(hGraph);
  if (host_src_bytes != 0) {
    void *host = scuda_alloc_host_resource(resources, host_src_bytes);
    if (host == nullptr || rpc_read(conn, host, host_src_bytes) < 0) {
      return -1;
    }
    copyParams.srcHost = host;
  }

  if (copyParams.dstMemoryType == CU_MEMORYTYPE_HOST) {
    size_t host_dst_bytes = scuda_memcpy3d_host_span_bytes(copyParams, false);
    void *host = scuda_alloc_host_resource(resources, host_dst_bytes);
    if (host == nullptr && host_dst_bytes != 0) {
      return -1;
    }
    resources->dtoh_copies.push_back(
        {copyParams.dstHost, host, host_dst_bytes});
    copyParams.dstHost = host;
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemcpyNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size(), &copyParams, ctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemsetNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEMSET_NODE_PARAMS memsetParams = {};
  CUcontext ctx = nullptr;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &memsetParams, sizeof(memsetParams)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemsetNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size(), &memsetParams, ctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddHostNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_HOST_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback = new scuda_host_callback_data{
      conn, nodeParams.fn, nodeParams.userData, scuda_get_graph_resources(hGraph)};
  CUDA_HOST_NODE_PARAMS serverParams = {};
  serverParams.fn = scuda_graph_host_callback;
  serverParams.userData = callback;
  callback->resources->callback_allocs.push_back(callback);

  result = cuGraphAddHostNode(&graphNode, hGraph,
                              deps.empty() ? nullptr : deps.data(),
                              deps.size(), &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphConditionalHandleCreate(conn_t *conn) {
  CUgraph hGraph = nullptr;
  CUcontext ctx = nullptr;
  unsigned int defaultLaunchValue = 0;
  unsigned int flags = 0;
  CUgraphConditionalHandle handle = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &defaultLaunchValue, sizeof(defaultLaunchValue)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphConditionalHandleCreate(&handle, hGraph, ctx,
                                          defaultLaunchValue, flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUgraphNodeParams nodeParams = {};
  uint32_t param_count = 0;
  size_t packed_size = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraph> child_graphs;
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    if (param_count == 0 && packed_size == 0) {
      nodeParams.kernel.kernelParams = nullptr;
      nodeParams.kernel.extra = nullptr;
      result = cuGraphAddNode(&graphNode, hGraph,
                              deps.empty() ? nullptr : deps.data(), nullptr,
                              deps.size(), &nodeParams);
    } else {
      CUfunction func =
          nodeParams.kernel.func != nullptr
              ? nodeParams.kernel.func
              : reinterpret_cast<CUfunction>(nodeParams.kernel.kern);
      scuda_kernel_param_layout layout;
      result = scuda_get_kernel_param_layout(func, &layout);
      if (result == CUDA_SUCCESS && layout.count == param_count) {
      std::vector<void *> params(param_count);
      for (uint32_t i = 0; i < param_count; ++i) {
        if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
          result = CUDA_ERROR_INVALID_VALUE;
          break;
        }
        params[i] = packed.data() + layout.offsets[i];
      }
      if (result == CUDA_SUCCESS) {
        nodeParams.kernel.kernelParams = params.data();
        nodeParams.kernel.extra = nullptr;
        result = cuGraphAddNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(), nullptr,
                                deps.size(), &nodeParams);
      }
    }
    }
  } else if (nodeParams.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    child_graphs.resize(nodeParams.conditional.size);
    nodeParams.conditional.phGraph_out = nullptr;
    result = cuGraphAddNode(&graphNode, hGraph,
                            deps.empty() ? nullptr : deps.data(), nullptr,
                            deps.size(), &nodeParams);
    if (result == CUDA_SUCCESS && nodeParams.conditional.phGraph_out != nullptr) {
      for (size_t i = 0; i < child_graphs.size(); ++i) {
        child_graphs[i] = nodeParams.conditional.phGraph_out[i];
      }
    }
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (scuda_server_trace_enabled()) {
    std::cerr << "SCUDA cuGraphAddNode type=" << nodeParams.type
              << " param_count=" << param_count
              << " packed_size=" << packed_size
              << " graph=" << hGraph << " node=" << graphNode
              << " result=" << result << std::endl;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      (!child_graphs.empty() &&
       rpc_write(conn, child_graphs.data(),
                 child_graphs.size() * sizeof(CUgraph)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphGetNodes(conn_t *conn) {
  CUgraph hGraph = nullptr;
  size_t requested = 0;
  size_t count = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &requested, sizeof(requested)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (requested == 0) {
    result = cuGraphGetNodes(hGraph, nullptr, &count);
  } else {
    count = requested;
  }
  std::vector<CUgraphNode> nodes(count);
  if (result == CUDA_SUCCESS && requested != 0) {
    result = cuGraphGetNodes(hGraph, nodes.data(), &count);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (requested != 0 && count != 0 &&
       rpc_write(conn, nodes.data(), count * sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchHostFunc(conn_t *conn) {
  CUstream stream = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &fn, sizeof(fn)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto &stream_resources = scuda_stream_capture_resource_map();
  auto &resources = stream_resources[stream];
  if (!resources) {
    resources = std::make_shared<scuda_graph_resources>();
  }
  auto *callback = new scuda_host_callback_data{conn, fn, userData, resources};
  resources->callback_allocs.push_back(callback);
  result = cuLaunchHostFunc(stream, scuda_graph_host_callback, callback);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamAddCallback(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *data = new scuda_stream_callback_data{conn, callback, userData};
  result = cuStreamAddCallback(stream, scuda_stream_callback, data, flags);
  if (result != CUDA_SUCCESS) {
    delete data;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventRecord(conn_t *conn, bool with_flags) {
  CUevent event = nullptr;
  CUstream stream = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto stream_it = scuda_stream_capture_resource_map().find(stream);
  if (stream_it != scuda_stream_capture_resource_map().end()) {
    scuda_event_capture_resource_map()[event] = stream_it->second;
  }

  result = with_flags ? cuEventRecordWithFlags(event, stream, flags)
                      : cuEventRecord(event, stream);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamWaitEvent(conn_t *conn) {
  CUstream stream = nullptr;
  CUevent event = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto event_it = scuda_event_capture_resource_map().find(event);
  if (event_it != scuda_event_capture_resource_map().end()) {
    auto &stream_resources = scuda_stream_capture_resource_map();
    auto &resources = stream_resources[stream];
    if (!resources) {
      resources = event_it->second;
    }
  }

  result = cuStreamWaitEvent(stream, event, flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamBeginCaptureToGraph(conn_t *conn) {
  CUstream stream = nullptr;
  CUgraph graph = nullptr;
  std::vector<CUgraphNode> deps;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamBeginCaptureToGraph(
      stream, graph, deps.empty() ? nullptr : deps.data(), nullptr,
      deps.size(), mode);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamUpdateCaptureDependencies(conn_t *conn) {
  CUstream stream = nullptr;
  std::vector<CUgraphNode> deps;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      scuda_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamUpdateCaptureDependencies_v2(
      stream, deps.empty() ? nullptr : deps.data(), nullptr, deps.size(),
      flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamGetCaptureInfo(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  const CUgraphNode *deps_ptr = nullptr;
  const CUgraphEdgeData *edge_ptr = nullptr;
  size_t dep_count = 0;
  bool has_edge_data = false;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamGetCaptureInfo_v3(stream, &status, &id, &graph, &deps_ptr,
                                     &edge_ptr, &dep_count);
  has_edge_data = edge_ptr != nullptr && dep_count != 0;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &id, sizeof(id)) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &dep_count, sizeof(dep_count)) < 0 ||
      rpc_write(conn, &has_edge_data, sizeof(has_edge_data)) < 0 ||
      (dep_count != 0 && deps_ptr != nullptr &&
       rpc_write(conn, deps_ptr, dep_count * sizeof(CUgraphNode)) < 0) ||
      (has_edge_data &&
       rpc_write(conn, edge_ptr, dep_count * sizeof(CUgraphEdgeData)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamBeginCapture(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto &stream_resources = scuda_stream_capture_resource_map();
  auto &resources = stream_resources[stream];
  if (!resources) {
    resources = std::make_shared<scuda_graph_resources>();
  }
  if (resources->capture_scratch == nullptr) {
    static constexpr size_t scratch_size = 128ull * 1024ull * 1024ull;
    void *scratch = nullptr;
    if (cuMemAllocHost(&scratch, scratch_size) == CUDA_SUCCESS) {
      resources->capture_scratch = scratch;
      resources->capture_scratch_size = scratch_size;
      resources->host_allocs.push_back(scratch);
    }
  }

  result = resources->capture_scratch == nullptr
               ? CUDA_ERROR_OUT_OF_MEMORY
               : cuStreamBeginCapture_v2(stream, mode);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamEndCapture(conn_t *conn) {
  CUstream stream = nullptr;
  CUgraph graph = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamEndCapture(stream, &graph);
  if (result == CUDA_SUCCESS) {
    auto &stream_resources = scuda_stream_capture_resource_map();
    auto stream_it = stream_resources.find(stream);
    if (stream_it != stream_resources.end()) {
      scuda_graph_resource_map()[graph] = stream_it->second;
      stream_resources.erase(stream_it);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphClone(conn_t *conn) {
  CUgraph clone = nullptr;
  CUgraph original = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &clone, sizeof(clone)) < 0 ||
      rpc_read(conn, &original, sizeof(original)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphClone(&clone, original);
  if (result == CUDA_SUCCESS) {
    auto original_it = scuda_graph_resource_map().find(original);
    if (original_it != scuda_graph_resource_map().end()) {
      scuda_graph_resource_map()[clone] = original_it->second;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &clone, sizeof(clone)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphInstantiateWithFlags(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  unsigned long long flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphInstantiateWithFlags(&exec, graph, flags);
  if (result == CUDA_SUCCESS) {
    auto graph_it = scuda_graph_resource_map().find(graph);
    if (graph_it != scuda_graph_resource_map().end()) {
      scuda_graph_exec_resource_map()[exec] = graph_it->second;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphInstantiateWithParams(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  CUDA_GRAPH_INSTANTIATE_PARAMS params = {};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphInstantiateWithParams(&exec, graph, &params);
  if (result == CUDA_SUCCESS) {
    auto graph_it = scuda_graph_resource_map().find(graph);
    if (graph_it != scuda_graph_resource_map().end()) {
      scuda_graph_exec_resource_map()[exec] = graph_it->second;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &params, sizeof(params)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecDestroy(conn_t *conn) {
  CUgraphExec exec = nullptr;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphExecDestroy(exec);
  scuda_graph_exec_resource_map().erase(exec);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphDestroy(conn_t *conn) {
  CUgraph graph = nullptr;
  if (rpc_read(conn, &graph, sizeof(graph)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphDestroy(graph);
  scuda_graph_resource_map().erase(graph);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyHtoDAsync_v2(conn_t *conn) {
  CUdeviceptr dstDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  std::vector<unsigned char> host_data(byteCount);
  if ((byteCount != 0 && rpc_read(conn, host_data.data(), byteCount) < 0) ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto &stream_resources = scuda_stream_capture_resource_map();
    auto &resources = stream_resources[stream];
    if (!resources) {
      resources = std::make_shared<scuda_graph_resources>();
    }
    void *host = scuda_alloc_capture_scratch(resources, byteCount);
    if (host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      memcpy(host, host_data.data(), byteCount);
      result = cuMemcpyHtoDAsync_v2(dstDevice, host, byteCount, stream);
    }
  } else {
    void *host = nullptr;
    CUresult alloc_result = cuMemAllocHost(&host, byteCount);
    std::vector<unsigned char> fallback;
    if (alloc_result != CUDA_SUCCESS) {
      fallback.resize(byteCount);
      host = fallback.data();
    }
    if (byteCount != 0 && host == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      memcpy(host, host_data.data(), byteCount);
    result = cuMemcpyHtoDAsync_v2(dstDevice, host, byteCount, stream);
    if (result == CUDA_SUCCESS) {
      result = cuStreamSynchronize(stream);
    }
    }
    if (alloc_result == CUDA_SUCCESS && host != nullptr) {
      cuMemFreeHost(host);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyDtoHAsync_v2(conn_t *conn) {
  void *dstHost = nullptr;
  CUdeviceptr srcDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }

  void *host = nullptr;
  CUresult alloc_result = CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned char> fallback;
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
      auto &stream_resources = scuda_stream_capture_resource_map();
      auto &resources = stream_resources[stream];
      if (!resources) {
        resources = std::make_shared<scuda_graph_resources>();
      }
    host = scuda_alloc_capture_scratch(resources, byteCount);
    if (host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS) {
        resources->dtoh_copies.push_back({dstHost, host, byteCount});
      }
    }
  } else {
    alloc_result = cuMemAllocHost(&host, byteCount);
    if (alloc_result != CUDA_SUCCESS) {
      fallback.resize(byteCount);
      host = fallback.data();
    }
    if (byteCount != 0 && host == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS) {
        result = cuStreamSynchronize(stream);
      }
    }
  }

  std::vector<unsigned char> empty_response;
  void *response_host = host;
  if (response_host == nullptr && byteCount != 0) {
    empty_response.resize(byteCount);
    response_host = empty_response.data();
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, response_host, byteCount) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    if (alloc_result == CUDA_SUCCESS && host != nullptr) {
      cuMemFreeHost(host);
    }
    return -1;
  }

  if (alloc_result == CUDA_SUCCESS && host != nullptr) {
    cuMemFreeHost(host);
  }
  return 0;
}

int handle_manual_cuCtxSynchronize(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuCtxSynchronize();
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamSynchronize(conn_t *conn) {
  CUstream stream = nullptr;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuStreamSynchronize(stream);
  std::shared_ptr<scuda_graph_resources> resources;
  uint32_t copy_count = 0;
  auto stream_it = scuda_stream_capture_resource_map().find(stream);
  if (stream_it != scuda_stream_capture_resource_map().end()) {
    resources = stream_it->second;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      scuda_write_graph_dtoh_copies(conn, resources, &copy_count) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphLaunch(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUstream stream = nullptr;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphLaunch(exec, stream);
  auto exec_it = scuda_graph_exec_resource_map().find(exec);
  if (result == CUDA_SUCCESS && exec_it != scuda_graph_exec_resource_map().end()) {
    scuda_stream_capture_resource_map()[stream] = exec_it->second;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventSynchronize(conn_t *conn) {
  CUevent event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuEventSynchronize(event);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuOccupancyMaxPotentialBlockSize(conn_t *conn,
                                                   bool with_flags) {
  CUfunction func = nullptr;
  size_t dynamicSMemSize = 0;
  int blockSizeLimit = 0;
  unsigned int flags = 0;
  int request_id;
  int minGridSize = 0;
  int blockSize = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &func, sizeof(func)) < 0 ||
      rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
      rpc_read(conn, &blockSizeLimit, sizeof(blockSizeLimit)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (with_flags) {
    result = cuOccupancyMaxPotentialBlockSizeWithFlags(
        &minGridSize, &blockSize, func, nullptr, dynamicSMemSize,
        blockSizeLimit, flags);
  } else {
    result = cuOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, func, nullptr, dynamicSMemSize,
        blockSizeLimit);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &minGridSize, sizeof(minGridSize)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(blockSize)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

void client_handler(int connfd) {
  conn_t conn = {connfd, 1};
  conn.request_id = 1;
  conn.local_request_parity = conn.request_id & 1;
  if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.write_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.call_mutex, NULL) < 0 ||
      pthread_cond_init(&conn.read_cond, NULL) < 0) {
    std::cerr << "Error initializing mutex." << std::endl;
    return;
  }

  printf("Client connected.\n");

  while (1) {
    int op = rpc_dispatch(&conn, 0);
    if (op < 0) {
      std::cerr << "RPC dispatch failed; closing client." << std::endl;
      break;
    }
    if (scuda_server_trace_enabled()) {
      std::cerr << "SCUDA server handling op " << op << std::endl;
    }

    if (op == SCUDA_RPC_cuGetExportTableMetadata) {
      if (handle_manual_cuGetExportTableMetadata(&conn) < 0) {
        std::cerr << "Error handling manual cuGetExportTable metadata request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuPrivateGetModuleNode) {
      if (handle_manual_cuPrivateGetModuleNode(&conn) < 0) {
        std::cerr << "Error handling manual private module node request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuModuleLoadData) {
      if (handle_manual_cuModuleLoadData(&conn) < 0) {
        std::cerr << "Error handling manual cuModuleLoadData request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuLibraryLoadData) {
      if (handle_manual_cuLibraryLoadData(&conn) < 0) {
        std::cerr << "Error handling manual cuLibraryLoadData request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuCtxCreate_v2) {
      if (handle_manual_cuCtxCreate_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuCtxCreate_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuMemPoolSetAttribute) {
      if (handle_manual_cuMemPoolSetAttribute(&conn) < 0) {
        std::cerr << "Error handling manual cuMemPoolSetAttribute request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuMemPoolGetAttribute) {
      if (handle_manual_cuMemPoolGetAttribute(&conn) < 0) {
        std::cerr << "Error handling manual cuMemPoolGetAttribute request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuPointerGetAttribute) {
      if (handle_manual_cuPointerGetAttribute(&conn) < 0) {
        std::cerr << "Error handling manual cuPointerGetAttribute request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuPointerGetAttributes) {
      if (handle_manual_cuPointerGetAttributes(&conn) < 0) {
        std::cerr << "Error handling manual cuPointerGetAttributes request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuArrayCreate_v2) {
      if (handle_manual_cuArrayCreate_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuArrayCreate_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuArray3DCreate_v2) {
      if (handle_manual_cuArray3DCreate_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuArray3DCreate_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuLinkCreate_v2) {
      if (handle_manual_cuLinkCreate_v2(&conn) < 0) {
        scuda_log_manual_handler_error("cuLinkCreate_v2");
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuLinkAddData_v2) {
      if (handle_manual_cuLinkAddData_v2(&conn) < 0) {
        scuda_log_manual_handler_error("cuLinkAddData_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuLinkComplete) {
      if (handle_manual_cuLinkComplete(&conn) < 0) {
        scuda_log_manual_handler_error("cuLinkComplete");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy3D_v2) {
      if (handle_manual_cuMemcpy3D_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuMemcpy3D_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy2D_v2) {
      if (handle_manual_cuMemcpy2D_v2(&conn) < 0) {
        scuda_log_manual_handler_error("cuMemcpy2D_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy2DUnaligned_v2) {
      if (handle_manual_cuMemcpy2DUnaligned_v2(&conn) < 0) {
        scuda_log_manual_handler_error("cuMemcpy2DUnaligned_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy2DAsync_v2) {
      if (handle_manual_cuMemcpy2DAsync_v2(&conn) < 0) {
        scuda_log_manual_handler_error("cuMemcpy2DAsync_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddMemAllocNode) {
      if (handle_manual_cuGraphAddMemAllocNode(&conn) < 0) {
        scuda_log_manual_handler_error("cuGraphAddMemAllocNode");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddMemFreeNode) {
      if (handle_manual_cuGraphAddMemFreeNode(&conn) < 0) {
        scuda_log_manual_handler_error("cuGraphAddMemFreeNode");
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuDeviceGetGraphMemAttribute) {
      if (handle_manual_cuDeviceGetGraphMemAttribute(&conn) < 0) {
        scuda_log_manual_handler_error("cuDeviceGetGraphMemAttribute");
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuDeviceSetGraphMemAttribute) {
      if (handle_manual_cuDeviceSetGraphMemAttribute(&conn) < 0) {
        scuda_log_manual_handler_error("cuDeviceSetGraphMemAttribute");
        break;
      }
      continue;
    }
    if (op == RPC_cuTexObjectCreate) {
      if (handle_manual_cuTexObjectCreate(&conn) < 0) {
        scuda_log_manual_handler_error("cuTexObjectCreate");
        break;
      }
      continue;
    }
    if (op == RPC_cuLibraryGetModule) {
      if (handle_manual_cuLibraryGetModule(&conn) < 0) {
        std::cerr << "Error handling manual cuLibraryGetModule request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuLibraryUnload) {
      if (handle_manual_cuLibraryUnload(&conn) < 0) {
        std::cerr << "Error handling manual cuLibraryUnload request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuModuleGetGlobal_v2) {
      if (handle_manual_cuModuleGetGlobal_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuModuleGetGlobal_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuFuncGetParamLayout) {
      if (handle_manual_cuFuncGetParamLayout(&conn) < 0) {
        std::cerr << "Error handling manual cuFuncGetParamLayout request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuOccupancyMaxPotentialBlockSize) {
      if (handle_manual_cuOccupancyMaxPotentialBlockSize(&conn, false) < 0) {
        std::cerr
            << "Error handling manual cuOccupancyMaxPotentialBlockSize request."
            << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuOccupancyMaxPotentialBlockSizeWithFlags) {
      if (handle_manual_cuOccupancyMaxPotentialBlockSize(&conn, true) < 0) {
        std::cerr << "Error handling manual "
                     "cuOccupancyMaxPotentialBlockSizeWithFlags request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuLaunchKernel) {
      if (handle_manual_cuLaunchKernel(&conn) < 0) {
        std::cerr << "Error handling manual cuLaunchKernel request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuLaunchCooperativeKernel) {
      if (handle_manual_cuLaunchCooperativeKernel(&conn) < 0) {
        std::cerr << "Error handling manual cuLaunchCooperativeKernel request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddKernelNode_v2) {
      if (handle_manual_cuGraphAddKernelNode(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphAddKernelNode request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddMemcpyNode) {
      if (handle_manual_cuGraphAddMemcpyNode(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphAddMemcpyNode request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddMemsetNode) {
      if (handle_manual_cuGraphAddMemsetNode(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphAddMemsetNode request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddHostNode) {
      if (handle_manual_cuGraphAddHostNode(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphAddHostNode request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuGraphConditionalHandleCreate) {
      if (handle_manual_cuGraphConditionalHandleCreate(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphConditionalHandleCreate "
                     "request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuGraphAddNode_v2) {
      if (handle_manual_cuGraphAddNode(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphAddNode request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphLaunch) {
      if (handle_manual_cuGraphLaunch(&conn) < 0) {
        scuda_log_manual_handler_error("cuGraphLaunch");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphGetNodes) {
      if (handle_manual_cuGraphGetNodes(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphGetNodes request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuLaunchHostFunc) {
      if (handle_manual_cuLaunchHostFunc(&conn) < 0) {
        std::cerr << "Error handling manual cuLaunchHostFunc request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuStreamAddCallback) {
      if (handle_manual_cuStreamAddCallback(&conn) < 0) {
        std::cerr << "Error handling manual cuStreamAddCallback request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuEventRecord) {
      if (handle_manual_cuEventRecord(&conn, false) < 0) {
        scuda_log_manual_handler_error("cuEventRecord");
        break;
      }
      continue;
    }
    if (op == RPC_cuEventRecordWithFlags) {
      if (handle_manual_cuEventRecord(&conn, true) < 0) {
        scuda_log_manual_handler_error("cuEventRecordWithFlags");
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamWaitEvent) {
      if (handle_manual_cuStreamWaitEvent(&conn) < 0) {
        scuda_log_manual_handler_error("cuStreamWaitEvent");
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuStreamBeginCaptureToGraph) {
      if (handle_manual_cuStreamBeginCaptureToGraph(&conn) < 0) {
        std::cerr << "Error handling manual cuStreamBeginCaptureToGraph "
                     "request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuStreamUpdateCaptureDependencies_v2) {
      if (handle_manual_cuStreamUpdateCaptureDependencies(&conn) < 0) {
        std::cerr << "Error handling manual "
                     "cuStreamUpdateCaptureDependencies request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == SCUDA_RPC_cuStreamGetCaptureInfo_v3) {
      if (handle_manual_cuStreamGetCaptureInfo(&conn) < 0) {
        std::cerr << "Error handling manual cuStreamGetCaptureInfo request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamBeginCapture_v2) {
      if (handle_manual_cuStreamBeginCapture(&conn) < 0) {
        std::cerr << "Error handling manual cuStreamBeginCapture request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamEndCapture) {
      if (handle_manual_cuStreamEndCapture(&conn) < 0) {
        std::cerr << "Error handling manual cuStreamEndCapture request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphClone) {
      if (handle_manual_cuGraphClone(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphClone request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphInstantiateWithFlags) {
      if (handle_manual_cuGraphInstantiateWithFlags(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphInstantiateWithFlags request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphInstantiateWithParams) {
      if (handle_manual_cuGraphInstantiateWithParams(&conn) < 0) {
        std::cerr
            << "Error handling manual cuGraphInstantiateWithParams request."
            << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphExecDestroy) {
      if (handle_manual_cuGraphExecDestroy(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphExecDestroy request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphDestroy) {
      if (handle_manual_cuGraphDestroy(&conn) < 0) {
        std::cerr << "Error handling manual cuGraphDestroy request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpyHtoDAsync_v2) {
      if (handle_manual_cuMemcpyHtoDAsync_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuMemcpyHtoDAsync_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpyDtoHAsync_v2) {
      if (handle_manual_cuMemcpyDtoHAsync_v2(&conn) < 0) {
        std::cerr << "Error handling manual cuMemcpyDtoHAsync_v2 request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuCtxSynchronize) {
      if (handle_manual_cuCtxSynchronize(&conn) < 0) {
        std::cerr << "Error handling manual cuCtxSynchronize request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamSynchronize) {
      if (handle_manual_cuStreamSynchronize(&conn) < 0) {
        std::cerr << "Error handling manual cuStreamSynchronize request."
                  << std::endl;
        break;
      }
      continue;
    }
    if (op == RPC_cuEventSynchronize) {
      if (handle_manual_cuEventSynchronize(&conn) < 0) {
        std::cerr << "Error handling manual cuEventSynchronize request."
                  << std::endl;
        break;
      }
      continue;
    }

    auto opHandler = get_handler(op);
    if (opHandler == nullptr) {
      std::cerr << "No RPC handler for op " << op << "; closing client."
                << std::endl;
      break;
    }
    if (opHandler(&conn) < 0) {
      std::cerr << "Error handling request." << std::endl;
      break;
    }
  }

  if (pthread_mutex_destroy(&conn.read_mutex) < 0 ||
      pthread_mutex_destroy(&conn.write_mutex) < 0)
    std::cerr << "Error destroying mutex." << std::endl;

  close(connfd);
}

int main() {
  int port = DEFAULT_PORT;
  struct sockaddr_in servaddr, cli;
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    printf("Socket creation failed.\n");
    exit(EXIT_FAILURE);
  }

  char *p = getenv("SCUDA_PORT");

  if (p == NULL) {
    port = DEFAULT_PORT;
  } else {
    port = atoi(p);
  }

  // Bind the socket
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(port);

  const int enable = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
    printf("Socket bind failed.\n");
    exit(EXIT_FAILURE);
  }

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
    printf("Socket bind failed.\n");
    exit(EXIT_FAILURE);
  }

  if (listen(sockfd, MAX_CLIENTS) != 0) {
    printf("Listen failed.\n");
    exit(EXIT_FAILURE);
  }

  printf("Server listening on port %d...\n", port);

  // Server loop
  while (1) {
    socklen_t len = sizeof(cli);
    int connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

    if (connfd < 0) {
      std::cerr << "Server accept failed." << std::endl;
      continue;
    }

    std::thread client_thread(client_handler, connfd);

    // detach the thread so it runs independently
    client_thread.detach();
  }

  close(sockfd);
  return 0;
}
