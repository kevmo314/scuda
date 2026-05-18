#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cudaProfiler.h>
#include <dlfcn.h>
#include <elf.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <netdb.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define SCUDA_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef SCUDA_CUDA_COMPAT_TYPES_ONLY

#include <cstdlib>
#include <cstring>

#include "codegen/gen_api.h"
#include "codegen/gen_client.h"
#include "rpc.h"

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;
static bool scuda_rpc_shutting_down = false;

const char *DEFAULT_PORT = "14833";

std::map<void *, void *> host_funcs;

void add_host_node(void *fn, void *udata);

struct scuda_fatbin_wrapper {
  uint32_t magic;
  uint32_t version;
  const void *data;
  const void *filename_or_fatbins;
};

struct scuda_fatbin_header {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t files_size;
};

static constexpr uint32_t SCUDA_FATBINC_MAGIC = 0x466243b1;
static constexpr uint32_t SCUDA_FATBIN_MAGIC = 0xba55ed50;
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
static constexpr uint32_t SCUDA_PRIVATE_EXPORT_MAX_SLOTS = 256;

struct scuda_kernel_param_layout {
  uint32_t count = 0;
  size_t offsets[64] = {};
  size_t sizes[64] = {};
};

struct scuda_private_node_mapping {
  CUfunction server_function = nullptr;
  uint64_t server_owner = 0;
};

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
#if CUDA_VERSION >= 13000
  case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
#endif
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

extern int rpc_size();
extern int rpc_open();
extern conn_t *rpc_client_get_connection(unsigned int index);
extern void rpc_close(conn_t *conn);

static bool scuda_trace_enabled() {
  static bool enabled = [] {
    const char *value = getenv("SCUDA_TRACE");
    return value != nullptr && strcmp(value, "0") != 0;
  }();
  return enabled;
}

static bool scuda_stub_missing_enabled() {
  static bool enabled = [] {
    const char *value = getenv("SCUDA_STUB_MISSING");
    return value == nullptr || strcmp(value, "0") != 0;
  }();
  return enabled;
}

static bool scuda_symbol_looks_like_driver_api(const char *symbol) {
  return symbol != nullptr && symbol[0] == 'c' && symbol[1] == 'u' &&
         symbol[2] >= 'A' && symbol[2] <= 'Z';
}

static void *scuda_real_dlsym(void *handle, const char *name) {
  static void *(*real_dlsym)(void *, const char *) = nullptr;
  if (real_dlsym == nullptr) {
    real_dlsym = reinterpret_cast<void *(*)(void *, const char *)>(
        dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
  }
  return real_dlsym != nullptr ? real_dlsym(handle, name) : nullptr;
}

extern "C" CUresult scuda_unsupported_driver_api() {
  std::cerr << "SCUDA unsupported generic Driver API called" << std::endl;
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult scuda_missing_driver_api_called(const char *symbol) {
  std::cerr << "SCUDA missing Driver API called: " << symbol << std::endl;
  return CUDA_ERROR_NOT_SUPPORTED;
}

static std::unordered_map<std::string, std::vector<uint64_t>> &
scuda_private_export_hashes() {
  static std::unordered_map<std::string, std::vector<uint64_t>> hashes;
  return hashes;
}

static std::unordered_map<CUfunction, scuda_private_node_mapping> &
scuda_private_node_map() {
  static std::unordered_map<CUfunction, scuda_private_node_mapping> mappings;
  return mappings;
}

static std::unordered_map<CUfunction, CUfunction> &scuda_host_function_map() {
  static std::unordered_map<CUfunction, CUfunction> mappings;
  return mappings;
}

static std::vector<CUmodule> &scuda_loaded_modules() {
  static std::vector<CUmodule> modules;
  return modules;
}

struct scuda_remote_device {
  unsigned int conn_index = 0;
  int remote_ordinal = 0;
  CUdevice remote_device = 0;
};

static std::mutex &scuda_routing_mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::vector<scuda_remote_device> &scuda_device_table() {
  static std::vector<scuda_remote_device> devices;
  return devices;
}

static bool &scuda_device_table_ready() {
  static bool ready = false;
  return ready;
}

static std::unordered_map<CUcontext, unsigned int> &scuda_context_owners() {
  static std::unordered_map<CUcontext, unsigned int> owners;
  return owners;
}

static std::unordered_map<CUmodule, unsigned int> &scuda_module_owners() {
  static std::unordered_map<CUmodule, unsigned int> owners;
  return owners;
}

static std::unordered_map<CUfunction, unsigned int> &scuda_function_owners() {
  static std::unordered_map<CUfunction, unsigned int> owners;
  return owners;
}

static std::unordered_map<CUstream, unsigned int> &scuda_stream_owners() {
  static std::unordered_map<CUstream, unsigned int> owners;
  return owners;
}

static std::unordered_map<CUevent, unsigned int> &scuda_event_owners() {
  static std::unordered_map<CUevent, unsigned int> owners;
  return owners;
}

static std::unordered_map<CUdeviceptr, unsigned int> &scuda_deviceptr_owners() {
  static std::unordered_map<CUdeviceptr, unsigned int> owners;
  return owners;
}

static std::mutex &scuda_host_function_mutex() {
  static std::mutex mutex;
  return mutex;
}

static unsigned char (&scuda_private_6e16_node_pool())[16][0x500] {
  static unsigned char nodes[16][0x500] = {};
  return nodes;
}

static std::atomic<unsigned int> &scuda_private_6e16_next_node() {
  static std::atomic<unsigned int> next{0};
  return next;
}

static std::mutex &scuda_private_node_mutex() {
  static std::mutex mutex;
  return mutex;
}

static void scuda_remember_loaded_module(CUmodule module) {
  if (module == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_host_function_mutex());
  auto &modules = scuda_loaded_modules();
  if (std::find(modules.begin(), modules.end(), module) == modules.end()) {
    modules.push_back(module);
  }
}

extern "C" void scuda_remember_loaded_module_for_rpc(CUmodule module) {
  scuda_remember_loaded_module(module);
}

static int scuda_conn_index(conn_t *conn) {
  if (conn == nullptr) {
    return -1;
  }
  for (int i = 0; i < nconns; ++i) {
    if (&conns[i] == conn) {
      return i;
    }
  }
  return -1;
}

static conn_t *scuda_conn_by_index(unsigned int index) {
  if (rpc_open() < 0 || index >= static_cast<unsigned int>(nconns)) {
    return nullptr;
  }
  return &conns[index];
}

static CUresult scuda_remote_cuInit(conn_t *conn, unsigned int flags) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuInit) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static CUresult scuda_remote_cuDeviceGetCount(conn_t *conn, int *count) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || count == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetCount) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, count, sizeof(*count)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static CUresult scuda_remote_cuDeviceGet(conn_t *conn, CUdevice *device,
                                         int ordinal) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || device == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGet) < 0 ||
      rpc_write(conn, &ordinal, sizeof(ordinal)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, device, sizeof(*device)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static CUresult scuda_ensure_device_table() {
  if (rpc_open() < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  if (scuda_device_table_ready()) {
    return CUDA_SUCCESS;
  }

  auto &devices = scuda_device_table();
  devices.clear();
  for (int i = 0; i < nconns; ++i) {
    int remote_count = 0;
    CUresult result = scuda_remote_cuDeviceGetCount(&conns[i], &remote_count);
    if (result != CUDA_SUCCESS) {
      devices.clear();
      return result;
    }
    for (int ordinal = 0; ordinal < remote_count; ++ordinal) {
      CUdevice remote_device = 0;
      result = scuda_remote_cuDeviceGet(&conns[i], &remote_device, ordinal);
      if (result != CUDA_SUCCESS) {
        devices.clear();
        return result;
      }
      devices.push_back(scuda_remote_device{static_cast<unsigned int>(i),
                                            ordinal, remote_device});
    }
  }
  scuda_device_table_ready() = true;
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_cuInit_multi(unsigned int flags) {
  if (rpc_open() < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  CUresult first_error = CUDA_SUCCESS;
  for (int i = 0; i < nconns; ++i) {
    CUresult result = scuda_remote_cuInit(&conns[i], flags);
    if (result != CUDA_SUCCESS && first_error == CUDA_SUCCESS) {
      first_error = result;
    }
  }
  return first_error;
}

extern "C" CUresult scuda_cuDeviceGetCount_multi(int *count) {
  if (count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = scuda_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  *count = static_cast<int>(scuda_device_table().size());
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_cuDeviceGet_multi(CUdevice *device, int ordinal) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = scuda_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  if (ordinal < 0 || ordinal >= static_cast<int>(scuda_device_table().size())) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = static_cast<CUdevice>(ordinal);
  return CUDA_SUCCESS;
}

extern "C" conn_t *scuda_rpc_conn_for_device(CUdevice *device) {
  if (device == nullptr || scuda_ensure_device_table() != CUDA_SUCCESS) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  int local = static_cast<int>(*device);
  auto &devices = scuda_device_table();
  if (local < 0 || local >= static_cast<int>(devices.size())) {
    return nullptr;
  }
  const scuda_remote_device &mapped = devices[local];
  *device = mapped.remote_device;
  return scuda_conn_by_index(mapped.conn_index);
}

extern "C" bool scuda_translate_device_for_conn(conn_t *conn,
                                                CUdevice *device) {
  if (conn == nullptr || device == nullptr ||
      scuda_ensure_device_table() != CUDA_SUCCESS) {
    return false;
  }

  int conn_index = scuda_conn_index(conn);
  if (conn_index < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  int local = static_cast<int>(*device);
  auto &devices = scuda_device_table();
  if (local < 0 || local >= static_cast<int>(devices.size())) {
    return false;
  }
  const scuda_remote_device &mapped = devices[local];
  if (mapped.conn_index != static_cast<unsigned int>(conn_index)) {
    return false;
  }
  *device = mapped.remote_device;
  return true;
}

extern "C" CUdevice scuda_local_device_for_remote(conn_t *conn,
                                                  CUdevice remote_device) {
  if (conn == nullptr || scuda_ensure_device_table() != CUDA_SUCCESS) {
    return -1;
  }
  int conn_index = scuda_conn_index(conn);
  if (conn_index < 0) {
    return -1;
  }

  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto &devices = scuda_device_table();
  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i].conn_index == static_cast<unsigned int>(conn_index) &&
        devices[i].remote_device == remote_device) {
      return static_cast<CUdevice>(i);
    }
  }
  return -1;
}

extern "C" CUresult scuda_cuDeviceCanAccessPeer_multi(int *canAccessPeer,
                                                      CUdevice dev,
                                                      CUdevice peerDev) {
  if (canAccessPeer == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = scuda_rpc_conn_for_device(&dev);
  if (conn == nullptr) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (!scuda_translate_device_for_conn(conn, &peerDev)) {
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
  }

  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (rpc_write_start_request(conn, RPC_cuDeviceCanAccessPeer) < 0 ||
      rpc_write(conn, canAccessPeer, sizeof(*canAccessPeer)) < 0 ||
      rpc_write(conn, &dev, sizeof(dev)) < 0 ||
      rpc_write(conn, &peerDev, sizeof(peerDev)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, canAccessPeer, sizeof(*canAccessPeer)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" void scuda_note_context_owner(CUcontext ctx, conn_t *conn) {
  int index = scuda_conn_index(conn);
  if (ctx == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_context_owners()[ctx] = static_cast<unsigned int>(index);
}

extern "C" void scuda_note_module_owner(CUmodule module, conn_t *conn) {
  int index = scuda_conn_index(conn);
  if (module == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_module_owners()[module] = static_cast<unsigned int>(index);
}

extern "C" void scuda_note_function_owner(CUfunction function, conn_t *conn) {
  int index = scuda_conn_index(conn);
  if (function == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_function_owners()[function] = static_cast<unsigned int>(index);
}

extern "C" void scuda_note_stream_owner(CUstream stream, conn_t *conn) {
  int index = scuda_conn_index(conn);
  if (stream == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_stream_owners()[stream] = static_cast<unsigned int>(index);
}

extern "C" void scuda_note_event_owner(CUevent event, conn_t *conn) {
  int index = scuda_conn_index(conn);
  if (event == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_event_owners()[event] = static_cast<unsigned int>(index);
}

extern "C" void scuda_note_deviceptr_owner(CUdeviceptr ptr, conn_t *conn) {
  int index = scuda_conn_index(conn);
  if (ptr == 0 || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_deviceptr_owners()[ptr] = static_cast<unsigned int>(index);
}

extern "C" void scuda_forget_deviceptr_owner(CUdeviceptr ptr) {
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  scuda_deviceptr_owners().erase(ptr);
}

static conn_t *scuda_conn_for_owner(unsigned int owner) {
  return scuda_conn_by_index(owner);
}

extern "C" conn_t *scuda_rpc_conn_for_context(CUcontext ctx) {
  if (ctx == nullptr) {
    return scuda_conn_by_index(0);
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto it = scuda_context_owners().find(ctx);
  if (it == scuda_context_owners().end()) {
    return scuda_conn_by_index(0);
  }
  return scuda_conn_for_owner(it->second);
}

extern "C" conn_t *scuda_rpc_conn_for_module(CUmodule module) {
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto it = scuda_module_owners().find(module);
  return it == scuda_module_owners().end() ? scuda_conn_by_index(0)
                                           : scuda_conn_for_owner(it->second);
}

extern "C" conn_t *scuda_rpc_conn_for_function(CUfunction function) {
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto it = scuda_function_owners().find(function);
  return it == scuda_function_owners().end() ? scuda_conn_by_index(0)
                                             : scuda_conn_for_owner(it->second);
}

extern "C" conn_t *scuda_rpc_conn_for_stream(CUstream stream) {
  if (stream == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto it = scuda_stream_owners().find(stream);
  return it == scuda_stream_owners().end() ? scuda_conn_by_index(0)
                                           : scuda_conn_for_owner(it->second);
}

extern "C" conn_t *scuda_rpc_conn_for_event(CUevent event) {
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto it = scuda_event_owners().find(event);
  return it == scuda_event_owners().end() ? scuda_conn_by_index(0)
                                          : scuda_conn_for_owner(it->second);
}

extern "C" conn_t *scuda_rpc_conn_for_deviceptr(CUdeviceptr ptr) {
  std::lock_guard<std::mutex> lock(scuda_routing_mutex());
  auto it = scuda_deviceptr_owners().find(ptr);
  return it == scuda_deviceptr_owners().end()
             ? scuda_conn_by_index(0)
             : scuda_conn_for_owner(it->second);
}

static bool scuda_read_file_span(const char *path,
                                 std::vector<unsigned char> *bytes) {
  if (path == nullptr || bytes == nullptr) {
    return false;
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(size));
  return file.read(reinterpret_cast<char *>(bytes->data()), size).good();
}

static bool scuda_lookup_elf_function_symbol(const char *path, uintptr_t offset,
                                             std::string *symbol) {
  std::vector<unsigned char> bytes;
  if (symbol == nullptr || !scuda_read_file_span(path, &bytes) ||
      bytes.size() < sizeof(Elf64_Ehdr)) {
    return false;
  }

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(bytes.data());
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr->e_shentsize != sizeof(Elf64_Shdr) || ehdr->e_shoff == 0 ||
      ehdr->e_shnum == 0) {
    return false;
  }
  size_t shoff = static_cast<size_t>(ehdr->e_shoff);
  size_t shnum = static_cast<size_t>(ehdr->e_shnum);
  if (shoff > bytes.size() ||
      shnum > (bytes.size() - shoff) / sizeof(Elf64_Shdr)) {
    return false;
  }
  const auto *sections =
      reinterpret_cast<const Elf64_Shdr *>(bytes.data() + shoff);

  uintptr_t best_value = 0;
  const char *best_name = nullptr;
  for (size_t i = 0; i < shnum; ++i) {
    if (sections[i].sh_type != SHT_SYMTAB &&
        sections[i].sh_type != SHT_DYNSYM) {
      continue;
    }
    if (sections[i].sh_link >= shnum) {
      continue;
    }
    const Elf64_Shdr &symtab = sections[i];
    const Elf64_Shdr &strtab = sections[symtab.sh_link];
    if (symtab.sh_entsize != sizeof(Elf64_Sym) ||
        symtab.sh_offset > bytes.size() || strtab.sh_offset > bytes.size() ||
        symtab.sh_size > bytes.size() - symtab.sh_offset ||
        strtab.sh_size > bytes.size() - strtab.sh_offset) {
      continue;
    }
    const auto *syms =
        reinterpret_cast<const Elf64_Sym *>(bytes.data() + symtab.sh_offset);
    const char *strings =
        reinterpret_cast<const char *>(bytes.data() + strtab.sh_offset);
    size_t count = static_cast<size_t>(symtab.sh_size / sizeof(Elf64_Sym));
    for (size_t j = 0; j < count; ++j) {
      const Elf64_Sym &sym = syms[j];
      if (ELF64_ST_TYPE(sym.st_info) != STT_FUNC ||
          sym.st_name >= strtab.sh_size || sym.st_value == 0 ||
          offset < sym.st_value ||
          (sym.st_size != 0 && offset >= sym.st_value + sym.st_size)) {
        continue;
      }
      if (best_name == nullptr || sym.st_value >= best_value) {
        best_value = sym.st_value;
        best_name = strings + sym.st_name;
      }
    }
  }

  if (best_name == nullptr || best_name[0] == '\0') {
    return false;
  }
  *symbol = best_name;
  return true;
}

static CUfunction scuda_resolve_host_function(CUfunction function) {
  if (function == nullptr) {
    return function;
  }

  {
    std::lock_guard<std::mutex> lock(scuda_host_function_mutex());
    auto mapped = scuda_host_function_map().find(function);
    if (mapped != scuda_host_function_map().end()) {
      return mapped->second;
    }
  }

  Dl_info info = {};
  if (dladdr(reinterpret_cast<void *>(function), &info) == 0) {
    return function;
  }
  std::string symbol_name;
  const char *kernel_name = info.dli_sname;
  if (kernel_name == nullptr && info.dli_fname != nullptr &&
      info.dli_fbase != nullptr) {
    uintptr_t offset = reinterpret_cast<uintptr_t>(function) -
                       reinterpret_cast<uintptr_t>(info.dli_fbase);
    if (scuda_lookup_elf_function_symbol(info.dli_fname, offset,
                                         &symbol_name)) {
      kernel_name = symbol_name.c_str();
    }
  }
  if (kernel_name == nullptr) {
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA could not resolve host kernel symbol for "
                << reinterpret_cast<void *>(function) << std::endl;
    }
    return function;
  }

  std::vector<CUmodule> modules;
  {
    std::lock_guard<std::mutex> lock(scuda_host_function_mutex());
    modules = scuda_loaded_modules();
  }
  for (CUmodule module : modules) {
    CUfunction remote = nullptr;
    CUresult result = cuModuleGetFunction(&remote, module, kernel_name);
    if (result == CUDA_SUCCESS && remote != nullptr) {
      std::lock_guard<std::mutex> lock(scuda_host_function_mutex());
      scuda_host_function_map()[function] = remote;
      if (scuda_trace_enabled()) {
        std::cerr << "SCUDA mapped host kernel " << kernel_name
                  << " host=" << reinterpret_cast<void *>(function)
                  << " remote=" << remote << std::endl;
      }
      return remote;
    }
  }
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA host kernel " << kernel_name << " was not found in "
              << modules.size() << " loaded modules" << std::endl;
  }

  return function;
}

static CUfunction scuda_translate_private_function(CUfunction function) {
  {
    std::lock_guard<std::mutex> lock(scuda_private_node_mutex());
    auto it = scuda_private_node_map().find(function);
    if (it != scuda_private_node_map().end()) {
      return it->second.server_function;
    }
  }
  return scuda_resolve_host_function(function);
}

extern "C" CUfunction
scuda_translate_private_function_for_rpc(CUfunction function) {
  return scuda_translate_private_function(function);
}

static bool scuda_is_private_function(CUfunction function) {
  std::lock_guard<std::mutex> lock(scuda_private_node_mutex());
  return scuda_private_node_map().find(function) !=
         scuda_private_node_map().end();
}

static CUresult scuda_get_remote_private_module_node(CUcontext context,
                                                     CUmodule module,
                                                     CUfunction *server_node,
                                                     uint64_t *server_owner) {
  if (server_node == nullptr || server_owner == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *server_node = nullptr;
  *server_owner = 0;
  conn_t *conn = rpc_client_get_connection(0);
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (conn == nullptr ||
      rpc_write_start_request(conn, SCUDA_RPC_cuPrivateGetModuleNode) < 0 ||
      rpc_write(conn, &context, sizeof(context)) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, server_node, sizeof(*server_node)) < 0 ||
      rpc_read(conn, server_owner, sizeof(*server_owner)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static uint64_t scuda_private_export_slot_hash(const char *table_name,
                                               int slot) {
  if (table_name == nullptr || slot < 0) {
    return 0;
  }
  auto &hashes = scuda_private_export_hashes();
  auto it = hashes.find(table_name);
  if (it == hashes.end() || static_cast<size_t>(slot) >= it->second.size()) {
    return 0;
  }
  return it->second[slot];
}

extern "C" CUresult
scuda_private_export_slot_called(int slot, const char *table_name,
                                 uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                 uint64_t arg3, uint64_t arg4, uint64_t arg5) {
  static uint64_t scuda_private_2131_fake_object[64] = {};

  fprintf(stderr,
          "SCUDA private export table called: %s[%d] hash=%#llx "
          "args=%#llx,%#llx,%#llx,%#llx,%#llx,%#llx\n",
          table_name, slot,
          static_cast<unsigned long long>(
              scuda_private_export_slot_hash(table_name, slot)),
          static_cast<unsigned long long>(arg0),
          static_cast<unsigned long long>(arg1),
          static_cast<unsigned long long>(arg2),
          static_cast<unsigned long long>(arg3),
          static_cast<unsigned long long>(arg4),
          static_cast<unsigned long long>(arg5));

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 4) {
    if (arg1 == 0) {
      return CUDA_ERROR_UNKNOWN;
    }
    *reinterpret_cast<uint64_t *>(arg1) = arg0 == 0 ? 0 : 1;
    return CUDA_SUCCESS;
  }

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 51) {
    if (arg2 == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (arg1 == UINT64_MAX) {
      return static_cast<CUresult>(400);
    }
    *reinterpret_cast<uint64_t *>(arg2) =
        reinterpret_cast<uint64_t>(&scuda_private_2131_fake_object[0]);
    return CUDA_SUCCESS;
  }

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 39) {
    if (arg1 == 0 || arg2 == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    *reinterpret_cast<uint64_t *>(arg2) = 0;
    return CUDA_SUCCESS;
  }

  if (table_name != nullptr &&
      strcmp(table_name, "6e163fbeb958444d835ce182aff1991e") == 0 &&
      slot == 7) {
    CUfunction server_node = nullptr;
    uint64_t server_owner = 0;
    CUresult node_result = scuda_get_remote_private_module_node(
        reinterpret_cast<CUcontext>(arg0), reinterpret_cast<CUmodule>(arg1),
        &server_node, &server_owner);
    if (node_result != CUDA_SUCCESS || server_node == nullptr) {
      if (scuda_trace_enabled()) {
        std::cerr << "SCUDA private 6e16[7] remote node failed result="
                  << static_cast<int>(node_result)
                  << " module=" << reinterpret_cast<void *>(arg1) << std::endl;
      }
      return node_result == CUDA_SUCCESS ? CUDA_ERROR_NOT_FOUND : node_result;
    }

    auto &node_pool = scuda_private_6e16_node_pool();
    unsigned int node_index =
        scuda_private_6e16_next_node().fetch_add(1, std::memory_order_relaxed) %
        (sizeof(node_pool) / sizeof(node_pool[0]));
    unsigned char *client_node = node_pool[node_index];
    memset(client_node, 0, sizeof(node_pool[node_index]));
    *reinterpret_cast<uint64_t *>(client_node + 0x0) = 0x100000001ULL;
    *reinterpret_cast<uint64_t *>(client_node + 0x8) = server_owner;
    *reinterpret_cast<uint64_t *>(client_node + 0x10) = 0xc;
    *reinterpret_cast<uint64_t *>(client_node + 0x18) = 1;
    *reinterpret_cast<uint64_t *>(client_node + 0x28) = 0x80;
    *reinterpret_cast<uint32_t *>(client_node + 0x370) = 2;
    *reinterpret_cast<uint64_t *>(client_node + 0x480) = 0;
    {
      std::lock_guard<std::mutex> lock(scuda_private_node_mutex());
      scuda_private_node_map()[reinterpret_cast<CUfunction>(client_node)] = {
          server_node, server_owner};
    }
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA private 6e16[7] mapped client_node[" << node_index
                << "]=" << static_cast<void *>(client_node)
                << " server_node=" << server_node
                << " server_owner=" << reinterpret_cast<void *>(server_owner)
                << std::endl;
    }
    if (arg2 != 0) {
      using scuda_private_iterator_callback =
          void (*)(void *, void *, uint64_t);
      auto callback = reinterpret_cast<scuda_private_iterator_callback>(arg2);
      callback(reinterpret_cast<void *>(arg3), client_node,
               *reinterpret_cast<uint64_t *>(client_node + 8));
    }
    return static_cast<CUresult>(1);
  }

  return CUDA_ERROR_NOT_SUPPORTED;
}

static bool scuda_stub_private_exports_enabled() {
  static bool enabled = [] {
    const char *value = getenv("SCUDA_STUB_PRIVATE_EXPORTS");
    return value != nullptr && strcmp(value, "0") != 0;
  }();
  return enabled;
}

static bool scuda_remote_private_exports_enabled() {
  static bool enabled = [] {
    const char *value = getenv("SCUDA_REMOTE_PRIVATE_EXPORTS");
    return value == nullptr || strcmp(value, "0") != 0;
  }();
  return enabled;
}

static std::atomic<bool> &scuda_private_export_tables_active_flag() {
  static std::atomic<bool> active{false};
  return active;
}

static bool scuda_private_export_remap_active() {
  return scuda_stub_private_exports_enabled() ||
         scuda_private_export_tables_active_flag().load(
             std::memory_order_relaxed);
}

static void *scuda_make_private_export_stub(int slot, const char *table_name) {
#if defined(__x86_64__)
  constexpr size_t stub_size = 52;
  unsigned char *code = static_cast<unsigned char *>(
      mmap(nullptr, stub_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&scuda_unsupported_driver_api);
  }

  void *handler = reinterpret_cast<void *>(&scuda_private_export_slot_called);
  size_t offset = 0;
  const unsigned char prologue[] = {
      0x48, 0x83, 0xec, 0x08, // sub rsp, 8
      0x41, 0x51,             // push r9
      0x41, 0x50,             // push r8
      0x49, 0x89, 0xc9,       // mov r9, rcx
      0x49, 0x89, 0xd0,       // mov r8, rdx
      0x48, 0x89, 0xf1,       // mov rcx, rsi
      0x48, 0x89, 0xfa,       // mov rdx, rdi
      0xbf                    // mov edi, imm32
  };
  memcpy(code + offset, prologue, sizeof(prologue));
  offset += sizeof(prologue);
  memcpy(code + offset, &slot, sizeof(slot));
  offset += sizeof(slot);
  code[offset++] = 0x48;
  code[offset++] = 0xbe;
  memcpy(code + offset, &table_name, sizeof(table_name));
  offset += sizeof(table_name);
  code[offset++] = 0x48;
  code[offset++] = 0xb8;
  memcpy(code + offset, &handler, sizeof(handler));
  offset += sizeof(handler);
  const unsigned char epilogue[] = {
      0xff, 0xd0,             // call rax
      0x48, 0x83, 0xc4, 0x18, // add rsp, 24
      0xc3                    // ret
  };
  memcpy(code + offset, epilogue, sizeof(epilogue));
  return code;
#else
  (void)slot;
  (void)table_name;
  return reinterpret_cast<void *>(&scuda_unsupported_driver_api);
#endif
}

static const void *
scuda_make_private_export_table(const char *table_name, size_t byte_size,
                                const std::vector<uint64_t> &code_hashes = {}) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<void *>> tables;
  std::lock_guard<std::mutex> lock(mutex);
  auto existing = tables.find(table_name);
  if (existing != tables.end()) {
    return existing->second.data();
  }

  size_t entries = byte_size / sizeof(void *);
  std::vector<void *> table(entries, nullptr);
  if (!table.empty()) {
    table[0] = reinterpret_cast<void *>(byte_size);
  }
  char *stable_table_name = strdup(table_name);
  for (size_t i = 1; i < entries; ++i) {
    table[i] =
        scuda_make_private_export_stub(static_cast<int>(i), stable_table_name);
  }
  if (!code_hashes.empty()) {
    scuda_private_export_hashes()[table_name] = code_hashes;
  }
  auto inserted = tables.emplace(table_name, std::move(table));
  scuda_private_export_tables_active_flag().store(true,
                                                  std::memory_order_relaxed);
  return inserted.first->second.data();
}

static std::string scuda_uuid_hex(const CUuuid *uuid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  if (uuid == nullptr) {
    return out;
  }
  out.reserve(32);
  for (unsigned char byte : uuid->bytes) {
    out.push_back(kHex[byte >> 4]);
    out.push_back(kHex[byte & 0xf]);
  }
  return out;
}

static const void *scuda_remote_private_export_table(const CUuuid *uuid) {
  if (uuid == nullptr || !scuda_remote_private_exports_enabled()) {
    return nullptr;
  }

  conn_t *conn = rpc_client_get_connection(0);
  if (conn == nullptr) {
    return nullptr;
  }

  CUresult result = CUDA_ERROR_UNKNOWN;
  uint64_t byte_size = 0;
  uint32_t slot_count = 0;
  uint32_t trusted = 0;
  uint64_t hashes[SCUDA_PRIVATE_EXPORT_MAX_SLOTS] = {};

  if (rpc_write_start_request(conn, SCUDA_RPC_cuGetExportTableMetadata) < 0 ||
      rpc_write(conn, uuid->bytes, sizeof(uuid->bytes)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 ||
      rpc_read(conn, &byte_size, sizeof(byte_size)) < 0 ||
      rpc_read(conn, &slot_count, sizeof(slot_count)) < 0 ||
      rpc_read(conn, &trusted, sizeof(trusted)) < 0) {
    return nullptr;
  }

  if (slot_count > SCUDA_PRIVATE_EXPORT_MAX_SLOTS) {
    rpc_read_end(conn);
    return nullptr;
  }
  size_t hash_bytes = static_cast<size_t>(slot_count) * sizeof(uint64_t);
  if (hash_bytes != 0 && rpc_read(conn, hashes, hash_bytes) < 0) {
    rpc_read_end(conn);
    return nullptr;
  }
  if (rpc_read_end(conn) < 0) {
    return nullptr;
  }

  if (result != CUDA_SUCCESS || trusted == 0 || byte_size == 0 ||
      byte_size % sizeof(void *) != 0 || slot_count == 0 ||
      slot_count != byte_size / sizeof(void *) ||
      slot_count > SCUDA_PRIVATE_EXPORT_MAX_SLOTS) {
    return nullptr;
  }

  std::string uuid_hex = scuda_uuid_hex(uuid);
  std::vector<uint64_t> code_hashes(hashes, hashes + slot_count);
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA remote cuGetExportTable metadata uuid=" << uuid_hex
              << " bytes=" << byte_size << " slots=" << slot_count << std::endl;
  }
  return scuda_make_private_export_table(
      uuid_hex.c_str(), static_cast<size_t>(byte_size), code_hashes);
}

static const void *scuda_private_export_table_from_env(const CUuuid *uuid) {
  if (uuid == nullptr) {
    return nullptr;
  }
  static std::unordered_map<std::string, size_t> configured_tables = [] {
    std::unordered_map<std::string, size_t> tables;
    const char *raw = getenv("SCUDA_PRIVATE_EXPORT_TABLES");
    if (raw == nullptr || raw[0] == '\0') {
      return tables;
    }

    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
      size_t colon = item.find(':');
      if (colon == std::string::npos) {
        continue;
      }
      std::string uuid_hex = item.substr(0, colon);
      if (uuid_hex.size() != 32) {
        continue;
      }
      char *end = nullptr;
      unsigned long long byte_size =
          strtoull(item.c_str() + colon + 1, &end, 0);
      if (end == item.c_str() + colon + 1 || byte_size == 0 ||
          byte_size % sizeof(void *) != 0) {
        continue;
      }
      tables[uuid_hex] = static_cast<size_t>(byte_size);
    }
    return tables;
  }();

  std::string uuid_hex = scuda_uuid_hex(uuid);
  auto it = configured_tables.find(uuid_hex);
  if (it == configured_tables.end()) {
    return nullptr;
  }
  return scuda_make_private_export_table(uuid_hex.c_str(), it->second);
}

static const char *scuda_error_name(CUresult error) {
  switch (error) {
  case CUDA_SUCCESS:
    return "CUDA_SUCCESS";
  case CUDA_ERROR_INVALID_VALUE:
    return "CUDA_ERROR_INVALID_VALUE";
  case CUDA_ERROR_OUT_OF_MEMORY:
    return "CUDA_ERROR_OUT_OF_MEMORY";
  case CUDA_ERROR_NOT_INITIALIZED:
    return "CUDA_ERROR_NOT_INITIALIZED";
  case CUDA_ERROR_DEINITIALIZED:
    return "CUDA_ERROR_DEINITIALIZED";
  case CUDA_ERROR_NO_DEVICE:
    return "CUDA_ERROR_NO_DEVICE";
  case CUDA_ERROR_INVALID_DEVICE:
    return "CUDA_ERROR_INVALID_DEVICE";
  case CUDA_ERROR_INVALID_IMAGE:
    return "CUDA_ERROR_INVALID_IMAGE";
  case CUDA_ERROR_INVALID_CONTEXT:
    return "CUDA_ERROR_INVALID_CONTEXT";
  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
  case CUDA_ERROR_MAP_FAILED:
    return "CUDA_ERROR_MAP_FAILED";
  case CUDA_ERROR_UNMAP_FAILED:
    return "CUDA_ERROR_UNMAP_FAILED";
  case CUDA_ERROR_ARRAY_IS_MAPPED:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";
  case CUDA_ERROR_ALREADY_MAPPED:
    return "CUDA_ERROR_ALREADY_MAPPED";
  case CUDA_ERROR_NO_BINARY_FOR_GPU:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";
  case CUDA_ERROR_ALREADY_ACQUIRED:
    return "CUDA_ERROR_ALREADY_ACQUIRED";
  case CUDA_ERROR_NOT_MAPPED:
    return "CUDA_ERROR_NOT_MAPPED";
  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
  case CUDA_ERROR_ECC_UNCORRECTABLE:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";
  case CUDA_ERROR_UNSUPPORTED_LIMIT:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";
  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
    return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
  case CUDA_ERROR_INVALID_PTX:
    return "CUDA_ERROR_INVALID_PTX";
  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
    return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
  case CUDA_ERROR_NVLINK_UNCORRECTABLE:
    return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
  case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
    return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
  case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
    return "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
  case CUDA_ERROR_JIT_COMPILATION_DISABLED:
    return "CUDA_ERROR_JIT_COMPILATION_DISABLED";
  case CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY:
    return "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY";
  case CUDA_ERROR_INVALID_SOURCE:
    return "CUDA_ERROR_INVALID_SOURCE";
  case CUDA_ERROR_FILE_NOT_FOUND:
    return "CUDA_ERROR_FILE_NOT_FOUND";
  case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
    return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
  case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
    return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
  case CUDA_ERROR_OPERATING_SYSTEM:
    return "CUDA_ERROR_OPERATING_SYSTEM";
  case CUDA_ERROR_INVALID_HANDLE:
    return "CUDA_ERROR_INVALID_HANDLE";
  case CUDA_ERROR_ILLEGAL_STATE:
    return "CUDA_ERROR_ILLEGAL_STATE";
  case CUDA_ERROR_NOT_FOUND:
    return "CUDA_ERROR_NOT_FOUND";
  case CUDA_ERROR_NOT_READY:
    return "CUDA_ERROR_NOT_READY";
  case CUDA_ERROR_ILLEGAL_ADDRESS:
    return "CUDA_ERROR_ILLEGAL_ADDRESS";
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
  case CUDA_ERROR_LAUNCH_TIMEOUT:
    return "CUDA_ERROR_LAUNCH_TIMEOUT";
  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
    return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
  case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
  case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
    return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
  case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
    return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
    return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
  case CUDA_ERROR_ASSERT:
    return "CUDA_ERROR_ASSERT";
  case CUDA_ERROR_TOO_MANY_PEERS:
    return "CUDA_ERROR_TOO_MANY_PEERS";
  case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
    return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
  case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
    return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
  case CUDA_ERROR_HARDWARE_STACK_ERROR:
    return "CUDA_ERROR_HARDWARE_STACK_ERROR";
  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
    return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
  case CUDA_ERROR_MISALIGNED_ADDRESS:
    return "CUDA_ERROR_MISALIGNED_ADDRESS";
  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
    return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
  case CUDA_ERROR_INVALID_PC:
    return "CUDA_ERROR_INVALID_PC";
  case CUDA_ERROR_LAUNCH_FAILED:
    return "CUDA_ERROR_LAUNCH_FAILED";
  case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
    return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
  case CUDA_ERROR_NOT_PERMITTED:
    return "CUDA_ERROR_NOT_PERMITTED";
  case CUDA_ERROR_NOT_SUPPORTED:
    return "CUDA_ERROR_NOT_SUPPORTED";
  case CUDA_ERROR_SYSTEM_NOT_READY:
    return "CUDA_ERROR_SYSTEM_NOT_READY";
  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
    return "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
  case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
    return "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";
  case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
    return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
  case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
    return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
  case CUDA_ERROR_STREAM_CAPTURE_MERGE:
    return "CUDA_ERROR_STREAM_CAPTURE_MERGE";
  case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
    return "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
  case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
    return "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
  case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
    return "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
  case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
    return "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
  case CUDA_ERROR_CAPTURED_EVENT:
    return "CUDA_ERROR_CAPTURED_EVENT";
  case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
    return "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
  case CUDA_ERROR_TIMEOUT:
    return "CUDA_ERROR_TIMEOUT";
  case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
    return "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
  case CUDA_ERROR_EXTERNAL_DEVICE:
    return "CUDA_ERROR_EXTERNAL_DEVICE";
  case CUDA_ERROR_UNKNOWN:
    return "CUDA_ERROR_UNKNOWN";
  default:
    return nullptr;
  }
}

extern "C" CUresult cuGetErrorName(CUresult error, const char **pStr) {
  if (pStr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const char *name = scuda_error_name(error);
  if (name == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pStr = name;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGetErrorString(CUresult error, const char **pStr) {
  return cuGetErrorName(error, pStr);
}

extern "C" CUresult cuProfilerInitialize(const char *, const char *,
                                         CUoutput_mode) {
  return CUDA_SUCCESS;
}

extern "C" CUresult cuProfilerStart(void) { return CUDA_SUCCESS; }

extern "C" CUresult cuProfilerStop(void) { return CUDA_SUCCESS; }

CUresult cuStreamDestroy_v2(CUstream hStream);
CUresult cuEventDestroy_v2(CUevent hEvent);
CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                               CUevent hEnd);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream);

#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
extern "C" CUresult cuStreamDestroy(CUstream hStream) {
  return cuStreamDestroy_v2(hStream);
}

#ifdef cuEventDestroy
#undef cuEventDestroy
#endif
extern "C" CUresult cuEventDestroy(CUevent hEvent) {
  return cuEventDestroy_v2(hEvent);
}

#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                       CUevent hEnd) {
  return cuEventElapsedTime_v2(pMilliseconds, hStart, hEnd);
}

extern "C" CUresult cuMemPoolSetAttribute(CUmemoryPool pool,
                                          CUmemPool_attribute attr,
                                          void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!scuda_mem_pool_attribute_size(attr, &value_size)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuMemPoolSetAttribute) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_write(conn, value, value_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static thread_local CUcontext scuda_current_context = nullptr;
static thread_local std::vector<CUcontext> scuda_context_stack;

extern "C" conn_t *scuda_rpc_conn_for_current_context() {
  return scuda_rpc_conn_for_context(scuda_current_context);
}

static CUresult scuda_set_remote_current_context(CUcontext ctx) {
  conn_t *conn =
      scuda_rpc_conn_for_context(ctx != nullptr ? ctx : scuda_current_context);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxSetCurrent) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" void scuda_note_ctx_create(CUcontext ctx, conn_t *conn) {
  scuda_note_context_owner(ctx, conn);
  scuda_context_stack.push_back(scuda_current_context);
  scuda_current_context = ctx;
}

extern "C" CUresult scuda_cuCtxPushCurrent_virtual(CUcontext ctx) {
  scuda_context_stack.push_back(scuda_current_context);
  CUresult result = scuda_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    scuda_current_context = ctx;
  } else {
    scuda_context_stack.pop_back();
  }
  return result;
}

extern "C" CUresult scuda_cuCtxPopCurrent_virtual(CUcontext *pctx) {
  CUcontext popped = scuda_current_context;
  if (pctx != nullptr) {
    *pctx = popped;
  }
  CUcontext previous = nullptr;
  if (!scuda_context_stack.empty()) {
    previous = scuda_context_stack.back();
    scuda_context_stack.pop_back();
  }
  CUresult result = scuda_set_remote_current_context(previous);
  if (result == CUDA_SUCCESS) {
    scuda_current_context = previous;
  } else {
    scuda_context_stack.push_back(previous);
  }
  return result;
}

extern "C" CUresult scuda_cuCtxSetCurrent_virtual(CUcontext ctx) {
  CUresult result = scuda_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    scuda_current_context = ctx;
  }
  return result;
}

extern "C" CUresult scuda_cuCtxGetCurrent_virtual(CUcontext *pctx) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pctx = scuda_current_context;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemPoolGetAttribute(CUmemoryPool pool,
                                          CUmemPool_attribute attr,
                                          void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!scuda_mem_pool_attribute_size(attr, &value_size)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuMemPoolGetAttribute) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, value_size) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

struct scuda_host_allocation {
  size_t size = 0;
  size_t storage_size = 0;
  size_t page_size = 0;
  size_t page_count = 0;
  unsigned int flags = 0;
  bool owned = false;
  bool owned_mmap = false;
  bool managed = false;
  CUdeviceptr device_ptr = 0;
  bool device_dirty = false;
  bool tracking_enabled = false;
  bool all_host_dirty = false;
  bool dirty_overflow = false;
  std::vector<uint64_t> host_dirty_bits;
  std::vector<uint32_t> host_dirty_log;
  uint32_t host_dirty_count = 0;
};

struct scuda_mapped_host_snapshot {
  void *host = nullptr;
  size_t size = 0;
  CUdeviceptr device_ptr = 0;
  bool device_dirty = false;
  bool managed = false;
};

using scuda_host_allocation_map =
    std::map<void *, scuda_host_allocation, std::less<void *>>;

static std::mutex &scuda_host_allocation_mutex() {
  static std::mutex mutex;
  return mutex;
}

static scuda_host_allocation_map &scuda_host_allocations() {
  static scuda_host_allocation_map allocations;
  return allocations;
}

static scuda_host_allocation_map::iterator
scuda_find_host_allocation_locked(void *p);

struct scuda_fault_entry {
  uintptr_t base = 0;
  uintptr_t end = 0;
  scuda_host_allocation *allocation = nullptr;
};

static constexpr size_t SCUDA_MAX_FAULT_ENTRIES = 1024;
static scuda_fault_entry scuda_fault_entries[SCUDA_MAX_FAULT_ENTRIES];
static volatile sig_atomic_t scuda_fault_entry_count = 0;
static struct sigaction scuda_previous_sigsegv_action;
static bool scuda_sigsegv_handler_installed = false;

static size_t scuda_page_size() {
  long page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

static size_t scuda_round_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static void scuda_call_previous_sigsegv(int sig, siginfo_t *info, void *uctx) {
  if (scuda_previous_sigsegv_action.sa_flags & SA_SIGINFO) {
    if (scuda_previous_sigsegv_action.sa_sigaction != nullptr) {
      scuda_previous_sigsegv_action.sa_sigaction(sig, info, uctx);
      return;
    }
  } else if (scuda_previous_sigsegv_action.sa_handler == SIG_IGN) {
    return;
  } else if (scuda_previous_sigsegv_action.sa_handler != SIG_DFL &&
             scuda_previous_sigsegv_action.sa_handler != nullptr) {
    scuda_previous_sigsegv_action.sa_handler(sig);
    return;
  }

  sigaction(sig, &scuda_previous_sigsegv_action, nullptr);
  raise(sig);
}

static void scuda_sigsegv_handler(int sig, siginfo_t *info, void *uctx) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(info->si_addr);
  sig_atomic_t count = scuda_fault_entry_count;
  for (sig_atomic_t i = 0; i < count; ++i) {
    const scuda_fault_entry &entry = scuda_fault_entries[i];
    if (addr < entry.base || addr >= entry.end || entry.allocation == nullptr) {
      continue;
    }

    scuda_host_allocation *allocation = entry.allocation;
    size_t page_size = allocation->page_size;
    size_t page_index = (addr - entry.base) / page_size;
    if (page_index >= allocation->page_count) {
      break;
    }

    if (!allocation->all_host_dirty) {
      size_t word = page_index / 64;
      uint64_t bit = 1ULL << (page_index % 64);
      uint64_t previous = __atomic_fetch_or(&allocation->host_dirty_bits[word],
                                            bit, __ATOMIC_RELAXED);
      if ((previous & bit) == 0) {
        uint32_t slot = __atomic_fetch_add(&allocation->host_dirty_count, 1,
                                           __ATOMIC_RELAXED);
        if (slot < allocation->page_count) {
          allocation->host_dirty_log[slot] = static_cast<uint32_t>(page_index);
        } else {
          allocation->dirty_overflow = true;
        }
      }
    }

    void *page = reinterpret_cast<void *>(entry.base + page_index * page_size);
    mprotect(page, page_size, PROT_READ | PROT_WRITE);
    return;
  }

  scuda_call_previous_sigsegv(sig, info, uctx);
}

static void scuda_install_sigsegv_handler() {
  if (scuda_sigsegv_handler_installed) {
    return;
  }
  struct sigaction action = {};
  action.sa_sigaction = scuda_sigsegv_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_SIGINFO | SA_NODEFER;
  if (sigaction(SIGSEGV, &action, &scuda_previous_sigsegv_action) == 0) {
    scuda_sigsegv_handler_installed = true;
  }
}

static bool scuda_add_fault_entry(void *base, size_t size,
                                  scuda_host_allocation *allocation) {
  if (scuda_fault_entry_count >=
      static_cast<sig_atomic_t>(SCUDA_MAX_FAULT_ENTRIES)) {
    return false;
  }
  sig_atomic_t index = scuda_fault_entry_count;
  scuda_fault_entries[index] = {
      reinterpret_cast<uintptr_t>(base),
      reinterpret_cast<uintptr_t>(base) + size,
      allocation,
  };
  scuda_fault_entry_count = index + 1;
  return true;
}

static void scuda_remove_fault_entry(void *base) {
  uintptr_t target = reinterpret_cast<uintptr_t>(base);
  sig_atomic_t count = scuda_fault_entry_count;
  for (sig_atomic_t i = 0; i < count; ++i) {
    if (scuda_fault_entries[i].base != target) {
      continue;
    }
    for (sig_atomic_t j = i + 1; j < count; ++j) {
      scuda_fault_entries[j - 1] = scuda_fault_entries[j];
    }
    scuda_fault_entry_count = count - 1;
    return;
  }
}

static bool scuda_host_flags_request_mapping(unsigned int flags) {
  return (flags & (CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTREGISTER_DEVICEMAP)) !=
         0;
}

static bool scuda_protect_host_range(void *host, size_t size, int prot) {
  if (host == nullptr || size == 0) {
    return true;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  size_t page_size = scuda_page_size();
  uintptr_t page_start = start & ~(static_cast<uintptr_t>(page_size) - 1);
  uintptr_t end = scuda_round_up(start + size, page_size);
  return mprotect(reinterpret_cast<void *>(page_start), end - page_start,
                  prot) == 0;
}

static bool
scuda_enable_dirty_tracking_locked(void *host,
                                   scuda_host_allocation *allocation) {
  if (host == nullptr || allocation == nullptr || allocation->device_ptr == 0) {
    return true;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(host);
  if ((base % allocation->page_size) != 0 ||
      (allocation->storage_size % allocation->page_size) != 0) {
    return true;
  }

  allocation->tracking_enabled = true;
  allocation->all_host_dirty = true;
  allocation->host_dirty_bits.assign((allocation->page_count + 63) / 64, 0);
  allocation->host_dirty_log.assign(allocation->page_count, 0);
  allocation->host_dirty_count = 0;
  allocation->dirty_overflow = false;

  scuda_install_sigsegv_handler();
  if (!scuda_add_fault_entry(host, allocation->storage_size, allocation)) {
    allocation->tracking_enabled = false;
    return false;
  }
  if (!scuda_protect_host_range(host, allocation->storage_size, PROT_READ)) {
    scuda_remove_fault_entry(host);
    allocation->tracking_enabled = false;
    return false;
  }
  return true;
}

static void scuda_disable_dirty_tracking(void *host,
                                         scuda_host_allocation &allocation) {
  if (!allocation.tracking_enabled) {
    return;
  }
  scuda_protect_host_range(host, allocation.storage_size,
                           PROT_READ | PROT_WRITE);
  scuda_remove_fault_entry(host);
  allocation.tracking_enabled = false;
}

static std::vector<scuda_mapped_host_snapshot> scuda_mapped_host_snapshots() {
  std::vector<scuda_mapped_host_snapshot> snapshots;
  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  for (const auto &entry : scuda_host_allocations()) {
    if (entry.second.device_ptr != 0) {
      snapshots.push_back({entry.first, entry.second.size,
                           entry.second.device_ptr, entry.second.device_dirty,
                           entry.second.managed});
    }
  }
  return snapshots;
}

static bool scuda_dirty_bit_is_set(const scuda_host_allocation &allocation,
                                   size_t page_index) {
  size_t word = page_index / 64;
  uint64_t bit = 1ULL << (page_index % 64);
  return (allocation.host_dirty_bits[word] & bit) != 0;
}

static void scuda_clear_dirty_bit(scuda_host_allocation &allocation,
                                  size_t page_index) {
  size_t word = page_index / 64;
  uint64_t bit = 1ULL << (page_index % 64);
  allocation.host_dirty_bits[word] &= ~bit;
}

static CUresult scuda_copy_host_range_to_device(void *host,
                                                CUdeviceptr device_ptr,
                                                size_t offset, size_t bytes) {
  return cuMemcpyHtoD_v2(device_ptr + offset,
                         static_cast<unsigned char *>(host) + offset, bytes);
}

static CUresult
scuda_sync_dirty_host_pages_to_device(void *host,
                                      scuda_host_allocation &allocation) {
  if (allocation.device_ptr == 0 || allocation.size == 0) {
    return CUDA_SUCCESS;
  }

  if (!allocation.tracking_enabled) {
    return scuda_copy_host_range_to_device(host, allocation.device_ptr, 0,
                                           allocation.size);
  }

  if (allocation.all_host_dirty) {
    CUresult result = scuda_copy_host_range_to_device(
        host, allocation.device_ptr, 0, allocation.size);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    scuda_protect_host_range(host, allocation.storage_size, PROT_READ);
    std::fill(allocation.host_dirty_bits.begin(),
              allocation.host_dirty_bits.end(), 0);
    allocation.host_dirty_count = 0;
    allocation.dirty_overflow = false;
    allocation.all_host_dirty = false;
    return CUDA_SUCCESS;
  }

  std::vector<uint32_t> pages;
  uint32_t dirty_count = allocation.host_dirty_count;
  if (dirty_count > allocation.page_count) {
    dirty_count = static_cast<uint32_t>(allocation.page_count);
    allocation.dirty_overflow = true;
  }

  if (allocation.dirty_overflow) {
    for (size_t page = 0; page < allocation.page_count; ++page) {
      if (scuda_dirty_bit_is_set(allocation, page)) {
        pages.push_back(static_cast<uint32_t>(page));
      }
    }
  } else {
    pages.assign(allocation.host_dirty_log.begin(),
                 allocation.host_dirty_log.begin() + dirty_count);
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    pages.erase(std::remove_if(pages.begin(), pages.end(),
                               [&](uint32_t page) {
                                 return !scuda_dirty_bit_is_set(allocation,
                                                                page);
                               }),
                pages.end());
  }

  size_t i = 0;
  while (i < pages.size()) {
    size_t first = pages[i];
    size_t last = first;
    ++i;
    while (i < pages.size() && pages[i] == last + 1) {
      last = pages[i];
      ++i;
    }

    size_t offset = first * allocation.page_size;
    size_t bytes = std::min((last - first + 1) * allocation.page_size,
                            allocation.size - offset);
    CUresult result = scuda_copy_host_range_to_device(
        host, allocation.device_ptr, offset, bytes);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    scuda_protect_host_range(static_cast<unsigned char *>(host) + offset, bytes,
                             PROT_READ);
    for (size_t page = first; page <= last; ++page) {
      scuda_clear_dirty_bit(allocation, page);
    }
  }

  allocation.host_dirty_count = 0;
  allocation.dirty_overflow = false;
  return CUDA_SUCCESS;
}

extern "C" void scuda_mark_host_range_clean(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  auto it = scuda_find_host_allocation_locked(host);
  if (it == scuda_host_allocations().end() || !it->second.tracking_enabled) {
    return;
  }

  auto &allocation = it->second;
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  uintptr_t end = std::min(start + size, base + allocation.size);
  if (start >= end) {
    return;
  }

  size_t first_page = (start - base) / allocation.page_size;
  size_t last_page = (end - 1 - base) / allocation.page_size;
  bool covers_all = first_page == 0 && end >= base + allocation.size;

  if (allocation.all_host_dirty && covers_all) {
    allocation.all_host_dirty = false;
    std::fill(allocation.host_dirty_bits.begin(),
              allocation.host_dirty_bits.end(), 0);
    allocation.host_dirty_count = 0;
    allocation.dirty_overflow = false;
  } else if (!allocation.all_host_dirty) {
    for (size_t page = first_page; page <= last_page; ++page) {
      scuda_clear_dirty_bit(allocation, page);
    }
  }

  uintptr_t protect_start = base + first_page * allocation.page_size;
  size_t protect_size = (last_page - first_page + 1) * allocation.page_size;
  scuda_protect_host_range(reinterpret_cast<void *>(protect_start),
                           protect_size, PROT_READ);
}

extern "C" void scuda_prepare_host_range_write(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  auto it = scuda_find_host_allocation_locked(host);
  if (it == scuda_host_allocations().end() || !it->second.tracking_enabled) {
    return;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  uintptr_t end = std::min(start + size, base + it->second.size);
  if (start >= end) {
    return;
  }

  size_t first_page = (start - base) / it->second.page_size;
  size_t last_page = (end - 1 - base) / it->second.page_size;
  uintptr_t protect_start = base + first_page * it->second.page_size;
  size_t protect_size = (last_page - first_page + 1) * it->second.page_size;
  scuda_protect_host_range(reinterpret_cast<void *>(protect_start),
                           protect_size, PROT_READ | PROT_WRITE);
}

CUresult scuda_sync_mapped_host_to_device() {
  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  for (auto &entry : scuda_host_allocations()) {
    CUresult result =
        scuda_sync_dirty_host_pages_to_device(entry.first, entry.second);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  return CUDA_SUCCESS;
}

static bool scuda_device_ptr_in_mapping(CUdeviceptr ptr,
                                        const scuda_mapped_host_snapshot &m) {
  return ptr >= m.device_ptr && ptr < m.device_ptr + m.size;
}

static bool scuda_host_ptr_in_mapping(CUdeviceptr ptr,
                                      const scuda_mapped_host_snapshot &m,
                                      CUdeviceptr *translated) {
  uintptr_t host = reinterpret_cast<uintptr_t>(m.host);
  if (ptr < host || ptr >= host + m.size) {
    return false;
  }
  if (translated != nullptr) {
    *translated = m.device_ptr + (ptr - host);
  }
  return true;
}

static void scuda_mark_mapped_device_dirty(void *host) {
  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  auto it = scuda_host_allocations().find(host);
  if (it != scuda_host_allocations().end()) {
    it->second.device_dirty = true;
  }
}

CUresult scuda_sync_mapped_host_to_device_for_launch(unsigned char *packed,
                                                     const size_t *offsets,
                                                     const size_t *sizes,
                                                     uint32_t count) {
  if (packed == nullptr || offsets == nullptr || sizes == nullptr) {
    return count == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = scuda_sync_mapped_host_to_device();
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::vector<scuda_mapped_host_snapshot> snapshots =
      scuda_mapped_host_snapshots();
  for (const auto &mapping : snapshots) {
    for (uint32_t i = 0; i < count; ++i) {
      if (sizes[i] != sizeof(CUdeviceptr)) {
        continue;
      }
      CUdeviceptr arg = 0;
      memcpy(&arg, packed + offsets[i], sizeof(arg));
      CUdeviceptr translated = 0;
      if (mapping.managed &&
          scuda_host_ptr_in_mapping(arg, mapping, &translated)) {
        memcpy(packed + offsets[i], &translated, sizeof(translated));
        scuda_mark_mapped_device_dirty(mapping.host);
        break;
      }
      if (scuda_device_ptr_in_mapping(arg, mapping)) {
        scuda_mark_mapped_device_dirty(mapping.host);
        break;
      }
    }
  }
  return CUDA_SUCCESS;
}

static CUresult scuda_sync_mapped_device_to_host() {
  for (const auto &mapping : scuda_mapped_host_snapshots()) {
    if (!mapping.device_dirty || mapping.size == 0) {
      continue;
    }
    CUresult result =
        cuMemcpyDtoH_v2(mapping.host, mapping.device_ptr, mapping.size);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
    auto it = scuda_host_allocations().find(mapping.host);
    if (it != scuda_host_allocations().end()) {
      it->second.device_dirty = false;
    }
  }
  return CUDA_SUCCESS;
}

static scuda_host_allocation_map::iterator
scuda_find_host_allocation_locked(void *p) {
  auto &allocations = scuda_host_allocations();
  auto exact = allocations.find(p);
  if (exact != allocations.end()) {
    return exact;
  }

  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  for (auto it = allocations.begin(); it != allocations.end(); ++it) {
    uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    if (addr >= base && addr < base + it->second.size) {
      return it;
    }
  }
  return allocations.end();
}

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize,
                                   unsigned int Flags) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr unsigned int supported_flags = CU_MEMHOSTALLOC_PORTABLE |
                                           CU_MEMHOSTALLOC_DEVICEMAP |
                                           CU_MEMHOSTALLOC_WRITECOMBINED;
  if ((Flags & ~supported_flags) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr device_ptr = 0;
  if (scuda_host_flags_request_mapping(Flags)) {
    CUresult result = cuMemAlloc_v2(&device_ptr, bytesize);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  void *ptr = nullptr;
  size_t page_size = scuda_page_size();
  size_t storage_size = scuda_round_up(bytesize, page_size);
  bool owned_mmap = device_ptr != 0;
  if (owned_mmap) {
    ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      ptr = nullptr;
    }
  } else if (posix_memalign(&ptr, page_size, bytesize) != 0) {
    ptr = nullptr;
  }
  if (ptr == nullptr) {
    if (device_ptr != 0) {
      cuMemFree_v2(device_ptr);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
    scuda_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.flags = Flags;
    allocation.owned = true;
    allocation.owned_mmap = owned_mmap;
    allocation.device_ptr = device_ptr;
    auto inserted =
        scuda_host_allocations().emplace(ptr, std::move(allocation));
    if (!scuda_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      scuda_host_allocations().erase(inserted.first);
      if (owned_mmap) {
        munmap(ptr, storage_size);
      } else {
        free(ptr);
      }
      if (device_ptr != 0) {
        cuMemFree_v2(device_ptr);
      }
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  *pp = ptr;
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA local cuMemHostAlloc ptr=" << ptr
              << " bytes=" << bytesize << " flags=" << Flags << std::endl;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize, 0);
}

#ifdef cuMemAllocHost
#undef cuMemAllocHost
#endif
extern "C" CUresult cuMemAllocHost(void **pp, size_t bytesize) {
  return cuMemAllocHost_v2(pp, bytesize);
}

extern "C" CUresult cuMemFreeHost(void *p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  bool owned = false;
  bool owned_mmap = false;
  size_t storage_size = 0;
  CUdeviceptr device_ptr = 0;
  {
    std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
    auto &allocations = scuda_host_allocations();
    auto it = allocations.find(p);
    if (it == allocations.end()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    owned = it->second.owned;
    owned_mmap = it->second.owned_mmap;
    storage_size = it->second.storage_size;
    device_ptr = it->second.device_ptr;
    scuda_disable_dirty_tracking(p, it->second);
    allocations.erase(it);
  }
  if (owned) {
    if (owned_mmap) {
      munmap(p, storage_size);
    } else {
      free(p);
    }
  }
  if (device_ptr != 0) {
    return cuMemFree_v2(device_ptr);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p,
                                                 unsigned int Flags) {
  if (pdptr == nullptr || p == nullptr || Flags != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  auto it = scuda_find_host_allocation_locked(p);
  if (it == scuda_host_allocations().end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((it->second.flags & CU_MEMHOSTALLOC_DEVICEMAP) == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  *pdptr = it->second.device_ptr + (addr - base);
  return CUDA_SUCCESS;
}

#ifdef cuMemHostGetDevicePointer
#undef cuMemHostGetDevicePointer
#endif
extern "C" CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                              unsigned int Flags) {
  return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  if (pFlags == nullptr || p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  auto it = scuda_find_host_allocation_locked(p);
  if (it == scuda_host_allocations().end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pFlags = it->second.flags;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemHostRegister_v2(void *p, size_t bytesize,
                                         unsigned int Flags) {
  if (p == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr unsigned int supported_flags =
      CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP |
      CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_READ_ONLY;
  if ((Flags & ~supported_flags) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr device_ptr = 0;
  if (scuda_host_flags_request_mapping(Flags)) {
    CUresult result = cuMemAlloc_v2(&device_ptr, bytesize);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
  if (scuda_host_allocations().find(p) != scuda_host_allocations().end()) {
    if (device_ptr != 0) {
      cuMemFree_v2(device_ptr);
    }
    return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
  }
  size_t page_size = scuda_page_size();
  scuda_host_allocation allocation;
  allocation.size = bytesize;
  allocation.storage_size = bytesize;
  allocation.page_size = page_size;
  allocation.page_count = scuda_round_up(bytesize, page_size) / page_size;
  allocation.flags = Flags;
  allocation.owned = false;
  allocation.owned_mmap = false;
  allocation.device_ptr = device_ptr;
  auto inserted = scuda_host_allocations().emplace(p, std::move(allocation));
  if (!scuda_enable_dirty_tracking_locked(p, &inserted.first->second)) {
    scuda_host_allocations().erase(inserted.first);
    if (device_ptr != 0) {
      cuMemFree_v2(device_ptr);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  return CUDA_SUCCESS;
}

#ifdef cuMemHostRegister
#undef cuMemHostRegister
#endif
extern "C" CUresult cuMemHostRegister(void *p, size_t bytesize,
                                      unsigned int Flags) {
  return cuMemHostRegister_v2(p, bytesize, Flags);
}

extern "C" CUresult cuMemHostUnregister(void *p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdeviceptr device_ptr = 0;
  {
    std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
    auto &allocations = scuda_host_allocations();
    auto it = allocations.find(p);
    if (it == allocations.end() || it->second.owned) {
      return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
    device_ptr = it->second.device_ptr;
    scuda_disable_dirty_tracking(p, it->second);
    allocations.erase(it);
  }
  if (device_ptr != 0) {
    return cuMemFree_v2(device_ptr);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_cuMemAllocManaged_safe(CUdeviceptr *dptr,
                                                 size_t bytesize,
                                                 unsigned int flags) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr device_ptr = 0;
  CUresult result = cuMemAlloc_v2(&device_ptr, bytesize);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  size_t page_size = scuda_page_size();
  size_t storage_size = scuda_round_up(bytesize, page_size);
  void *ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    cuMemFree_v2(device_ptr);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
    scuda_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.flags = flags;
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.managed = true;
    allocation.device_ptr = device_ptr;
    auto inserted =
        scuda_host_allocations().emplace(ptr, std::move(allocation));
    if (!inserted.second) {
      munmap(ptr, storage_size);
      cuMemFree_v2(device_ptr);
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (!scuda_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      scuda_host_allocations().erase(inserted.first);
      munmap(ptr, storage_size);
      cuMemFree_v2(device_ptr);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }

  *dptr = reinterpret_cast<CUdeviceptr>(ptr);
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_cuMemFree_v2_safe(CUdeviceptr dptr) {
  void *host = reinterpret_cast<void *>(dptr);
  scuda_host_allocation allocation;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(scuda_host_allocation_mutex());
    auto it = scuda_find_host_allocation_locked(host);
    if (it != scuda_host_allocations().end() &&
        reinterpret_cast<void *>(dptr) == it->first && it->second.managed) {
      allocation = std::move(it->second);
      scuda_disable_dirty_tracking(it->first, allocation);
      scuda_host_allocations().erase(it);
      found = true;
    }
  }
  if (!found) {
    conn_t *conn = scuda_rpc_conn_for_deviceptr(dptr);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemFree_v2) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      scuda_forget_deviceptr_owner(dptr);
    }
    return return_value;
  }

  CUresult result = CUDA_SUCCESS;
  if (allocation.device_ptr != 0) {
    result = cuMemFree_v2(allocation.device_ptr);
  }
  if (allocation.owned_mmap && host != nullptr) {
    munmap(host, allocation.storage_size);
  } else if (allocation.owned && host != nullptr) {
    free(host);
  }
  return result;
}

extern "C" CUresult cuPointerGetAttribute(void *data,
                                          CUpointer_attribute attribute,
                                          CUdeviceptr ptr) {
  if (data == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!scuda_pointer_attribute_size(attribute, &value_size)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  unsigned char value[64] = {};
  if (value_size > sizeof(value)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuPointerGetAttribute) < 0 ||
      rpc_write(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_write(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, value_size) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    memcpy(data, value, value_size);
  }
  return return_value;
}

extern "C" CUresult
scuda_cuPointerGetAttributes_safe(unsigned int numAttributes,
                                  CUpointer_attribute *attributes, void **data,
                                  CUdeviceptr ptr) {
  if (numAttributes != 0 && (attributes == nullptr || data == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::vector<size_t> value_sizes(numAttributes, 0);
  for (unsigned int i = 0; i < numAttributes; ++i) {
    if (!scuda_pointer_attribute_size(attributes[i], &value_sizes[i]) ||
        value_sizes[i] > 64) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  }

  conn_t *conn = rpc_client_get_connection(0);
  if (rpc_write_start_request(conn, SCUDA_RPC_cuPointerGetAttributes) < 0 ||
      rpc_write(conn, &numAttributes, sizeof(numAttributes)) < 0 ||
      (numAttributes != 0 &&
       rpc_write(conn, attributes,
                 numAttributes * sizeof(CUpointer_attribute)) < 0) ||
      rpc_write(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  std::vector<std::vector<unsigned char>> values(numAttributes);
  for (unsigned int i = 0; i < numAttributes; ++i) {
    size_t remote_size = 0;
    if (rpc_read(conn, &remote_size, sizeof(remote_size)) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (remote_size > 64) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    values[i].resize(remote_size);
    if (remote_size != 0 && rpc_read(conn, values[i].data(), remote_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }

  CUresult return_value;
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (return_value == CUDA_SUCCESS) {
    for (unsigned int i = 0; i < numAttributes; ++i) {
      if (data[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      if (values[i].size() > value_sizes[i]) {
        return CUDA_ERROR_NOT_SUPPORTED;
      }
      memcpy(data[i], values[i].data(), values[i].size());
    }
  }
  return return_value;
}

#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
extern "C" CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                         CUdeviceptr dptr) {
  return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
extern "C" CUresult cuMemGetInfo(size_t *free, size_t *total) {
  return cuMemGetInfo_v2(free, total);
}

extern "C" CUresult
scuda_cuArrayCreate_v2_safe(CUarray *pHandle,
                            const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  if (pHandle == nullptr || pAllocateArray == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuArrayCreate_v2) < 0 ||
      rpc_write(conn, pAllocateArray, sizeof(*pAllocateArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(*pHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
scuda_cuArray3DCreate_v2_safe(CUarray *pHandle,
                              const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  if (pHandle == nullptr || pAllocateArray == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuArray3DCreate_v2) < 0 ||
      rpc_write(conn, pAllocateArray, sizeof(*pAllocateArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(*pHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult scuda_rpc_noarg_driver_call(int op) {
  conn_t *conn = scuda_rpc_conn_for_current_context();
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult scuda_rpc_stream_driver_call(int op, CUstream stream) {
  conn_t *conn = stream == nullptr ? scuda_rpc_conn_for_current_context()
                                   : scuda_rpc_conn_for_stream(stream);
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult scuda_rpc_event_driver_call(int op, CUevent event) {
  conn_t *conn = scuda_rpc_conn_for_event(event);
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &event, sizeof(event)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuCtxSynchronize() {
  CUresult result = scuda_rpc_noarg_driver_call(RPC_cuCtxSynchronize);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return scuda_sync_mapped_device_to_host();
}

extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
  conn_t *conn = hStream == nullptr ? scuda_rpc_conn_for_current_context()
                                    : scuda_rpc_conn_for_stream(hStream);
  CUresult result = CUDA_ERROR_UNKNOWN;
  uint32_t copy_count = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamSynchronize) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &copy_count, sizeof(copy_count)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  for (uint32_t i = 0; i < copy_count; ++i) {
    void *dst = nullptr;
    size_t bytes = 0;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0 ||
        rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (bytes != 0) {
      scuda_prepare_host_range_write(dst, bytes);
      if (rpc_read(conn, dst, bytes) < 0) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      scuda_mark_host_range_clean(dst, bytes);
    }
  }
  if (rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return scuda_sync_mapped_device_to_host();
}

#ifdef cuStreamSynchronize_ptsz
#undef cuStreamSynchronize_ptsz
#endif
extern "C" CUresult cuStreamSynchronize_ptsz(CUstream hStream) {
  return cuStreamSynchronize(hStream);
}

extern "C" CUresult cuEventSynchronize(CUevent hEvent) {
  CUresult result = scuda_rpc_event_driver_call(RPC_cuEventSynchronize, hEvent);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return scuda_sync_mapped_device_to_host();
}

extern "C" CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags,
                                   CUdevice dev) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = scuda_rpc_conn_for_device(&dev);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, SCUDA_RPC_cuCtxCreate_v2) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_write(conn, &dev, sizeof(dev)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    scuda_note_ctx_create(*pctx, conn);
  }
  return return_value;
}

#if CUDA_VERSION >= 12050
extern "C" CUresult cuCtxCreate_v4(CUcontext *pctx,
                                   CUctxCreateParams *ctxCreateParams,
                                   unsigned int flags, CUdevice dev) {
  if (ctxCreateParams != nullptr &&
      (ctxCreateParams->execAffinityParams != nullptr ||
       ctxCreateParams->numExecAffinityParams != 0 ||
       ctxCreateParams->cigParams != nullptr)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuCtxCreate_v2(pctx, flags, dev);
}
#endif

static CUresult scuda_occupancy_max_potential_block_size(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags, bool with_flags) {
  if (minGridSize == nullptr || blockSize == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (blockSizeToDynamicSMemSize != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  int op = with_flags ? RPC_cuOccupancyMaxPotentialBlockSizeWithFlags
                      : RPC_cuOccupancyMaxPotentialBlockSize;
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
      rpc_write(conn, &blockSizeLimit, sizeof(blockSizeLimit)) < 0 ||
      (with_flags && rpc_write(conn, &flags, sizeof(flags)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, minGridSize, sizeof(int)) < 0 ||
      rpc_read(conn, blockSize, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                 CUfunction func,
                                 CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                 size_t dynamicSMemSize, int blockSizeLimit) {
  return scuda_occupancy_max_potential_block_size(
      minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
      blockSizeLimit, 0, false);
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  return scuda_occupancy_max_potential_block_size(
      minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
      blockSizeLimit, flags, true);
}

static bool scuda_pack_module_image(const void *image, uint32_t *kind,
                                    std::vector<unsigned char> *bytes) {
  if (image == nullptr || kind == nullptr || bytes == nullptr) {
    return false;
  }

  const auto *wrapper = reinterpret_cast<const scuda_fatbin_wrapper *>(image);
  const void *fatbin = image;
  if (wrapper->magic == SCUDA_FATBINC_MAGIC &&
      (wrapper->version == 1 || wrapper->version == 2) &&
      wrapper->data != nullptr) {
    fatbin = wrapper->data;
    *kind = wrapper->version == 2 ? SCUDA_MODULE_IMAGE_FATBINC_V2
                                  : SCUDA_MODULE_IMAGE_FATBINC_V1;
  } else {
    *kind = SCUDA_MODULE_IMAGE_FATBIN_RAW;
  }

  const auto *header = reinterpret_cast<const scuda_fatbin_header *>(fatbin);
  if (header->magic != SCUDA_FATBIN_MAGIC || header->header_size == 0) {
    const auto *elf = static_cast<const unsigned char *>(image);
    if (std::memcmp(elf, ELFMAG, SELFMAG) == 0 && elf[EI_CLASS] == ELFCLASS64) {
      const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
      size_t image_size = sizeof(Elf64_Ehdr);
      if (ehdr->e_phoff != 0 && ehdr->e_phentsize == sizeof(Elf64_Phdr)) {
        image_size =
            std::max(image_size, static_cast<size_t>(ehdr->e_phoff) +
                                     static_cast<size_t>(ehdr->e_phnum) *
                                         sizeof(Elf64_Phdr));
        const auto *phdrs =
            reinterpret_cast<const Elf64_Phdr *>(elf + ehdr->e_phoff);
        for (int i = 0; i < ehdr->e_phnum; ++i) {
          image_size =
              std::max(image_size, static_cast<size_t>(phdrs[i].p_offset) +
                                       static_cast<size_t>(phdrs[i].p_filesz));
        }
      }
      if (ehdr->e_shoff != 0 && ehdr->e_shentsize == sizeof(Elf64_Shdr)) {
        image_size =
            std::max(image_size, static_cast<size_t>(ehdr->e_shoff) +
                                     static_cast<size_t>(ehdr->e_shnum) *
                                         sizeof(Elf64_Shdr));
        const auto *shdrs =
            reinterpret_cast<const Elf64_Shdr *>(elf + ehdr->e_shoff);
        for (int i = 0; i < ehdr->e_shnum; ++i) {
          if (shdrs[i].sh_type != SHT_NOBITS) {
            image_size =
                std::max(image_size, static_cast<size_t>(shdrs[i].sh_offset) +
                                         static_cast<size_t>(shdrs[i].sh_size));
          }
        }
      }
      *kind = SCUDA_MODULE_IMAGE_FATBIN_RAW;
      bytes->assign(elf, elf + image_size);
      return true;
    }

    const char *ptx = static_cast<const char *>(image);
    const char *ptx_start = ptx;
    while (*ptx_start == ' ' || *ptx_start == '\t' || *ptx_start == '\r' ||
           *ptx_start == '\n') {
      ++ptx_start;
    }
    if (std::strncmp(ptx_start, ".version", 8) == 0 ||
        std::strncmp(ptx_start, "//", 2) == 0) {
      *kind = SCUDA_MODULE_IMAGE_FATBIN_RAW;
      bytes->assign(reinterpret_cast<const unsigned char *>(ptx),
                    reinterpret_cast<const unsigned char *>(ptx) +
                        std::strlen(ptx) + 1);
      return true;
    }

    return false;
  }
  size_t image_size = static_cast<size_t>(header->header_size) +
                      static_cast<size_t>(header->files_size);
  const auto *raw = static_cast<const unsigned char *>(fatbin);
  bytes->assign(raw, raw + image_size);
  return true;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  if (module == nullptr || image == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  uint32_t kind = 0;
  std::vector<unsigned char> image_bytes;
  if (!scuda_pack_module_image(image, &kind, &image_bytes)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = scuda_rpc_conn_for_current_context();
  CUresult return_value;
  size_t image_size = image_bytes.size();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleLoadData) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
      rpc_write(conn, image_bytes.data(), image_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, module, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    scuda_remember_loaded_module(*module);
    scuda_note_module_owner(*module, conn);
  }
  return return_value;
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                       unsigned int numOptions,
                                       CUjit_option *options,
                                       void **optionValues) {
  (void)options;
  (void)optionValues;
  (void)numOptions;
  return cuModuleLoadData(module, image);
}

extern "C" CUresult
cuLibraryLoadData(CUlibrary *library, const void *code,
                  CUjit_option *jitOptions, void **jitOptionsValues,
                  unsigned int numJitOptions, CUlibraryOption *libraryOptions,
                  void **libraryOptionValues, unsigned int numLibraryOptions) {
  if (library == nullptr || code == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  bool can_ignore_library_options =
      numLibraryOptions == 1 && libraryOptions != nullptr &&
      (libraryOptions[0] == CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE ||
       libraryOptions[0] == CU_LIBRARY_BINARY_IS_PRESERVED);
  if (numJitOptions != 0 ||
      !(numLibraryOptions == 0 || can_ignore_library_options)) {
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA cuLibraryLoadData unsupported options: jit="
                << numJitOptions << " library=" << numLibraryOptions
                << " first_library_option="
                << (libraryOptions == nullptr ? -1 : (int)libraryOptions[0])
                << std::endl;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  uint32_t kind = 0;
  std::vector<unsigned char> image_bytes;
  if (!scuda_pack_module_image(code, &kind, &image_bytes)) {
    if (scuda_trace_enabled()) {
      const auto *wrapper =
          reinterpret_cast<const scuda_fatbin_wrapper *>(code);
      std::cerr << "SCUDA cuLibraryLoadData could not pack image" << " magic=0x"
                << std::hex << wrapper->magic << " version=0x"
                << wrapper->version << std::dec << std::endl;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = scuda_rpc_conn_for_current_context();
  CUresult return_value;
  size_t image_size = image_bytes.size();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
      rpc_write(conn, image_bytes.data(), image_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult
scuda_get_kernel_param_layout(CUfunction f, scuda_kernel_param_layout *layout) {
  if (layout == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  f = scuda_translate_private_function(f);
  conn_t *conn = scuda_rpc_conn_for_function(f);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, SCUDA_RPC_cuFuncGetParamLayout) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &layout->count, sizeof(layout->count)) < 0 ||
      rpc_read(conn, layout->offsets, sizeof(layout->offsets)) < 0 ||
      rpc_read(conn, layout->sizes, sizeof(layout->sizes)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
               unsigned int gridDimZ, unsigned int blockDimX,
               unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void **kernelParams, void **extra) {
  if (extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  f = scuda_translate_private_function(f);

  scuda_kernel_param_layout layout;
  CUresult status = scuda_get_kernel_param_layout(f, &layout);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (layout.count > 64) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  size_t total_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    if (kernelParams == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    total_size = std::max(total_size, layout.offsets[i] + layout.sizes[i]);
  }

  std::vector<unsigned char> packed(total_size);
  for (uint32_t i = 0; i < layout.count; ++i) {
    memcpy(packed.data() + layout.offsets[i], kernelParams[i], layout.sizes[i]);
  }

  status = scuda_sync_mapped_host_to_device_for_launch(
      packed.data(), layout.offsets, layout.sizes, layout.count);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  conn_t *conn = scuda_rpc_conn_for_function(f);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchKernel) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      rpc_write(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_write(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_write(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_write(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_write(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_write(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_write(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, &total_size, sizeof(total_size)) < 0 ||
      rpc_write(conn, packed.data(), packed.size()) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                     void **kernelParams, void **extra) {
  if (config == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (config->numAttrs != 0 && scuda_trace_enabled()) {
    std::cerr << "SCUDA cuLaunchKernelEx ignoring " << config->numAttrs
              << " launch attributes" << std::endl;
  }
  return cuLaunchKernel(f, config->gridDimX, config->gridDimY, config->gridDimZ,
                        config->blockDimX, config->blockDimY, config->blockDimZ,
                        config->sharedMemBytes, config->hStream, kernelParams,
                        extra);
}

extern "C" CUresult scuda_cuFuncGetAttribute_safe(int *pi,
                                                  CUfunction_attribute attrib,
                                                  CUfunction hfunc) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = scuda_translate_private_function(hfunc);
  if (translated != hfunc) {
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA translated cuFuncGetAttribute attr=" << attrib
                << " client=" << hfunc << " server=" << translated << std::endl;
    }
    return cuFuncGetAttribute(pi, attrib, translated);
  }
  return cuFuncGetAttribute(pi, attrib, hfunc);
}

extern "C" CUresult scuda_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  if (numBlocks == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = scuda_translate_private_function(func);
  if (translated != func) {
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA translated "
                   "cuOccupancyMaxActiveBlocksPerMultiprocessor"
                << " client=" << func << " server=" << translated
                << " blockSize=" << blockSize
                << " dynamicSMemSize=" << dynamicSMemSize << std::endl;
    }
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, translated, blockSize, dynamicSMemSize);
  }
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                     dynamicSMemSize);
}

extern "C" CUresult
scuda_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_safe(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (numBlocks == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = scuda_translate_private_function(func);
  if (translated != func) {
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA translated "
                   "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
                << " client=" << func << " server=" << translated
                << " blockSize=" << blockSize
                << " dynamicSMemSize=" << dynamicSMemSize << " flags=" << flags
                << std::endl;
    }
    return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, translated, blockSize, dynamicSMemSize, flags);
  }
  return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      numBlocks, func, blockSize, dynamicSMemSize, flags);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount, CUstream hStream) {
  conn_t *conn = scuda_rpc_conn_for_deviceptr(srcDevice);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoHAsync_v2) < 0 ||
      rpc_write(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  scuda_prepare_host_range_write(dstHost, ByteCount);
  if (rpc_read(conn, dstHost, ByteCount) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    scuda_mark_host_range_clean(dstHost, ByteCount);
  }
  return return_value;
}

#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
extern "C" CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

static bool scuda_is_local_address(const void *ptr) {
  if (ptr == nullptr) {
    return false;
  }
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  uintptr_t page = reinterpret_cast<uintptr_t>(ptr) &
                   ~(static_cast<uintptr_t>(page_size) - 1);
  unsigned char vec = 0;
  return mincore(reinterpret_cast<void *>(page), page_size, &vec) == 0;
}

static size_t scuda_memcpy3d_host_span(const CUDA_MEMCPY3D &copy, bool source) {
  size_t x = source ? copy.srcXInBytes : copy.dstXInBytes;
  size_t y = source ? copy.srcY : copy.dstY;
  size_t z = source ? copy.srcZ : copy.dstZ;
  size_t pitch = source ? copy.srcPitch : copy.dstPitch;
  size_t height = source ? copy.srcHeight : copy.dstHeight;
  if (pitch == 0) {
    pitch = copy.WidthInBytes + x;
  }
  if (height == 0) {
    height = copy.Height + y;
  }
  if (copy.WidthInBytes == 0 || copy.Height == 0 || copy.Depth == 0) {
    return 0;
  }
  return (z + copy.Depth - 1) * pitch * height + (y + copy.Height - 1) * pitch +
         x + copy.WidthInBytes;
}

static size_t scuda_memcpy2d_host_span(const CUDA_MEMCPY2D &copy, bool source) {
  size_t x = source ? copy.srcXInBytes : copy.dstXInBytes;
  size_t y = source ? copy.srcY : copy.dstY;
  size_t pitch = source ? copy.srcPitch : copy.dstPitch;
  if (pitch == 0) {
    pitch = copy.WidthInBytes + x;
  }
  if (copy.WidthInBytes == 0 || copy.Height == 0) {
    return 0;
  }
  return (y + copy.Height - 1) * pitch + x + copy.WidthInBytes;
}

static int scuda_memcpy2d_rpc_op(bool async, bool unaligned) {
  if (async) {
    return RPC_cuMemcpy2DAsync_v2;
  }
  if (unaligned) {
    return RPC_cuMemcpy2DUnaligned_v2;
  }
  return RPC_cuMemcpy2D_v2;
}

static CUresult scuda_cuMemcpy2D_common(const CUDA_MEMCPY2D *pCopy,
                                        CUstream stream, bool async,
                                        bool unaligned) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUDA_MEMCPY2D copy = *pCopy;
  size_t src_host_size = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                             ? scuda_memcpy2d_host_span(copy, true)
                             : 0;
  size_t dst_host_size = copy.dstMemoryType == CU_MEMORYTYPE_HOST
                             ? scuda_memcpy2d_host_span(copy, false)
                             : 0;
  const void *src_host = copy.srcHost;
  void *dst_host = copy.dstHost;
  if ((src_host_size != 0 && src_host == nullptr) ||
      (dst_host_size != 0 && dst_host == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = nullptr;
  if (copy.srcMemoryType == CU_MEMORYTYPE_DEVICE && copy.srcDevice != 0) {
    conn = scuda_rpc_conn_for_deviceptr(copy.srcDevice);
  } else if (copy.dstMemoryType == CU_MEMORYTYPE_DEVICE &&
             copy.dstDevice != 0) {
    conn = scuda_rpc_conn_for_deviceptr(copy.dstDevice);
  } else if (stream != nullptr) {
    conn = scuda_rpc_conn_for_stream(stream);
  } else {
    conn = scuda_rpc_conn_for_current_context();
  }
  CUresult return_value;
  size_t returned_dst_size = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, scuda_memcpy2d_rpc_op(async, unaligned)) <
          0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &src_host_size, sizeof(src_host_size)) < 0 ||
      (src_host_size != 0 && rpc_write(conn, src_host, src_host_size) < 0) ||
      rpc_write(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      (async && rpc_write(conn, &stream, sizeof(stream)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size > dst_host_size) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size != 0) {
    scuda_prepare_host_range_write(dst_host, returned_dst_size);
    if (rpc_read(conn, dst_host, returned_dst_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && returned_dst_size != 0) {
    scuda_mark_host_range_clean(dst_host, returned_dst_size);
  }
  return return_value;
}

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  return scuda_cuMemcpy2D_common(pCopy, nullptr, false, false);
}

#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif
extern "C" CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2D_v2(pCopy);
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  return scuda_cuMemcpy2D_common(pCopy, nullptr, false, true);
}

#ifdef cuMemcpy2DUnaligned
#undef cuMemcpy2DUnaligned
#endif
extern "C" CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2DUnaligned_v2(pCopy);
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                       CUstream hStream) {
  return scuda_cuMemcpy2D_common(pCopy, hStream, true, false);
}

#ifdef cuMemcpy2DAsync
#undef cuMemcpy2DAsync
#endif
extern "C" CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy2DAsync_ptsz
#undef cuMemcpy2DAsync_ptsz
#endif
extern "C" CUresult cuMemcpy2DAsync_ptsz(const CUDA_MEMCPY2D *pCopy,
                                         CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

static CUresult scuda_graph_mem_attribute_rpc(CUdevice device,
                                              CUgraphMem_attribute attr,
                                              cuuint64_t *value, bool set) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = scuda_rpc_conn_for_device(&device);
  CUresult result = CUDA_ERROR_UNKNOWN;
  int op = set ? SCUDA_RPC_cuDeviceSetGraphMemAttribute
               : SCUDA_RPC_cuDeviceGetGraphMemAttribute;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &device, sizeof(device)) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      (set && rpc_write(conn, value, sizeof(*value)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (!set && rpc_read(conn, value, sizeof(*value)) < 0) ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuDeviceGetGraphMemAttribute(CUdevice device,
                                                 CUgraphMem_attribute attr,
                                                 void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cuuint64_t remote_value = 0;
  CUresult result =
      scuda_graph_mem_attribute_rpc(device, attr, &remote_value, false);
  if (result == CUDA_SUCCESS) {
    memcpy(value, &remote_value, sizeof(remote_value));
  }
  return result;
}

extern "C" CUresult cuDeviceSetGraphMemAttribute(CUdevice device,
                                                 CUgraphMem_attribute attr,
                                                 void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cuuint64_t remote_value = 0;
  memcpy(&remote_value, value, sizeof(remote_value));
  return scuda_graph_mem_attribute_rpc(device, attr, &remote_value, true);
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUDA_MEMCPY3D copy = *pCopy;
  size_t src_host_size = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                             ? scuda_memcpy3d_host_span(copy, true)
                             : 0;
  size_t dst_host_size = copy.dstMemoryType == CU_MEMORYTYPE_HOST
                             ? scuda_memcpy3d_host_span(copy, false)
                             : 0;
  const void *src_host = copy.srcHost;
  void *dst_host = copy.dstHost;
  if ((src_host_size != 0 && src_host == nullptr) ||
      (dst_host_size != 0 && dst_host == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = nullptr;
  if (copy.srcMemoryType == CU_MEMORYTYPE_DEVICE && copy.srcDevice != 0) {
    conn = scuda_rpc_conn_for_deviceptr(copy.srcDevice);
  } else if (copy.dstMemoryType == CU_MEMORYTYPE_DEVICE &&
             copy.dstDevice != 0) {
    conn = scuda_rpc_conn_for_deviceptr(copy.dstDevice);
  } else {
    conn = scuda_rpc_conn_for_current_context();
  }
  CUresult return_value;
  size_t returned_dst_size = 0;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemcpy3D_v2) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &src_host_size, sizeof(src_host_size)) < 0 ||
      (src_host_size != 0 && rpc_write(conn, src_host, src_host_size) < 0) ||
      rpc_write(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size > dst_host_size) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size != 0) {
    scuda_prepare_host_range_write(dst_host, returned_dst_size);
    if (rpc_read(conn, dst_host, returned_dst_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && returned_dst_size != 0) {
    scuda_mark_host_range_clean(dst_host, returned_dst_size);
  }
  return return_value;
}

#ifdef cuMemcpy3D
#undef cuMemcpy3D
#endif
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  return cuMemcpy3D_v2(pCopy);
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount, CUstream hStream) {
  bool dst_is_host = scuda_is_local_address(reinterpret_cast<void *>(dst));
  bool src_is_host = scuda_is_local_address(reinterpret_cast<void *>(src));

  if (dst_is_host && !src_is_host) {
    return cuMemcpyDtoHAsync_v2(reinterpret_cast<void *>(dst), src, ByteCount,
                                hStream);
  }
  if (!dst_is_host && src_is_host) {
    return cuMemcpyHtoDAsync_v2(dst, reinterpret_cast<const void *>(src),
                                ByteCount, hStream);
  }
  if (dst_is_host && src_is_host) {
    memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src),
           ByteCount);
    return CUDA_SUCCESS;
  }

  conn_t *conn = scuda_rpc_conn_for_deviceptr(dst);
  conn_t *src_conn = scuda_rpc_conn_for_deviceptr(src);
  if (conn != src_conn) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemcpyAsync) < 0 ||
      rpc_write(conn, &dst, sizeof(dst)) < 0 ||
      rpc_write(conn, &src, sizeof(src)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuMemcpyAsync_ptsz
#undef cuMemcpyAsync_ptsz
#endif
extern "C" CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                       size_t ByteCount, CUstream hStream) {
  return cuMemcpyAsync(dst, src, ByteCount, hStream);
}

static CUresult
scuda_validate_graph_dependencies(const CUgraphNode *dependencies,
                                  size_t numDependencies) {
  if (numDependencies != 0 && dependencies == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

static CUresult scuda_queue_graph_dependencies(conn_t *conn,
                                               const CUgraphNode *dependencies,
                                               const size_t *numDependencies) {
  if (numDependencies == nullptr ||
      rpc_write(conn, numDependencies, sizeof(*numDependencies)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (*numDependencies == 0) {
    return CUDA_SUCCESS;
  }
  if (rpc_write(conn, dependencies, *numDependencies * sizeof(CUgraphNode)) <
      0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
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
  size_t rows = height * depth;
  return pitch * rows;
}

static CUresult
scuda_pack_kernel_params(const CUDA_KERNEL_NODE_PARAMS *nodeParams,
                         scuda_kernel_param_layout *layout,
                         std::vector<unsigned char> *packed) {
  if (nodeParams == nullptr || layout == nullptr || packed == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (nodeParams->extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUfunction func = nodeParams->func;
#if CUDA_VERSION >= 12000
  if (func == nullptr) {
    func = reinterpret_cast<CUfunction>(nodeParams->kern);
  }
#endif
  CUresult status = scuda_get_kernel_param_layout(func, layout);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (layout->count > 64) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (layout->count != 0 && nodeParams->kernelParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t total_size = 0;
  for (uint32_t i = 0; i < layout->count; ++i) {
    if (nodeParams->kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    total_size = std::max(total_size, layout->offsets[i] + layout->sizes[i]);
  }
  packed->assign(total_size, 0);
  for (uint32_t i = 0; i < layout->count; ++i) {
    memcpy(packed->data() + layout->offsets[i], nodeParams->kernelParams[i],
           layout->sizes[i]);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult
cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                        const CUgraphNode *dependencies, size_t numDependencies,
                        const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  scuda_kernel_param_layout layout;
  std::vector<unsigned char> packed;
  status = scuda_pack_kernel_params(nodeParams, &layout, &packed);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  CUDA_KERNEL_NODE_PARAMS serial_params = *nodeParams;
  serial_params.func = scuda_translate_private_function(serial_params.func);
  serial_params.kernelParams = nullptr;
  serial_params.extra = nullptr;

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  size_t packed_size = packed.size();
  if (rpc_write_start_request(conn, RPC_cuGraphAddKernelNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &serial_params, sizeof(serial_params)) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, &packed_size, sizeof(packed_size)) < 0 ||
      rpc_write(conn, packed.data(), packed.size()) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
  if (phGraphNode == nullptr || copyParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  size_t host_src_bytes = 0;
  if (copyParams->srcMemoryType == CU_MEMORYTYPE_HOST) {
    host_src_bytes = scuda_memcpy3d_host_span_bytes(*copyParams, true);
    if (host_src_bytes != 0 && copyParams->srcHost == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuGraphAddMemcpyNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, copyParams, sizeof(*copyParams)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &host_src_bytes, sizeof(host_src_bytes)) < 0 ||
      (host_src_bytes != 0 &&
       rpc_write(conn, copyParams->srcHost, host_src_bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                     CUcontext ctx) {
  if (phGraphNode == nullptr || memsetParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuGraphAddMemsetNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, memsetParams, sizeof(*memsetParams)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                   const CUgraphNode *dependencies, size_t numDependencies,
                   const CUDA_HOST_NODE_PARAMS *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  add_host_node(reinterpret_cast<void *>(nodeParams->fn), nodeParams->userData);

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuGraphAddHostNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuGraphConditionalHandleCreate(
    CUgraphConditionalHandle *pHandle_out, CUgraph hGraph, CUcontext ctx,
    unsigned int defaultLaunchValue, unsigned int flags) {
  if (pHandle_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuGraphConditionalHandleCreate) <
          0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &defaultLaunchValue, sizeof(defaultLaunchValue)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle_out, sizeof(*pHandle_out)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuGraphAddNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                                      const CUgraphNode *dependencies,
                                      const CUgraphEdgeData *dependencyData,
                                      size_t numDependencies,
                                      CUgraphNodeParams *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  uint32_t param_count = 0;
  size_t packed_size = 0;
  std::vector<unsigned char> packed;
  CUgraphNodeParams serial_params = *nodeParams;
  if (nodeParams->type == CU_GRAPH_NODE_TYPE_KERNEL) {
    scuda_kernel_param_layout layout;
    status = scuda_pack_kernel_params(
        reinterpret_cast<const CUDA_KERNEL_NODE_PARAMS *>(&nodeParams->kernel),
        &layout, &packed);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    param_count = layout.count;
    packed_size = packed.size();
    CUfunction translated =
        serial_params.kernel.func != nullptr
            ? scuda_translate_private_function(serial_params.kernel.func)
            : scuda_translate_private_function(
                  reinterpret_cast<CUfunction>(serial_params.kernel.kern));
    if (serial_params.kernel.func != nullptr) {
      serial_params.kernel.func = translated;
    } else {
      serial_params.kernel.kern = reinterpret_cast<CUkernel>(translated);
    }
    serial_params.kernel.kernelParams = nullptr;
    serial_params.kernel.extra = nullptr;
  } else if (nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    serial_params.conditional.phGraph_out = nullptr;
  } else {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  unsigned int child_count = nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL
                                 ? nodeParams->conditional.size
                                 : 0;
  std::vector<CUgraph> child_graphs(child_count);
  if (rpc_write_start_request(conn, SCUDA_RPC_cuGraphAddNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &serial_params, sizeof(serial_params)) < 0 ||
      rpc_write(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_write(conn, &packed_size, sizeof(packed_size)) < 0 ||
      (packed_size != 0 && rpc_write(conn, packed.data(), packed_size) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      (child_count != 0 && rpc_read(conn, child_graphs.data(),
                                    child_count * sizeof(CUgraph)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && child_count != 0) {
    auto *client_graphs =
        static_cast<CUgraph *>(malloc(child_count * sizeof(CUgraph)));
    if (client_graphs == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    memcpy(client_graphs, child_graphs.data(), child_count * sizeof(CUgraph));
    nodeParams->conditional.phGraph_out = client_graphs;
  }
  return return_value;
}

#ifdef cuGraphAddNode
#undef cuGraphAddNode
#endif
#if CUDA_VERSION >= 13000
extern "C" CUresult cuGraphAddNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                   const CUgraphNode *dependencies,
                                   const CUgraphEdgeData *dependencyData,
                                   size_t numDependencies,
                                   CUgraphNodeParams *nodeParams) {
  return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData,
                           numDependencies, nodeParams);
}
#else
extern "C" CUresult cuGraphAddNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                   const CUgraphNode *dependencies,
                                   size_t numDependencies,
                                   CUgraphNodeParams *nodeParams) {
  return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, nullptr,
                           numDependencies, nodeParams);
}
#endif

extern "C" CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes,
                                    size_t *numNodes) {
  if (numNodes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t requested = nodes == nullptr ? 0 : *numNodes;
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  size_t returned = 0;
  if (rpc_write_start_request(conn, RPC_cuGraphGetNodes) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_write(conn, &requested, sizeof(requested)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (nodes != nullptr && returned != 0 &&
       rpc_read(conn, nodes, returned * sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *numNodes = returned;
  return return_value;
}

#ifdef cuGraphInstantiate
#undef cuGraphInstantiate
#endif
extern "C" CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                       unsigned long long flags) {
  return cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
}

extern "C" CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                     void *userData) {
  if (fn == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  add_host_node(reinterpret_cast<void *>(fn), userData);

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuLaunchHostFunc) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &fn, sizeof(fn)) < 0 ||
      rpc_write(conn, &userData, sizeof(userData)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuStreamAddCallback(CUstream hStream,
                                        CUstreamCallback callback,
                                        void *userData, unsigned int flags) {
  if (callback == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuStreamAddCallback) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &callback, sizeof(callback)) < 0 ||
      rpc_write(conn, &userData, sizeof(userData)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static bool scuda_is_writable_user_pointer(const void *ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return false;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = start + size;
  if (end < start) {
    return false;
  }

  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    uintptr_t region_start = 0;
    uintptr_t region_end = 0;
    char perms[5] = {};
    if (sscanf(line.c_str(), "%lx-%lx %4s", &region_start, &region_end,
               perms) != 3) {
      continue;
    }
    if (start >= region_start && end <= region_end && perms[1] == 'w') {
      return true;
    }
  }
  return false;
}

static CUresult scuda_cuStreamGetCaptureInfo(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  static thread_local std::vector<CUgraphNode> capture_dependencies;
  static thread_local std::vector<CUgraphEdgeData> capture_edge_data;

  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  size_t dependency_count = 0;
  bool has_edge_data = false;
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuStreamGetCaptureInfo_v3) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &id, sizeof(id)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &dependency_count, sizeof(dependency_count)) < 0 ||
      rpc_read(conn, &has_edge_data, sizeof(has_edge_data)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  capture_dependencies.resize(dependency_count);
  if (dependency_count != 0 &&
      rpc_read(conn, capture_dependencies.data(),
               dependency_count * sizeof(CUgraphNode)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  capture_edge_data.clear();
  if (has_edge_data) {
    capture_edge_data.resize(dependency_count);
    if (dependency_count != 0 &&
        rpc_read(conn, capture_edge_data.data(),
                 dependency_count * sizeof(CUgraphEdgeData)) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  bool num_dependencies_writable =
      numDependencies_out != nullptr &&
      scuda_is_writable_user_pointer(numDependencies_out,
                                     sizeof(*numDependencies_out));
  if (edgeData_out != nullptr &&
      (numDependencies_out == nullptr || !num_dependencies_writable)) {
    numDependencies_out = reinterpret_cast<size_t *>(edgeData_out);
    edgeData_out = nullptr;
    num_dependencies_writable = scuda_is_writable_user_pointer(
        numDependencies_out, sizeof(*numDependencies_out));
  }
  if (numDependencies_out != nullptr && !num_dependencies_writable) {
    numDependencies_out = nullptr;
  }
  if (edgeData_out != nullptr &&
      !scuda_is_writable_user_pointer(edgeData_out, sizeof(*edgeData_out))) {
    edgeData_out = nullptr;
  }

  if (captureStatus_out != nullptr) {
    *captureStatus_out = status;
  }
  if (id_out != nullptr) {
    *id_out = id;
  }
  if (graph_out != nullptr) {
    *graph_out = graph;
  }
  if (dependencies_out != nullptr) {
    *dependencies_out =
        capture_dependencies.empty() ? nullptr : capture_dependencies.data();
  }
  if (edgeData_out != nullptr) {
    *edgeData_out =
        capture_edge_data.empty() ? nullptr : capture_edge_data.data();
  }
  if (numDependencies_out != nullptr) {
    *numDependencies_out = dependency_count;
  }
  return return_value;
}

extern "C" CUresult cuStreamGetCaptureInfo_v3(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  return scuda_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                      graph_out, dependencies_out, edgeData_out,
                                      numDependencies_out);
}

extern "C" CUresult cuStreamGetCaptureInfo_v2(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, size_t *numDependencies_out) {
  return scuda_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                      graph_out, dependencies_out, nullptr,
                                      numDependencies_out);
}

#ifdef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo
#endif
#if CUDA_VERSION >= 12000
extern "C" CUresult cuStreamGetCaptureInfo(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  return scuda_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                      graph_out, dependencies_out, edgeData_out,
                                      numDependencies_out);
}
#else
extern "C" CUresult
cuStreamGetCaptureInfo(CUstream stream,
                       CUstreamCaptureStatus *captureStatus_out,
                       cuuint64_t *id_out) {
  return scuda_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                      nullptr, nullptr, nullptr, nullptr);
}
#endif

extern "C" CUresult
cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph,
                            const CUgraphNode *dependencies,
                            const CUgraphEdgeData *dependencyData,
                            size_t numDependencies, CUstreamCaptureMode mode) {
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, SCUDA_RPC_cuStreamBeginCaptureToGraph) <
          0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &mode, sizeof(mode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuStreamUpdateCaptureDependencies_v2(
    CUstream hStream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult status =
      scuda_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(
          conn, SCUDA_RPC_cuStreamUpdateCaptureDependencies_v2) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      scuda_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuStreamUpdateCaptureDependencies
#undef cuStreamUpdateCaptureDependencies
#endif
#if CUDA_VERSION >= 13000
extern "C" CUresult
cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies,
                                  const CUgraphEdgeData *dependencyData,
                                  size_t numDependencies, unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(
      hStream, dependencies, dependencyData, numDependencies, flags);
}
#else
extern "C" CUresult cuStreamUpdateCaptureDependencies(CUstream hStream,
                                                      CUgraphNode *dependencies,
                                                      size_t numDependencies,
                                                      unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, nullptr,
                                              numDependencies, flags);
}
#endif

#if CUDA_VERSION >= 13000
extern "C" CUresult cuStreamUpdateCaptureDependencies_ptsz(
    CUstream hStream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(
      hStream, dependencies, dependencyData, numDependencies, flags);
}
#else
extern "C" CUresult cuStreamUpdateCaptureDependencies_ptsz(
    CUstream hStream, CUgraphNode *dependencies, size_t numDependencies,
    unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, nullptr,
                                              numDependencies, flags);
}
#endif

static bool scuda_uuid_equals(const CUuuid *uuid, const unsigned char *bytes) {
  return uuid != nullptr && memcmp(uuid->bytes, bytes, 16) == 0;
}

extern "C" CUresult scuda_cudart_get_module_from_cubin(CUmodule *module,
                                                       const void *fatbin) {
  CUresult result = cuModuleLoadData(module, fatbin);
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA cudart get_module_from_cubin result="
              << static_cast<int>(result)
              << " module=" << (module == nullptr ? nullptr : *module)
              << std::endl;
  }
  return result;
}

extern "C" CUresult scuda_cudart_get_primary_context(CUcontext *ctx,
                                                     CUdevice dev) {
  CUresult result = cuDevicePrimaryCtxRetain(ctx, dev);
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA cudart get_primary_context dev=" << dev
              << " result=" << static_cast<int>(result)
              << " ctx=" << (ctx == nullptr ? nullptr : *ctx) << std::endl;
  }
  return result;
}

extern "C" CUresult scuda_cudart_get_module_from_cubin_ext1(CUmodule *module,
                                                            const void *fatbin,
                                                            void *arg3,
                                                            void *arg4,
                                                            unsigned int arg5) {
  if (arg3 != nullptr || arg4 != nullptr || arg5 != 0) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = cuModuleLoadData(module, fatbin);
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA cudart get_module_from_cubin_ext1 result="
              << static_cast<int>(result)
              << " module=" << (module == nullptr ? nullptr : *module)
              << std::endl;
  }
  return result;
}

extern "C" void scuda_cudart_noop_size_arg(size_t) {}

extern "C" uintptr_t scuda_dark_return_zero() { return 0; }

extern "C" uintptr_t scuda_dark_return_zero_1(const void *) { return 0; }

extern "C" uintptr_t scuda_dark_return_zero_2(const void *, const void *) {
  return 0;
}

extern "C" CUresult scuda_cudart_get_module_from_cubin_ext2(const void *fatbin,
                                                            CUmodule *module,
                                                            void *arg3,
                                                            void *arg4,
                                                            unsigned int arg5) {
  if (arg3 != nullptr || arg4 != nullptr || arg5 != 0) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = cuModuleLoadData(module, fatbin);
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA cudart get_module_from_cubin_ext2 result="
              << static_cast<int>(result)
              << " module=" << (module == nullptr ? nullptr : *module)
              << std::endl;
  }
  return result;
}

extern "C" CUresult scuda_cudart_load_compilers() { return CUDA_SUCCESS; }

extern "C" void scuda_dark_get_unknown_buffer1(void **ptr, size_t *size) {
  static unsigned char buffer[1024] = {};
  if (ptr != nullptr) {
    *ptr = buffer;
  }
  if (size != nullptr) {
    *size = sizeof(buffer);
  }
}

extern "C" void scuda_dark_get_unknown_buffer2(void **ptr, size_t *size) {
  static unsigned char buffer[14] = {};
  if (ptr != nullptr) {
    *ptr = buffer;
  }
  if (size != nullptr) {
    *size = sizeof(buffer);
  }
}

using scuda_context_storage_dtor_t = void (*)(CUcontext context, void *key,
                                              void *value);

struct scuda_context_storage_value {
  void *value;
  scuda_context_storage_dtor_t dtor;
};

static std::mutex &scuda_context_storage_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<
    CUcontext, std::unordered_map<void *, scuda_context_storage_value>> &
scuda_context_storage() {
  static auto *storage = new std::unordered_map<
      CUcontext, std::unordered_map<void *, scuda_context_storage_value>>();
  return *storage;
}

static CUresult scuda_normalize_context(CUcontext *ctx) {
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (*ctx != nullptr) {
    return CUDA_SUCCESS;
  }
  return cuCtxGetCurrent(ctx);
}

extern "C" CUresult
scuda_context_local_storage_put(CUcontext ctx, void *key, void *value,
                                scuda_context_storage_dtor_t dtor) {
  CUresult result = scuda_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(scuda_context_storage_mutex());
  scuda_context_storage()[ctx][key] = {value, dtor};
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA context storage put ctx=" << ctx << " key=" << key
              << " value=" << value << std::endl;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_context_local_storage_delete(CUcontext ctx,
                                                       void *key) {
  CUresult result = scuda_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(scuda_context_storage_mutex());
  auto ctx_it = scuda_context_storage().find(ctx);
  if (ctx_it == scuda_context_storage().end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  auto value_it = ctx_it->second.find(key);
  if (value_it == ctx_it->second.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  // The private context-local-storage destructor ABI is not stable enough to
  // invoke here; some CUDA libraries call explicit delete during their own
  // teardown and free the value themselves.
  ctx_it->second.erase(value_it);
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA context storage delete ctx=" << ctx << " key=" << key
              << std::endl;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_context_local_storage_get(void **value, CUcontext ctx,
                                                    void *key) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = scuda_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(scuda_context_storage_mutex());
  auto ctx_it = scuda_context_storage().find(ctx);
  if (ctx_it == scuda_context_storage().end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  auto value_it = ctx_it->second.find(key);
  if (value_it == ctx_it->second.end()) {
    if (scuda_trace_enabled()) {
      std::cerr << "SCUDA context storage get missing key ctx=" << ctx
                << " key=" << key << std::endl;
    }
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *value = value_it->second.value;
  if (scuda_trace_enabled()) {
    std::cerr << "SCUDA context storage get ctx=" << ctx << " key=" << key
              << " value=" << *value << std::endl;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_ctx_create_bypass(CUcontext *, unsigned int,
                                            CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult scuda_heap_alloc(const void **, size_t, size_t) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult scuda_heap_free(const void *, size_t *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult scuda_device_get_attribute_ext(CUdevice dev,
                                                   unsigned int attribute, int,
                                                   size_t (*result)[2]) {
  if (result == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int value = 0;
  CUresult status = cuDeviceGetAttribute(
      &value, static_cast<CUdevice_attribute>(attribute), dev);
  if (status == CUDA_SUCCESS) {
    (*result)[0] = static_cast<size_t>(value);
    (*result)[1] = 0;
  }
  return status;
}

extern "C" CUresult scuda_device_get_something(unsigned char *result,
                                               CUdevice) {
  if (result != nullptr) {
    *result = 0;
  }
  return CUDA_SUCCESS;
}

struct scuda_integrity_pass3_input {
  uint32_t driver_version;
  uint32_t version;
  uint32_t current_process;
  uint32_t current_thread;
  const void *cudart_table;
  const void *integrity_check_table;
  const void *fn_address;
  uint64_t unix_seconds;
};

struct scuda_integrity_device_hash_info {
  CUuuid guid;
  int32_t pci_domain;
  int32_t pci_bus;
  int32_t pci_device;
};

static void scuda_integrity_single_pass(unsigned char state[66],
                                        unsigned char byte) {
  static constexpr unsigned char MIXING_TABLE[256] = {
      0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1,
      0xEC, 0xF0, 0x06, 0x13, 0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C,
      0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C, 0x82, 0xCA, 0x1E, 0x9B, 0x57, 0x3C,
      0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A, 0x17, 0xE5, 0x12,
      0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E,
      0xBB, 0x2F, 0xEE, 0x7A, 0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F,
      0x94, 0xC2, 0x10, 0x89, 0x0B, 0x22, 0x5F, 0x21, 0x80, 0x7F, 0x5D, 0x9A,
      0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E, 0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03,
      0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7, 0x5E, 0x92, 0x2A,
      0xAC, 0x56, 0xAA, 0xC6, 0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6,
      0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1, 0x45, 0x9D, 0x70, 0x59,
      0x64, 0x71, 0x87, 0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02,
      0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0, 0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69,
      0x34, 0x40, 0x7E, 0x0F, 0x55, 0x47, 0xA3, 0x23, 0xDD, 0x51, 0xAF, 0x3A,
      0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26, 0x2C, 0x53, 0x0D, 0x6E,
      0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52,
      0x6A, 0xDC, 0x37, 0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08,
      0x0C, 0xBD, 0xB1, 0x4A, 0x78, 0x88, 0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D,
      0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39, 0xF2, 0xEF, 0xB7, 0x0E,
      0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
      0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33,
      0x9F, 0x11, 0x83, 0x14};

  unsigned char index = state[0x40];
  state[index + 0x10] = byte;
  unsigned char next_index = (index + 1) & 0xf;
  state[index + 0x20] = state[index] ^ byte;
  unsigned char mixed = MIXING_TABLE[(byte ^ state[0x41]) & 0xff];
  unsigned char old = state[index + 0x30];
  state[index + 0x30] = mixed ^ old;
  state[0x41] = mixed ^ old;
  state[0x40] = next_index;
  if (next_index != 0) {
    return;
  }

  unsigned char rolling = 0x29;
  unsigned char round = 0;
  while (true) {
    rolling ^= state[0];
    state[0] = rolling;
    for (size_t i = 1; i < 0x30; ++i) {
      rolling = state[i] ^ MIXING_TABLE[rolling];
      state[i] = rolling;
    }
    rolling = static_cast<unsigned char>(rolling + round);
    round = static_cast<unsigned char>(round + 1);
    if (round == 0x12) {
      break;
    }
    rolling = MIXING_TABLE[rolling];
  }
}

static void scuda_integrity_hash_pass(unsigned char state[66], const void *data,
                                      size_t size, unsigned char xor_mask) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < size; ++i) {
    scuda_integrity_single_pass(state, bytes[i] ^ xor_mask);
  }
}

static void scuda_integrity_zero_result(unsigned char state[66]) {
  memset(state, 0, 16);
  memset(state + 48, 0, 18);
}

static void scuda_integrity_pass5(unsigned char state[66], uint64_t result[2]) {
  unsigned char temp = static_cast<unsigned char>(16 - state[64]);
  for (unsigned char i = 0; i < temp; ++i) {
    scuda_integrity_single_pass(state, temp);
  }
  for (size_t i = 0x30; i < 0x40; ++i) {
    scuda_integrity_single_pass(state, state[i]);
  }
  memcpy(&result[0], state, sizeof(uint64_t));
  memcpy(&result[1], state + sizeof(uint64_t), sizeof(uint64_t));
}

static CUresult scuda_get_device_hash_info(
    std::vector<scuda_integrity_device_hash_info> *devices) {
  if (devices == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int count = 0;
  CUresult status = cuDeviceGetCount(&count);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  devices->clear();
  devices->reserve(count);
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = 0;
    status = cuDeviceGet(&device, ordinal);
    if (status != CUDA_SUCCESS) {
      return status;
    }

    scuda_integrity_device_hash_info info = {};
    status = cuDeviceGetUuid_v2(&info.guid, device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGetAttribute(&info.pci_domain,
                                  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGetAttribute(&info.pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                  device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGetAttribute(&info.pci_device,
                                  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    devices->push_back(info);
  }
  return CUDA_SUCCESS;
}

static const void *scuda_get_cudart_export_table();
static const void *scuda_get_integrity_check_table();

extern "C" CUresult scuda_integrity_check(unsigned int version,
                                          uint64_t unix_seconds,
                                          uint64_t (*result)[2]) {
  if (result == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (version % 10 == 0) {
    (*result)[0] = 0x3341181c03cb675cULL;
    (*result)[1] = 0x8ed383aa1f4cd1e8ULL;
    return CUDA_SUCCESS;
  }
  if (version % 10 == 1) {
    (*result)[0] = 0x1841181c03cb675cULL;
    (*result)[1] = 0x8ed383aa1f4cd1e8ULL;
    return CUDA_SUCCESS;
  }

  int driver_version = 0;
  CUresult status = cuDriverGetVersion(&driver_version);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  std::vector<scuda_integrity_device_hash_info> devices;
  status = scuda_get_device_hash_info(&devices);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  static constexpr unsigned char pass1_result[16] = {
      0x14, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52,
      0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A, 0x9F};

  unsigned char state[66] = {};
  scuda_integrity_hash_pass(state, pass1_result, sizeof(pass1_result), 0x36);

  scuda_integrity_pass3_input input = {
      static_cast<uint32_t>(driver_version),
      version,
      static_cast<uint32_t>(getpid()),
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_self())),
      scuda_get_cudart_export_table(),
      scuda_get_integrity_check_table(),
      reinterpret_cast<const void *>(&scuda_integrity_check),
      unix_seconds,
  };
  scuda_integrity_hash_pass(state, &input, sizeof(input), 0);
  for (const auto &device : devices) {
    scuda_integrity_hash_pass(state, &device, sizeof(device), 0);
  }

  uint64_t pass5_result[2] = {};
  scuda_integrity_pass5(state, pass5_result);
  scuda_integrity_zero_result(state);
  scuda_integrity_hash_pass(state, pass1_result, sizeof(pass1_result), 0x5c);
  scuda_integrity_hash_pass(state, pass5_result, sizeof(pass5_result), 0);
  scuda_integrity_pass5(state, *result);
  return CUDA_SUCCESS;
}

extern "C" CUresult scuda_context_check(CUcontext, unsigned int *result1,
                                        const void **result2) {
  if (result1 != nullptr) {
    *result1 = 0;
  }
  if (result2 != nullptr) {
    *result2 = nullptr;
  }
  return CUDA_SUCCESS;
}

extern "C" unsigned int scuda_context_check_fn3() { return 0; }

static const void *scuda_get_cudart_export_table() {
  static const void *table[13] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&scuda_cudart_get_module_from_cubin),
      reinterpret_cast<const void *>(&scuda_cudart_get_primary_context),
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<const void *>(&scuda_cudart_get_module_from_cubin_ext1),
      reinterpret_cast<const void *>(&scuda_cudart_noop_size_arg),
      reinterpret_cast<const void *>(&scuda_cudart_get_module_from_cubin_ext2),
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<const void *>(&scuda_cudart_load_compilers),
  };
  return table;
}

template <size_t N>
static const void *scuda_table_start(const void *(&table)[N]) {
  return table;
}

static const void *scuda_get_tools_tls_table() {
  static const void *table[4] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&scuda_dark_return_zero),
      reinterpret_cast<const void *>(&scuda_dark_return_zero_1),
      reinterpret_cast<const void *>(&scuda_dark_return_zero_1)};
  return scuda_table_start(table);
}

static const void *scuda_get_tools_runtime_callback_hooks_table() {
  static const void *table[7] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&scuda_dark_return_zero_2),
      reinterpret_cast<const void *>(&scuda_dark_get_unknown_buffer1),
      reinterpret_cast<const void *>(&scuda_dark_return_zero_2),
      reinterpret_cast<const void *>(&scuda_dark_return_zero_2),
      reinterpret_cast<const void *>(&scuda_dark_return_zero_2),
      reinterpret_cast<const void *>(&scuda_dark_get_unknown_buffer2),
  };
  return scuda_table_start(table);
}

static const void *scuda_get_context_local_storage_table() {
  static const void *table[4] = {
      reinterpret_cast<const void *>(&scuda_context_local_storage_put),
      reinterpret_cast<const void *>(&scuda_context_local_storage_delete),
      reinterpret_cast<const void *>(&scuda_context_local_storage_get),
      nullptr,
  };
  return scuda_table_start(table);
}

static const void *scuda_get_ctx_create_bypass_table() {
  static const void *table[2] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&scuda_ctx_create_bypass)};
  return scuda_table_start(table);
}

static const void *scuda_get_heap_access_table() {
  static const void *table[3] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&scuda_heap_alloc),
      reinterpret_cast<const void *>(&scuda_heap_free),
  };
  return scuda_table_start(table);
}

static const void *scuda_get_device_extended_rt_table() {
  static const void *table[26] = {};
  static std::once_flag once;
  std::call_once(once, [] {
    const_cast<void **>(table)[0] = reinterpret_cast<void *>(sizeof(table));
    const_cast<void **>(table)[5] =
        reinterpret_cast<void *>(&scuda_device_get_attribute_ext);
    const_cast<void **>(table)[13] =
        reinterpret_cast<void *>(&scuda_device_get_something);
  });
  return scuda_table_start(table);
}

static const void *scuda_get_integrity_check_table() {
  static const void *table[3] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&scuda_integrity_check),
      nullptr,
  };
  return scuda_table_start(table);
}

static const void *scuda_get_context_checks_table() {
  static const void *table[4] = {
      reinterpret_cast<const void *>(sizeof(table)),
      nullptr,
      reinterpret_cast<const void *>(&scuda_context_check),
      reinterpret_cast<const void *>(&scuda_context_check_fn3),
  };
  return scuda_table_start(table);
}

extern "C" CUresult cuGetExportTable(const void **ppExportTable,
                                     const CUuuid *pExportTableId) {
  static constexpr unsigned char CUDART_INTERFACE_UUID[16] = {
      0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
      0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};
  static constexpr unsigned char TOOLS_TLS_UUID[16] = {
      0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
      0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc};
  static constexpr unsigned char TOOLS_RUNTIME_CALLBACK_HOOKS_UUID[16] = {
      0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
      0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66};
  static constexpr unsigned char CONTEXT_LOCAL_STORAGE_UUID[16] = {
      0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
      0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93};
  static constexpr unsigned char CTX_CREATE_BYPASS_UUID[16] = {
      0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a,
      0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86};
  static constexpr unsigned char HEAP_ACCESS_UUID[16] = {
      0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a,
      0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae};
  static constexpr unsigned char DEVICE_EXTENDED_RT_UUID[16] = {
      0xb1, 0x05, 0x41, 0xe1, 0xf7, 0xc7, 0xc7, 0x4a,
      0x9f, 0x64, 0xf2, 0x23, 0xbe, 0x99, 0xf1, 0xe2};
  static constexpr unsigned char INTEGRITY_CHECK_UUID[16] = {
      0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
      0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2};
  static constexpr unsigned char CONTEXT_CHECKS_UUID[16] = {
      0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
      0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e};

  if (scuda_trace_enabled() && pExportTableId != nullptr) {
    std::cerr << "SCUDA cuGetExportTable requested UUID:";
    for (unsigned char byte : pExportTableId->bytes) {
      fprintf(stderr, " %02x", byte);
    }
    std::cerr << std::endl;
  }

  if (ppExportTable != nullptr) {
    *ppExportTable = nullptr;
  }
  if (ppExportTable == nullptr || pExportTableId == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (scuda_uuid_equals(pExportTableId, CUDART_INTERFACE_UUID)) {
    *ppExportTable = scuda_get_cudart_export_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, TOOLS_TLS_UUID)) {
    *ppExportTable = scuda_get_tools_tls_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, TOOLS_RUNTIME_CALLBACK_HOOKS_UUID)) {
    *ppExportTable = scuda_get_tools_runtime_callback_hooks_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, CONTEXT_LOCAL_STORAGE_UUID)) {
    *ppExportTable = scuda_get_context_local_storage_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, CTX_CREATE_BYPASS_UUID)) {
    *ppExportTable = scuda_get_ctx_create_bypass_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, HEAP_ACCESS_UUID)) {
    *ppExportTable = scuda_get_heap_access_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, DEVICE_EXTENDED_RT_UUID)) {
    *ppExportTable = scuda_get_device_extended_rt_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, INTEGRITY_CHECK_UUID)) {
    *ppExportTable = scuda_get_integrity_check_table();
    return CUDA_SUCCESS;
  }
  if (scuda_uuid_equals(pExportTableId, CONTEXT_CHECKS_UUID)) {
    *ppExportTable = scuda_get_context_checks_table();
    return CUDA_SUCCESS;
  }
  {
    const void *private_table =
        scuda_remote_private_export_table(pExportTableId);
    if (private_table != nullptr) {
      *ppExportTable = private_table;
      return CUDA_SUCCESS;
    }
  }
  if (scuda_stub_private_exports_enabled()) {
    const void *private_table =
        scuda_private_export_table_from_env(pExportTableId);
    if (private_table != nullptr) {
      *ppExportTable = private_table;
      return CUDA_SUCCESS;
    }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

static void *scuda_make_missing_stub(const char *symbol) {
  if (symbol == nullptr) {
    return nullptr;
  }

#if defined(__x86_64__)
  static std::unordered_map<std::string, void *> stubs;
  auto existing = stubs.find(symbol);
  if (existing != stubs.end()) {
    return existing->second;
  }

  constexpr size_t stub_size = 22;
  unsigned char *code = static_cast<unsigned char *>(
      mmap(nullptr, stub_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&scuda_unsupported_driver_api);
  }

  char *stable_symbol = strdup(symbol);
  void *handler = reinterpret_cast<void *>(&scuda_missing_driver_api_called);
  code[0] = 0x48;
  code[1] = 0xbf;
  memcpy(code + 2, &stable_symbol, sizeof(stable_symbol));
  code[10] = 0x48;
  code[11] = 0xb8;
  memcpy(code + 12, &handler, sizeof(handler));
  code[20] = 0xff;
  code[21] = 0xe0;
  stubs[symbol] = code;
  return code;
#else
  return reinterpret_cast<void *>(&scuda_unsupported_driver_api);
#endif
}

#define SCUDA_DEFINE_UNSUPPORTED_STUB(name)                                    \
  extern "C" CUresult scuda_unsupported_##name() {                             \
    std::cerr << "SCUDA unsupported Driver API called: " #name << std::endl;   \
    return CUDA_ERROR_NOT_SUPPORTED;                                           \
  }

SCUDA_DEFINE_UNSUPPORTED_STUB(cuCtxCreate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuModuleLoadData)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuModuleLoadFatBinary)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLibraryLoadData)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLibraryGetKernelCount)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLibraryEnumerateKernels)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuKernelGetName)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuKernelGetParamInfo)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLinkCreate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLinkAddData)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLinkAddFile)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemGetInfo)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemGetAddressRange)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemHostGetDevicePointer)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemHostRegister)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemHostUnregister)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuPointerGetAttribute)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpyDtoH)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpyDtoHAsync)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpy2DUnaligned)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpy2DAsync)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpy3D)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DAsync)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DPeer)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DPeerAsync)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemAdvise)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemPrefetchAsync)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuMemRangeGetAttribute)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGetErrorString)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGetErrorName)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGraphInstantiate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuUserObjectCreate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuStreamBeginCaptureToGraph)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuStreamGetCaptureInfo)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuStreamUpdateCaptureDependencies)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGraphExecKernelNodeSetParams)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGraphAddNode)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGraphNodeSetParams)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGraphExecNodeSetParams)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGraphConditionalHandleCreate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDeviceRegisterAsyncNotification)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDeviceUnregisterAsyncNotification)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLogsRegisterCallback)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLogsUnregisterCallback)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLogsCurrent)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLogsDumpToFile)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuLogsDumpToMemory)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDeviceGetDevResource)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDevSmResourceSplitByCount)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDevResourceGenerateDesc)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuCtxFromGreenCtx)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxCreate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxDestroy)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuCtxGetDevResource)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxGetDevResource)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxGetId)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxStreamCreate)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuStreamGetDevResource)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuCtxRecordEvent)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuCtxWaitEvent)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxRecordEvent)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuGreenCtxWaitEvent)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDevSmResourceSplit)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuKernelGetLibrary)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDeviceGetHostAtomicCapabilities)
SCUDA_DEFINE_UNSUPPORTED_STUB(cuDeviceGetP2PAtomicCapabilities)

static void *scuda_get_unsupported_stub(const char *symbol) {
  static const std::unordered_map<std::string, void *> stubs = {
#define SCUDA_STUB_ENTRY(name)                                                 \
  { #name, (void *)&scuda_unsupported_##name }
      SCUDA_STUB_ENTRY(cuCtxCreate),
      SCUDA_STUB_ENTRY(cuModuleLoadData),
      SCUDA_STUB_ENTRY(cuModuleLoadFatBinary),
      SCUDA_STUB_ENTRY(cuLibraryLoadData),
      SCUDA_STUB_ENTRY(cuLibraryGetKernelCount),
      SCUDA_STUB_ENTRY(cuLibraryEnumerateKernels),
      SCUDA_STUB_ENTRY(cuKernelGetName),
      SCUDA_STUB_ENTRY(cuKernelGetParamInfo),
      SCUDA_STUB_ENTRY(cuLinkCreate),
      SCUDA_STUB_ENTRY(cuLinkAddData),
      SCUDA_STUB_ENTRY(cuLinkAddFile),
      SCUDA_STUB_ENTRY(cuMemGetInfo),
      SCUDA_STUB_ENTRY(cuMemGetAddressRange),
      SCUDA_STUB_ENTRY(cuMemHostGetDevicePointer),
      SCUDA_STUB_ENTRY(cuMemHostRegister),
      SCUDA_STUB_ENTRY(cuMemHostUnregister),
      SCUDA_STUB_ENTRY(cuPointerGetAttribute),
      SCUDA_STUB_ENTRY(cuMemcpyDtoH),
      SCUDA_STUB_ENTRY(cuMemcpyDtoHAsync),
      SCUDA_STUB_ENTRY(cuMemcpy2DUnaligned),
      SCUDA_STUB_ENTRY(cuMemcpy2DAsync),
      SCUDA_STUB_ENTRY(cuMemcpy3D),
      SCUDA_STUB_ENTRY(cuMemcpy3DAsync),
      SCUDA_STUB_ENTRY(cuMemcpy3DPeer),
      SCUDA_STUB_ENTRY(cuMemcpy3DPeerAsync),
      SCUDA_STUB_ENTRY(cuMemAdvise),
      SCUDA_STUB_ENTRY(cuMemPrefetchAsync),
      SCUDA_STUB_ENTRY(cuMemRangeGetAttribute),
      SCUDA_STUB_ENTRY(cuGetErrorString),
      SCUDA_STUB_ENTRY(cuGetErrorName),
      SCUDA_STUB_ENTRY(cuGraphInstantiate),
      SCUDA_STUB_ENTRY(cuUserObjectCreate),
      SCUDA_STUB_ENTRY(cuStreamBeginCaptureToGraph),
      SCUDA_STUB_ENTRY(cuStreamGetCaptureInfo),
      SCUDA_STUB_ENTRY(cuStreamUpdateCaptureDependencies),
      SCUDA_STUB_ENTRY(cuGraphExecKernelNodeSetParams),
      SCUDA_STUB_ENTRY(cuGraphAddNode),
      SCUDA_STUB_ENTRY(cuGraphNodeSetParams),
      SCUDA_STUB_ENTRY(cuGraphExecNodeSetParams),
      SCUDA_STUB_ENTRY(cuGraphConditionalHandleCreate),
      SCUDA_STUB_ENTRY(cuDeviceRegisterAsyncNotification),
      SCUDA_STUB_ENTRY(cuDeviceUnregisterAsyncNotification),
      SCUDA_STUB_ENTRY(cuLogsRegisterCallback),
      SCUDA_STUB_ENTRY(cuLogsUnregisterCallback),
      SCUDA_STUB_ENTRY(cuLogsCurrent),
      SCUDA_STUB_ENTRY(cuLogsDumpToFile),
      SCUDA_STUB_ENTRY(cuLogsDumpToMemory),
      SCUDA_STUB_ENTRY(cuDeviceGetDevResource),
      SCUDA_STUB_ENTRY(cuDevSmResourceSplitByCount),
      SCUDA_STUB_ENTRY(cuDevResourceGenerateDesc),
      SCUDA_STUB_ENTRY(cuCtxFromGreenCtx),
      SCUDA_STUB_ENTRY(cuGreenCtxCreate),
      SCUDA_STUB_ENTRY(cuGreenCtxDestroy),
      SCUDA_STUB_ENTRY(cuCtxGetDevResource),
      SCUDA_STUB_ENTRY(cuGreenCtxGetDevResource),
      SCUDA_STUB_ENTRY(cuGreenCtxGetId),
      SCUDA_STUB_ENTRY(cuGreenCtxStreamCreate),
      SCUDA_STUB_ENTRY(cuStreamGetDevResource),
      SCUDA_STUB_ENTRY(cuCtxRecordEvent),
      SCUDA_STUB_ENTRY(cuCtxWaitEvent),
      SCUDA_STUB_ENTRY(cuGreenCtxRecordEvent),
      SCUDA_STUB_ENTRY(cuGreenCtxWaitEvent),
      SCUDA_STUB_ENTRY(cuDevSmResourceSplit),
      SCUDA_STUB_ENTRY(cuKernelGetLibrary),
      SCUDA_STUB_ENTRY(cuDeviceGetHostAtomicCapabilities),
      SCUDA_STUB_ENTRY(cuDeviceGetP2PAtomicCapabilities),
#undef SCUDA_STUB_ENTRY
  };
  if (symbol == nullptr) {
    return nullptr;
  }
  auto it = stubs.find(symbol);
  return it == stubs.end() ? nullptr : it->second;
}

void rpc_close(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }

  if (!conn->closed) {
    conn->closed = 1;
    shutdown(conn->connfd, SHUT_RDWR);
    close(conn->connfd);
  }

  pthread_mutex_lock(&conn->read_mutex);
  pthread_cond_broadcast(&conn->read_cond);
  pthread_mutex_unlock(&conn->read_mutex);
}

static void scuda_rpc_shutdown() {
  if (pthread_mutex_lock(&conn_mutex) < 0) {
    return;
  }
  if (scuda_rpc_shutting_down) {
    pthread_mutex_unlock(&conn_mutex);
    return;
  }
  scuda_rpc_shutting_down = true;
  int count = nconns;
  for (int i = 0; i < count; ++i) {
    rpc_close(&conns[i]);
  }
  pthread_mutex_unlock(&conn_mutex);

  for (int i = 0; i < count; ++i) {
    if (conns[i].read_thread != 0) {
      pthread_join(conns[i].read_thread, nullptr);
      conns[i].read_thread = 0;
    }
    if (conns[i].rpc_thread != 0) {
      pthread_join(conns[i].rpc_thread, nullptr);
      conns[i].rpc_thread = 0;
    }
  }

  if (pthread_mutex_lock(&conn_mutex) == 0) {
    nconns = 0;
    scuda_rpc_shutting_down = false;
    pthread_mutex_unlock(&conn_mutex);
  }
}

__attribute__((destructor)) static void scuda_rpc_destructor() {
  scuda_rpc_shutdown();
}

typedef void (*func_t)(void *);

void add_host_node(void *fn, void *udata) { host_funcs[fn] = udata; }

void invoke_host_func(void *fn) {
  for (const auto &pair : host_funcs) {
    if (pair.first == fn) {
      func_t func = reinterpret_cast<func_t>(pair.first);
      std::cout << "Invoking function at: " << pair.first << std::endl;
      func(pair.second);
      return;
    }
  }
}

void *rpc_client_dispatch_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  int op;

  while (true) {
    op = rpc_dispatch(conn, 1);

    if (op == 1) {
      std::cout << "Transferring memory..." << std::endl;

      int found = 0;

      rpc_read(conn, &found, sizeof(int));

      for (int i = 0; i < found; ++i) {
        void *host_data = nullptr;
        void *dst = nullptr;
        size_t count = 0;

        if (rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0) {
          std::cerr << "Failed to read transfer parameters." << std::endl;
          break;
        }

        host_data = malloc(count);
        if (!host_data) {
          std::cerr << "Memory allocation failed." << std::endl;
          break;
        }

        // Read the actual data from the server (sent from `src` in device
        // memory)
        if (rpc_read(conn, host_data, count) < 0) {
          std::cerr << "Failed to read device data from server." << std::endl;
          free(host_data);
          break;
        }

        // Copy received data to the destination (dst) on the host
        scuda_prepare_host_range_write(dst, count);
        memcpy(dst, host_data, count);
        scuda_mark_host_range_clean(dst, count);
        free(host_data);
      }

      void *temp_mem;
      if (rpc_read(conn, &temp_mem, sizeof(void *)) <= 0) {
        std::cerr << "rpc_read failed for mem. Closing connection."
                  << std::endl;
        break;
      }

      int request_id = rpc_read_end(conn);
      void *mem = temp_mem;

      if (mem == nullptr) {
        std::cerr << "Invalid function pointer!" << std::endl;
        continue;
      }

      invoke_host_func(mem);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        std::cerr << "rpc_write failed. Closing connection." << std::endl;
        break;
      }
    } else if (op == 2) {
      CUstream stream = nullptr;
      CUresult status = CUDA_ERROR_UNKNOWN;
      CUstreamCallback callback = nullptr;
      void *user_data = nullptr;
      if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
          rpc_read(conn, &status, sizeof(status)) < 0 ||
          rpc_read(conn, &callback, sizeof(callback)) < 0 ||
          rpc_read(conn, &user_data, sizeof(user_data)) < 0) {
        std::cerr << "Failed to read stream callback request." << std::endl;
        break;
      }

      int request_id = rpc_read_end(conn);
      if (request_id < 0) {
        break;
      }

      callback(stream, status, user_data);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        std::cerr << "rpc_write failed. Closing connection." << std::endl;
        break;
      }
    } else if (op < 0 || conn->closed) {
      break;
    }
  }

  if (!conn->closed) {
    std::cerr << "Exiting dispatch thread due to an error." << std::endl;
  }
  return nullptr;
}

int rpc_open() {
  if (pthread_mutex_lock(&conn_mutex) < 0)
    return -1;

  if (nconns > 0) {
    if (pthread_mutex_unlock(&conn_mutex) < 0)
      return -1;
    return 0;
  }

  std::cout << "Opening connection to server" << std::endl;

  char *server_ips = getenv("SCUDA_SERVER");
  if (server_ips == NULL) {
    printf("SCUDA_SERVER environment variable not set\n");
    std::exit(1);
  }

  char *server_ip = strdup(server_ips);
  char *token;
  while ((token = strsep(&server_ip, ","))) {
    char *host;
    char *port;

    // Split the string into IP address and port
    char *colon = strchr(token, ':');
    if (colon == NULL) {
      host = token;
      port = const_cast<char *>(DEFAULT_PORT);
    } else {
      *colon = '\0';
      host = token;
      port = colon + 1;
    }

    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
      std::cout << "getaddrinfo of " << host << " port " << port << " failed"
                << std::endl;
      continue;
    }

    int flag = 1;
    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1) {
      printf("socket creation failed...\n");
      exit(1);
    }

    int opts = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag,
                          sizeof(int));
    if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0) {
      std::cerr << "Connecting to " << host << " port " << port
                << " failed: " << strerror(errno) << std::endl;
      exit(1);
    }

    conns[nconns] = {};
    conns[nconns].connfd = sockfd;
    conns[nconns].request_id = 0;
    conns[nconns].closed = 0;
    conns[nconns].local_request_parity = conns[nconns].request_id & 1;
    if (pthread_mutex_init(&conns[nconns].read_mutex, NULL) < 0 ||
        pthread_mutex_init(&conns[nconns].write_mutex, NULL) < 0 ||
        pthread_mutex_init(&conns[nconns].call_mutex, NULL) < 0 ||
        pthread_cond_init(&conns[nconns].read_cond, NULL) < 0) {
      return -1;
    }

    pthread_create(&conns[nconns].read_thread, NULL, rpc_client_dispatch_thread,
                   (void *)&conns[nconns]);

    nconns++;
  }

  if (pthread_mutex_unlock(&conn_mutex) < 0)
    return -1;
  if (nconns == 0)
    return -1;
  return 0;
}

conn_t *rpc_client_get_connection(unsigned int index) {
  if (rpc_open() < 0)
    return nullptr;
  return &conns[index];
}

int rpc_size() { return nconns; }

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                     int cudaVersion, cuuint64_t flags);

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  if (scuda_trace_enabled()) {
    std::cout << "cuGetProcAddress getting symbol: " << symbol << std::endl;
  }

  if (symbol != nullptr && strcmp(symbol, "cuFuncGetAttribute") == 0) {
    *pfn = reinterpret_cast<void *>(&scuda_cuFuncGetAttribute_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr && strcmp(symbol, "cuPointerGetAttributes") == 0) {
    *pfn = reinterpret_cast<void *>(&scuda_cuPointerGetAttributes_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr &&
      strcmp(symbol, "cuOccupancyMaxActiveBlocksPerMultiprocessor") == 0) {
    *pfn = reinterpret_cast<void *>(
        &scuda_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr &&
      strcmp(symbol, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") ==
          0) {
    *pfn = reinterpret_cast<void *>(
        &scuda_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  auto it = get_function_pointer(symbol);
  if (it != nullptr) {
    *pfn = it;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    if (scuda_trace_enabled()) {
      std::cout << "cuGetProcAddress: Mapped symbol '" << symbol
                << "' to function: " << *pfn << std::endl;
    }
    return CUDA_SUCCESS;
  }

  static const std::unordered_map<std::string, void *> manual_function_map = {
#if CUDA_VERSION >= 12050
      {"cuCtxCreate", (void *)cuCtxCreate_v4},
      {"cuCtxCreate_v4", (void *)cuCtxCreate_v4},
#else
      {"cuCtxCreate", (void *)cuCtxCreate_v2},
#endif
      {"cuCtxCreate_v2", (void *)cuCtxCreate_v2},
      {"cuOccupancyMaxPotentialBlockSize",
       (void *)cuOccupancyMaxPotentialBlockSize},
      {"cuOccupancyMaxPotentialBlockSizeWithFlags",
       (void *)cuOccupancyMaxPotentialBlockSizeWithFlags},
      {"cuProfilerInitialize", (void *)cuProfilerInitialize},
      {"cuProfilerStart", (void *)cuProfilerStart},
      {"cuProfilerStop", (void *)cuProfilerStop},
      {"cuStreamDestroy", (void *)cuStreamDestroy},
      {"cuEventDestroy", (void *)cuEventDestroy},
      {"cuEventElapsedTime", (void *)cuEventElapsedTime},
      {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
      {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
      {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
      {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
      {"cuMemFreeHost", (void *)cuMemFreeHost},
      {"cuMemHostAlloc", (void *)cuMemHostAlloc},
      {"cuMemHostGetDevicePointer", (void *)cuMemHostGetDevicePointer_v2},
      {"cuMemHostGetDevicePointer_v2", (void *)cuMemHostGetDevicePointer_v2},
      {"cuMemHostGetFlags", (void *)cuMemHostGetFlags},
      {"cuMemHostRegister", (void *)cuMemHostRegister},
      {"cuMemHostRegister_v2", (void *)cuMemHostRegister_v2},
      {"cuMemHostUnregister", (void *)cuMemHostUnregister},
      {"cuMemGetAddressRange", (void *)cuMemGetAddressRange},
      {"cuMemGetInfo", (void *)cuMemGetInfo},
      {"cuArrayCreate", (void *)cuArrayCreate_v2},
      {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
      {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
      {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
      {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
      {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
      {"cuMemcpyDtoA", (void *)cuMemcpyDtoA_v2},
      {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
      {"cuMemcpyAtoD", (void *)cuMemcpyAtoD_v2},
      {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
      {"cuMemcpyAtoH", (void *)cuMemcpyAtoH_v2},
      {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
      {"cuMemcpyAtoA", (void *)cuMemcpyAtoA_v2},
      {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
      {"cuMemcpy2D", (void *)cuMemcpy2D_v2},
      {"cuMemcpy2D_v2", (void *)cuMemcpy2D_v2},
      {"cuMemcpy2DUnaligned", (void *)cuMemcpy2DUnaligned_v2},
      {"cuMemcpy2DUnaligned_v2", (void *)cuMemcpy2DUnaligned_v2},
      {"cuMemcpy2DAsync", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy2DAsync_v2", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy2DAsync_ptsz", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy3D", (void *)cuMemcpy3D_v2},
      {"cuMemcpy3D_v2", (void *)cuMemcpy3D_v2},
      {"cuPointerGetAttribute", (void *)cuPointerGetAttribute},
      {"cuPointerGetAttributes", (void *)scuda_cuPointerGetAttributes_safe},
      {"cuGetExportTable", (void *)cuGetExportTable},
      {"cuModuleLoadData", (void *)cuModuleLoadData},
      {"cuModuleLoadDataEx", (void *)cuModuleLoadDataEx},
      {"cuLibraryLoadData", (void *)cuLibraryLoadData},
      {"cuLaunchKernel", (void *)cuLaunchKernel},
      {"cuLaunchKernelEx", (void *)cuLaunchKernelEx},
      {"cuMemcpyAsync", (void *)cuMemcpyAsync},
      {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
      {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
      {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
      {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
      {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
      {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
      {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
      {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
      {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo},
      {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
      {"cuStreamGetCaptureInfo_v3", (void *)cuStreamGetCaptureInfo_v3},
      {"cuCtxSynchronize", (void *)cuCtxSynchronize},
      {"cuStreamSynchronize", (void *)cuStreamSynchronize},
      {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize_ptsz},
      {"cuEventSynchronize", (void *)cuEventSynchronize},
      {"cuGetErrorName", (void *)cuGetErrorName},
      {"cuGetErrorString", (void *)cuGetErrorString},
      {"cuGraphAddKernelNode", (void *)cuGraphAddKernelNode_v2},
      {"cuGraphAddKernelNode_v2", (void *)cuGraphAddKernelNode_v2},
      {"cuGraphAddNode", (void *)cuGraphAddNode},
      {"cuGraphAddNode_v2", (void *)cuGraphAddNode_v2},
      {"cuGraphConditionalHandleCreate",
       (void *)cuGraphConditionalHandleCreate},
      {"cuGraphAddMemcpyNode", (void *)cuGraphAddMemcpyNode},
      {"cuGraphAddMemsetNode", (void *)cuGraphAddMemsetNode},
      {"cuGraphAddHostNode", (void *)cuGraphAddHostNode},
      {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
      {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
      {"cuDeviceGetGraphMemAttribute", (void *)cuDeviceGetGraphMemAttribute},
      {"cuDeviceSetGraphMemAttribute", (void *)cuDeviceSetGraphMemAttribute},
      {"cuGraphKernelNodeGetParams", (void *)cuGraphKernelNodeGetParams_v2},
      {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
      {"cuGraphExecKernelNodeSetParams",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphExecKernelNodeSetParams_v2",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphGetNodes", (void *)cuGraphGetNodes},
      {"cuGraphInstantiate", (void *)cuGraphInstantiate},
      {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
      {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
      {"cuStreamAddCallback", (void *)cuStreamAddCallback},
      {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
      {"cuStreamBeginCaptureToGraph", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamBeginCaptureToGraph_ptsz", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamUpdateCaptureDependencies",
       (void *)cuStreamUpdateCaptureDependencies},
      {"cuStreamUpdateCaptureDependencies_v2",
       (void *)cuStreamUpdateCaptureDependencies_v2},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       (void *)cuStreamUpdateCaptureDependencies_ptsz},
  };
  auto manual_it = manual_function_map.find(symbol);
  if (manual_it != manual_function_map.end()) {
    *pfn = manual_it->second;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  void *unsupported_stub = scuda_get_unsupported_stub(symbol);
  if (unsupported_stub != nullptr) {
    *pfn = unsupported_stub;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
    *pfn = (void *)&cuGetProcAddress_v2;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (strcmp(symbol, "cuGetProcAddress") == 0) {
    *pfn = (void *)&cuGetProcAddress;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  if (scuda_trace_enabled()) {
    std::cout << "cuGetProcAddress: Symbol '" << symbol
              << "' not found in cudaFunctionMap." << std::endl;
  }

  // fall back to dlsym
  static void *(*real_dlsym)(void *, const char *) = NULL;
  if (real_dlsym == NULL) {
    real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                         "GLIBC_2.2.5");
  }

  *pfn = real_dlsym(RTLD_DEFAULT, symbol);
  if (*pfn != nullptr) {
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!libCudaHandle) {
    if (scuda_stub_missing_enabled() &&
        scuda_symbol_looks_like_driver_api(symbol)) {
      *pfn = scuda_make_missing_stub(symbol);
      if (symbolStatus != nullptr) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
      }
      return CUDA_SUCCESS;
    }
    *pfn = nullptr;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return CUDA_SUCCESS;
  }

  *pfn = real_dlsym(libCudaHandle, symbol);
  if (!(*pfn)) {
    if (scuda_trace_enabled()) {
      std::cerr << "Error: Could not resolve symbol '" << symbol
                << "' using dlsym." << std::endl;
    }
    if (scuda_stub_missing_enabled() &&
        scuda_symbol_looks_like_driver_api(symbol)) {
      *pfn = scuda_make_missing_stub(symbol);
      if (symbolStatus != nullptr) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
      }
      return CUDA_SUCCESS;
    }
    *pfn = nullptr;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return CUDA_SUCCESS;
  }

  if (symbolStatus != nullptr) {
    *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                     int cudaVersion, cuuint64_t flags) {
  return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}

void *dlsym(void *handle, const char *name) __THROW {
  if (scuda_trace_enabled()) {
    std::cout << "dlsym: " << name << std::endl;
  }

  if (!scuda_symbol_looks_like_driver_api(name)) {
    return scuda_real_dlsym(handle, name);
  }

  if (name != nullptr && strcmp(name, "cuFuncGetAttribute") == 0) {
    return reinterpret_cast<void *>(&scuda_cuFuncGetAttribute_safe);
  }
  if (name != nullptr && strcmp(name, "cuPointerGetAttributes") == 0) {
    return reinterpret_cast<void *>(&scuda_cuPointerGetAttributes_safe);
  }
  if (name != nullptr &&
      strcmp(name, "cuOccupancyMaxActiveBlocksPerMultiprocessor") == 0) {
    return reinterpret_cast<void *>(
        &scuda_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe);
  }
  if (name != nullptr &&
      strcmp(name, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") ==
          0) {
    return reinterpret_cast<void *>(
        &scuda_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_safe);
  }

  void *func = get_function_pointer(name);

  /** proc address function calls are basically dlsym; we should handle this
   * differently at the top level. */
  if (strcmp(name, "cuGetProcAddress_v2") == 0) {
    return (void *)&cuGetProcAddress_v2;
  }
  if (strcmp(name, "cuGetProcAddress") == 0) {
    return (void *)&cuGetProcAddress;
  }

  if (func != nullptr) {
    // std::cout << "[dlsym] Function address from cudaFunctionMap: " << func
    // << " " << name << std::endl;
    return func;
  }

  static const std::unordered_map<std::string, void *> manual_function_map = {
#if CUDA_VERSION >= 12050
      {"cuCtxCreate", (void *)cuCtxCreate_v4},
      {"cuCtxCreate_v4", (void *)cuCtxCreate_v4},
#else
      {"cuCtxCreate", (void *)cuCtxCreate_v2},
#endif
      {"cuCtxCreate_v2", (void *)cuCtxCreate_v2},
      {"cuOccupancyMaxPotentialBlockSize",
       (void *)cuOccupancyMaxPotentialBlockSize},
      {"cuOccupancyMaxPotentialBlockSizeWithFlags",
       (void *)cuOccupancyMaxPotentialBlockSizeWithFlags},
      {"cuProfilerInitialize", (void *)cuProfilerInitialize},
      {"cuProfilerStart", (void *)cuProfilerStart},
      {"cuProfilerStop", (void *)cuProfilerStop},
      {"cuStreamDestroy", (void *)cuStreamDestroy},
      {"cuEventDestroy", (void *)cuEventDestroy},
      {"cuEventElapsedTime", (void *)cuEventElapsedTime},
      {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
      {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
      {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
      {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
      {"cuMemFreeHost", (void *)cuMemFreeHost},
      {"cuMemHostAlloc", (void *)cuMemHostAlloc},
      {"cuMemHostGetDevicePointer", (void *)cuMemHostGetDevicePointer_v2},
      {"cuMemHostGetDevicePointer_v2", (void *)cuMemHostGetDevicePointer_v2},
      {"cuMemHostGetFlags", (void *)cuMemHostGetFlags},
      {"cuMemHostRegister", (void *)cuMemHostRegister},
      {"cuMemHostRegister_v2", (void *)cuMemHostRegister_v2},
      {"cuMemHostUnregister", (void *)cuMemHostUnregister},
      {"cuMemGetAddressRange", (void *)cuMemGetAddressRange},
      {"cuMemGetInfo", (void *)cuMemGetInfo},
      {"cuArrayCreate", (void *)cuArrayCreate_v2},
      {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
      {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
      {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
      {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
      {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
      {"cuMemcpyDtoA", (void *)cuMemcpyDtoA_v2},
      {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
      {"cuMemcpyAtoD", (void *)cuMemcpyAtoD_v2},
      {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
      {"cuMemcpyAtoH", (void *)cuMemcpyAtoH_v2},
      {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
      {"cuMemcpyAtoA", (void *)cuMemcpyAtoA_v2},
      {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
      {"cuMemcpy2D", (void *)cuMemcpy2D_v2},
      {"cuMemcpy2D_v2", (void *)cuMemcpy2D_v2},
      {"cuMemcpy2DUnaligned", (void *)cuMemcpy2DUnaligned_v2},
      {"cuMemcpy2DUnaligned_v2", (void *)cuMemcpy2DUnaligned_v2},
      {"cuMemcpy2DAsync", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy2DAsync_v2", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy2DAsync_ptsz", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy3D", (void *)cuMemcpy3D_v2},
      {"cuMemcpy3D_v2", (void *)cuMemcpy3D_v2},
      {"cuPointerGetAttribute", (void *)cuPointerGetAttribute},
      {"cuPointerGetAttributes", (void *)scuda_cuPointerGetAttributes_safe},
      {"cuGetExportTable", (void *)cuGetExportTable},
      {"cuModuleLoadData", (void *)cuModuleLoadData},
      {"cuModuleLoadDataEx", (void *)cuModuleLoadDataEx},
      {"cuLibraryLoadData", (void *)cuLibraryLoadData},
      {"cuLaunchKernel", (void *)cuLaunchKernel},
      {"cuLaunchKernelEx", (void *)cuLaunchKernelEx},
      {"cuMemcpyAsync", (void *)cuMemcpyAsync},
      {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
      {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
      {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
      {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
      {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
      {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
      {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
      {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
      {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo},
      {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
      {"cuStreamGetCaptureInfo_v3", (void *)cuStreamGetCaptureInfo_v3},
      {"cuCtxSynchronize", (void *)cuCtxSynchronize},
      {"cuStreamSynchronize", (void *)cuStreamSynchronize},
      {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize_ptsz},
      {"cuEventSynchronize", (void *)cuEventSynchronize},
      {"cuGetErrorName", (void *)cuGetErrorName},
      {"cuGetErrorString", (void *)cuGetErrorString},
      {"cuGraphAddKernelNode", (void *)cuGraphAddKernelNode_v2},
      {"cuGraphAddKernelNode_v2", (void *)cuGraphAddKernelNode_v2},
      {"cuGraphAddNode", (void *)cuGraphAddNode},
      {"cuGraphAddNode_v2", (void *)cuGraphAddNode_v2},
      {"cuGraphConditionalHandleCreate",
       (void *)cuGraphConditionalHandleCreate},
      {"cuGraphAddMemcpyNode", (void *)cuGraphAddMemcpyNode},
      {"cuGraphAddMemsetNode", (void *)cuGraphAddMemsetNode},
      {"cuGraphAddHostNode", (void *)cuGraphAddHostNode},
      {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
      {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
      {"cuDeviceGetGraphMemAttribute", (void *)cuDeviceGetGraphMemAttribute},
      {"cuDeviceSetGraphMemAttribute", (void *)cuDeviceSetGraphMemAttribute},
      {"cuGraphKernelNodeGetParams", (void *)cuGraphKernelNodeGetParams_v2},
      {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
      {"cuGraphExecKernelNodeSetParams",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphExecKernelNodeSetParams_v2",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphGetNodes", (void *)cuGraphGetNodes},
      {"cuGraphInstantiate", (void *)cuGraphInstantiate},
      {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
      {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
      {"cuStreamAddCallback", (void *)cuStreamAddCallback},
      {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
      {"cuStreamBeginCaptureToGraph", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamBeginCaptureToGraph_ptsz", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamUpdateCaptureDependencies",
       (void *)cuStreamUpdateCaptureDependencies},
      {"cuStreamUpdateCaptureDependencies_v2",
       (void *)cuStreamUpdateCaptureDependencies_v2},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       (void *)cuStreamUpdateCaptureDependencies_ptsz},
  };
  auto manual_it = manual_function_map.find(name);
  if (manual_it != manual_function_map.end()) {
    return manual_it->second;
  }

  void *unsupported_stub = scuda_get_unsupported_stub(name);
  if (unsupported_stub != nullptr) {
    return unsupported_stub;
  }

  if (scuda_stub_missing_enabled() &&
      scuda_symbol_looks_like_driver_api(name)) {
    return scuda_make_missing_stub(name);
  }

  // std::cout << "[dlsym] Falling back to real_dlsym for name: " << name <<
  // std::endl;
  return scuda_real_dlsym(handle, name);
}
