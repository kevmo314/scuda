#include <cuda.h>
#include <nvml.h>

#include <cstring>
#include <vector>

#include "codegen/gen_api.h"
#include "rpc.h"

extern int rpc_open();
extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);

// CUDA <= 12.6 ships NVML API 12, which does not define the versioned
// temperature struct. Keep the wrapper ABI-compatible with newer nvidia-smi.
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) ||                        \
    (defined(NVML_API_VERSION) && NVML_API_VERSION >= 13)
using scuda_nvmlTemperature_t = nvmlTemperature_t;
#else
typedef struct {
  unsigned int version;
  nvmlTemperatureSensors_t sensorType;
  int temperature;
} scuda_nvmlTemperature_t;
#endif

namespace {

struct scuda_nvml_remote_device {
  unsigned int conn_index = 0;
  unsigned int remote_index = 0;
  nvmlDevice_t remote_device = nullptr;
};

std::vector<scuda_nvml_remote_device> devices;
bool devices_ready = false;

nvmlReturn_t rpc_error() { return NVML_ERROR_UNKNOWN; }

int open_connection() {
  return rpc_open();
}

int connection_count() {
  if (open_connection() < 0) {
    return 0;
  }
  return rpc_size();
}

conn_t *connection(unsigned int index = 0) {
  int count = connection_count();
  if (index >= static_cast<unsigned int>(count)) {
    return nullptr;
  }
  return rpc_client_get_connection(index);
}

nvmlReturn_t call_no_args_on(conn_t *c, int op) {
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_uint_out_on(conn_t *c, int op, unsigned int *value) {
  nvmlReturn_t result = rpc_error();
  unsigned int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

nvmlReturn_t call_device_from_index_on(conn_t *c, int op, unsigned int index,
                                       nvmlDevice_t *device) {
  nvmlReturn_t result = rpc_error();
  nvmlDevice_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &index, sizeof(index)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

nvmlReturn_t ensure_devices() {
  if (open_connection() < 0) {
    return rpc_error();
  }
  if (devices_ready) {
    return NVML_SUCCESS;
  }

  devices.clear();
  int count = connection_count();
  for (int i = 0; i < count; ++i) {
    conn_t *c = connection(i);
    unsigned int device_count = 0;
    nvmlReturn_t result = call_uint_out_on(c, RPC_nvmlDeviceGetCount_v2,
                                           &device_count);
    if (result != NVML_SUCCESS) {
      devices.clear();
      return result;
    }
    for (unsigned int ordinal = 0; ordinal < device_count; ++ordinal) {
      nvmlDevice_t remote = nullptr;
      result = call_device_from_index_on(
          c, RPC_nvmlDeviceGetHandleByIndex_v2, ordinal, &remote);
      if (result != NVML_SUCCESS) {
        devices.clear();
        return result;
      }
      devices.push_back(scuda_nvml_remote_device{static_cast<unsigned int>(i),
                                                 ordinal, remote});
    }
  }
  devices_ready = true;
  return NVML_SUCCESS;
}

conn_t *connection_for_device(nvmlDevice_t *device) {
  if (device == nullptr || ensure_devices() != NVML_SUCCESS) {
    return nullptr;
  }
  if (devices.empty()) {
    return nullptr;
  }
  auto *mapped = reinterpret_cast<scuda_nvml_remote_device *>(*device);
  if (mapped < devices.data() || mapped >= devices.data() + devices.size()) {
    return connection();
  }
  *device = mapped->remote_device;
  return connection(mapped->conn_index);
}

nvmlReturn_t call_no_args(int op) { return call_no_args_on(connection(), op); }

nvmlReturn_t call_string_no_device(int op, char *value, unsigned int length) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &length, sizeof(length)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      (length != 0 && rpc_read(c, value, length) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_int_out(int op, int *value) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

nvmlReturn_t call_uint_out(int op, unsigned int *value) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  unsigned int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

nvmlReturn_t call_device_from_index(int op, unsigned int index,
                                    nvmlDevice_t *device) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlDevice_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &index, sizeof(index)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

nvmlReturn_t call_device_from_string(int op, const char *value,
                                     nvmlDevice_t *device) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlDevice_t temp = nullptr;
  unsigned int length =
      value == nullptr ? 0 : static_cast<unsigned int>(strlen(value) + 1);
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_write(c, value, length) < 0) ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

nvmlReturn_t call_device_string(int op, nvmlDevice_t device, char *value,
                                unsigned int length) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &length, sizeof(length)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      (length != 0 && rpc_read(c, value, length) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

template <typename T>
nvmlReturn_t call_device_struct(int op, nvmlDevice_t device, T *value) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  T temp = value == nullptr ? T{} : *value;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &temp, sizeof(temp)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

template <typename T>
nvmlReturn_t call_device_value(int op, nvmlDevice_t device, T *value) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  T temp = {};
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

template <typename Arg, typename Out>
nvmlReturn_t call_device_arg_value(int op, nvmlDevice_t device, Arg arg,
                                   Out *value) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  Out temp = {};
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &arg, sizeof(arg)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

nvmlReturn_t call_processes(int op, nvmlDevice_t device,
                            unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  unsigned int requested_count = infoCount == nullptr ? 0 : *infoCount;
  int has_infos = infos == nullptr ? 0 : 1;
  unsigned int returned_count = 0;
  unsigned int copied_count = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &requested_count, sizeof(requested_count)) < 0 ||
      rpc_write(c, &has_infos, sizeof(has_infos)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &returned_count, sizeof(returned_count)) < 0 ||
      rpc_read(c, &copied_count, sizeof(copied_count)) < 0 ||
      (copied_count != 0 &&
       rpc_read(c, infos, copied_count * sizeof(infos[0])) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (infoCount != nullptr) {
    *infoCount = returned_count;
  }
  return result;
}

nvmlReturn_t call_event_set_create(nvmlEventSet_t *set) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlEventSet_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetCreate) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (set != nullptr) {
    *set = temp;
  }
  return result;
}

nvmlReturn_t call_event_set_free(nvmlEventSet_t set) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetFree) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_event_set_wait(nvmlEventSet_t set, nvmlEventData_t *data,
                                 unsigned int timeoutms) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlEventData_t temp = {};
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetWait_v2) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 ||
      rpc_write(c, &timeoutms, sizeof(timeoutms)) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (data != nullptr) {
    *data = temp;
  }
  return result;
}

nvmlReturn_t call_device_register_events(nvmlDevice_t device,
                                         unsigned long long eventTypes,
                                         nvmlEventSet_t set) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  if (c == nullptr ||
      rpc_write_start_request(c, RPC_nvmlDeviceRegisterEvents) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &eventTypes, sizeof(eventTypes)) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

template <typename A, typename B>
nvmlReturn_t call_device_two_values(int op, nvmlDevice_t device, A *first,
                                    B *second) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  A first_temp = {};
  B second_temp = {};
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &first_temp, sizeof(first_temp)) < 0 ||
      rpc_read(c, &second_temp, sizeof(second_temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (first != nullptr) {
    *first = first_temp;
  }
  if (second != nullptr) {
    *second = second_temp;
  }
  return result;
}

} // namespace

#ifdef nvmlInit
#undef nvmlInit
#endif
#ifdef nvmlDeviceGetCount
#undef nvmlDeviceGetCount
#endif
#ifdef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByIndex
#endif
#ifdef nvmlDeviceGetHandleByPciBusId
#undef nvmlDeviceGetHandleByPciBusId
#endif
#ifdef nvmlDeviceGetPciInfo
#undef nvmlDeviceGetPciInfo
#endif
#ifdef nvmlDeviceGetComputeRunningProcesses
#undef nvmlDeviceGetComputeRunningProcesses
#endif
#ifdef nvmlDeviceGetGraphicsRunningProcesses
#undef nvmlDeviceGetGraphicsRunningProcesses
#endif
#ifdef nvmlDeviceGetMPSComputeRunningProcesses
#undef nvmlDeviceGetMPSComputeRunningProcesses
#endif
#ifdef nvmlEventSetWait
#undef nvmlEventSetWait
#endif

extern "C" nvmlReturn_t nvmlInit_v2(void) {
  if (open_connection() < 0) {
    return rpc_error();
  }
  nvmlReturn_t first_error = NVML_SUCCESS;
  int count = connection_count();
  for (int i = 0; i < count; ++i) {
    nvmlReturn_t result = call_no_args_on(connection(i), RPC_nvmlInit_v2);
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
  if (open_connection() < 0) {
    return rpc_error();
  }
  nvmlReturn_t first_error = NVML_SUCCESS;
  int count = connection_count();
  for (int i = 0; i < count; ++i) {
    conn_t *c = connection(i);
    nvmlReturn_t result = rpc_error();
    if (rpc_write_start_request(c, RPC_nvmlInitWithFlags) < 0 ||
        rpc_write(c, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(c) < 0 ||
        rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
      result = rpc_error();
    }
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" nvmlReturn_t nvmlShutdown(void) {
  return call_no_args(RPC_nvmlShutdown);
}

extern "C" const char *nvmlErrorString(nvmlReturn_t result) {
  switch (result) {
  case NVML_SUCCESS:
    return "Success";
  case NVML_ERROR_UNINITIALIZED:
    return "Uninitialized";
  case NVML_ERROR_INVALID_ARGUMENT:
    return "Invalid Argument";
  case NVML_ERROR_NOT_SUPPORTED:
    return "Not Supported";
  case NVML_ERROR_NO_PERMISSION:
    return "Insufficient Permissions";
  case NVML_ERROR_ALREADY_INITIALIZED:
    return "Already Initialized";
  case NVML_ERROR_NOT_FOUND:
    return "Not Found";
  case NVML_ERROR_INSUFFICIENT_SIZE:
    return "Insufficient Size";
  case NVML_ERROR_INSUFFICIENT_POWER:
    return "Insufficient External Power";
  case NVML_ERROR_DRIVER_NOT_LOADED:
    return "Driver Not Loaded";
  case NVML_ERROR_TIMEOUT:
    return "Timeout";
  case NVML_ERROR_IRQ_ISSUE:
    return "IRQ Issue";
  case NVML_ERROR_LIBRARY_NOT_FOUND:
    return "NVML Shared Library Not Found";
  case NVML_ERROR_FUNCTION_NOT_FOUND:
    return "Function Not Found";
  case NVML_ERROR_CORRUPTED_INFOROM:
    return "Corrupted InfoROM";
  case NVML_ERROR_GPU_IS_LOST:
    return "GPU is lost";
  case NVML_ERROR_RESET_REQUIRED:
    return "GPU requires reset";
  case NVML_ERROR_OPERATING_SYSTEM:
    return "The operating system has blocked the request";
  case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
    return "RM has detected an NVML/RM version mismatch";
  case NVML_ERROR_IN_USE:
    return "GPU is currently in use";
  case NVML_ERROR_MEMORY:
    return "Insufficient memory";
  case NVML_ERROR_NO_DATA:
    return "No data";
  case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED:
    return "VGPU ECC not supported";
  case NVML_ERROR_INSUFFICIENT_RESOURCES:
    return "Insufficient resources";
  default:
    return "Unknown Error";
  }
}

extern "C" nvmlReturn_t nvmlInternalGetExportTable(const void **ppExportTable,
                                                   const void *exportTableId) {
  static void *empty_table[512] = {};
  if (ppExportTable == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *ppExportTable = empty_table;
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
  return call_event_set_create(set);
}

extern "C" nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
  return call_event_set_free(set);
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set,
                                            nvmlEventData_t *data,
                                            unsigned int timeoutms) {
  return call_event_set_wait(set, data, timeoutms);
}

extern "C" nvmlReturn_t nvmlEventSetWait(nvmlEventSet_t set,
                                         nvmlEventData_t *data,
                                         unsigned int timeoutms) {
  return nvmlEventSetWait_v2(set, data, timeoutms);
}

extern "C" nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device,
                                                 unsigned long long eventTypes,
                                                 nvmlEventSet_t set) {
  return call_device_register_events(device, eventTypes, set);
}

extern "C" nvmlReturn_t nvmlSystemGetDriverVersion(char *version,
                                                   unsigned int length) {
  return call_string_no_device(RPC_nvmlSystemGetDriverVersion, version, length);
}

extern "C" nvmlReturn_t nvmlSystemGetNVMLVersion(char *version,
                                                 unsigned int length) {
  return call_string_no_device(RPC_nvmlSystemGetNVMLVersion, version, length);
}

extern "C" nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
  return call_int_out(RPC_nvmlSystemGetCudaDriverVersion, cudaDriverVersion);
}

extern "C" nvmlReturn_t
nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
  return call_int_out(RPC_nvmlSystemGetCudaDriverVersion_v2, cudaDriverVersion);
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (deviceCount != nullptr) {
    *deviceCount = static_cast<unsigned int>(devices.size());
  }
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
  return nvmlDeviceGetCount_v2(deviceCount);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                                      nvmlDevice_t *device) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (device == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  if (index >= devices.size()) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *device = reinterpret_cast<nvmlDevice_t>(&devices[index]);
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index,
                                                   nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid,
                                                  nvmlDevice_t *device) {
  return call_device_from_string(RPC_nvmlDeviceGetHandleByUUID, uuid, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId,
                                                         nvmlDevice_t *device) {
  return call_device_from_string(RPC_nvmlDeviceGetHandleByPciBusId_v2, pciBusId,
                                 device);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                                      nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                                          unsigned int length) {
  return call_device_string(RPC_nvmlDeviceGetName, device, name, length);
}

extern "C" nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid,
                                          unsigned int length) {
  return call_device_string(RPC_nvmlDeviceGetUUID, device, uuid, length);
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device,
                                           unsigned int *index) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (devices.empty() || index == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  auto *mapped = reinterpret_cast<scuda_nvml_remote_device *>(device);
  if (mapped < devices.data() || mapped >= devices.data() + devices.size()) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *index = static_cast<unsigned int>(mapped - devices.data());
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device,
                                                 unsigned int *minorNumber) {
  return call_device_value(RPC_nvmlDeviceGetMinorNumber, device, minorNumber);
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device,
                                                nvmlPciInfo_t *pci) {
  return call_device_struct(RPC_nvmlDeviceGetPciInfo_v3, device, pci);
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device,
                                                nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device,
                                             nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device,
                                                nvmlMemory_t *memory) {
  return call_device_struct(RPC_nvmlDeviceGetMemoryInfo, device, memory);
}

extern "C" nvmlReturn_t
nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                              nvmlUtilization_t *utilization) {
  return call_device_struct(RPC_nvmlDeviceGetUtilizationRates, device,
                            utilization);
}

extern "C" nvmlReturn_t
nvmlDeviceGetTemperature(nvmlDevice_t device,
                         nvmlTemperatureSensors_t sensorType,
                         unsigned int *temp) {
  return call_device_arg_value(RPC_nvmlDeviceGetTemperature, device, sensorType,
                               temp);
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device,
                                                unsigned int *power) {
  return call_device_value(RPC_nvmlDeviceGetPowerUsage, device, power);
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device,
                                                          unsigned int *limit) {
  return call_device_value(RPC_nvmlDeviceGetPowerManagementLimit, device,
                           limit);
}

extern "C" nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device,
                                               nvmlClockType_t type,
                                               unsigned int *clock) {
  return call_device_arg_value(RPC_nvmlDeviceGetClockInfo, device, type, clock);
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device,
                                                  nvmlClockType_t type,
                                                  unsigned int *clock) {
  return call_device_arg_value(RPC_nvmlDeviceGetMaxClockInfo, device, type,
                               clock);
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device,
                                                      nvmlPstates_t *pState) {
  return call_device_value(RPC_nvmlDeviceGetPerformanceState, device, pState);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device,
                                                 nvmlComputeMode_t *mode) {
  return call_device_value(RPC_nvmlDeviceGetComputeMode, device, mode);
}

extern "C" nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device,
                                                     nvmlEnableState_t *mode) {
  return call_device_value(RPC_nvmlDeviceGetPersistenceMode, device, mode);
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device,
                                              unsigned int *speed) {
  return call_device_value(RPC_nvmlDeviceGetFanSpeed, device, speed);
}

extern "C" nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device,
                                           nvmlBrandType_t *type) {
  return call_device_value(RPC_nvmlDeviceGetBrand, device, type);
}

extern "C" nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device,
                                                  char *version,
                                                  unsigned int length) {
  return call_device_string(RPC_nvmlDeviceGetVbiosVersion, device, version,
                            length);
}

extern "C" nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial,
                                            unsigned int length) {
  return call_device_string(RPC_nvmlDeviceGetSerial, device, serial, length);
}

extern "C" nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device,
                                                     char *partNumber,
                                                     unsigned int length) {
  return call_device_string(RPC_nvmlDeviceGetBoardPartNumber, device,
                            partNumber, length);
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device,
                                                 nvmlEnableState_t *display) {
  return call_device_value(RPC_nvmlDeviceGetDisplayMode, device, display);
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device,
                                                   nvmlEnableState_t *active) {
  return call_device_value(RPC_nvmlDeviceGetDisplayActive, device, active);
}

extern "C" nvmlReturn_t
nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *value) {
  return call_device_value(RPC_nvmlDeviceGetCurrPcieLinkGeneration, device,
                           value);
}

extern "C" nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device,
                                                       unsigned int *value) {
  return call_device_value(RPC_nvmlDeviceGetCurrPcieLinkWidth, device, value);
}

extern "C" nvmlReturn_t
nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *value) {
  return call_device_value(RPC_nvmlDeviceGetMaxPcieLinkGeneration, device,
                           value);
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device,
                                                      unsigned int *value) {
  return call_device_value(RPC_nvmlDeviceGetMaxPcieLinkWidth, device, value);
}

extern "C" nvmlReturn_t
nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter,
                            unsigned int *value) {
  return call_device_arg_value(RPC_nvmlDeviceGetPcieThroughput, device, counter,
                               value);
}

extern "C" nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device,
                                                       unsigned int *value) {
  return call_device_value(RPC_nvmlDeviceGetPcieReplayCounter, device, value);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetComputeRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetComputeRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetGraphicsRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetGraphicsRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetMPSComputeRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device,
                                                       unsigned int *count) {
  return call_device_value(RPC_nvmlDeviceGetMaxMigDeviceCount, device, count);
}

extern "C" nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device,
                                             nvmlEnableState_t *current,
                                             nvmlEnableState_t *pending) {
  return call_device_two_values(RPC_nvmlDeviceGetEccMode, device, current,
                                pending);
}

extern "C" nvmlReturn_t
nvmlDeviceGetTemperatureV(nvmlDevice_t device,
                          scuda_nvmlTemperature_t *temperature) {
  return call_device_struct(RPC_nvmlDeviceGetTemperatureV, device, temperature);
}

extern "C" nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device,
                                                        unsigned int *limit) {
  return call_device_value(RPC_nvmlDeviceGetEnforcedPowerLimit, device, limit);
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device,
                                                   nvmlMemory_v2_t *memory) {
  return call_device_struct(RPC_nvmlDeviceGetMemoryInfo_v2, device, memory);
}

extern "C" nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device,
                                             unsigned int *currentMode,
                                             unsigned int *pendingMode) {
  return call_device_two_values(RPC_nvmlDeviceGetMigMode, device, currentMode,
                                pendingMode);
}

extern "C" nvmlReturn_t
nvmlDeviceGetVirtualizationMode(nvmlDevice_t device,
                                nvmlGpuVirtualizationMode_t *pVirtualMode) {
  return call_device_value(RPC_nvmlDeviceGetVirtualizationMode, device,
                           pVirtualMode);
}

extern "C" nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device,
                                                    unsigned int *isMigDevice) {
  return call_device_value(RPC_nvmlDeviceIsMigDeviceHandle, device,
                           isMigDevice);
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(
    nvmlDevice_t device, unsigned int link,
    nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType) {
  return call_device_arg_value(RPC_nvmlDeviceGetNvLinkRemoteDeviceType, device,
                               link, pNvLinkDeviceType);
}

extern "C" nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(
    nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  return call_device_arg_value(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2, device,
                               link, pci);
}
