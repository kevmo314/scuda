#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <unordered_map>

#include <arpa/inet.h>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <netdb.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "gen_api.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const size_t size);
extern int rpc_read(void *data, size_t size);
extern int rpc_wait_for_response(const unsigned int request_id);
extern int rpc_end_request(void *return_value, const unsigned int request_id);

nvmlReturn_t nvmlInit_v2(void)
{
    int request_id = rpc_start_request(RPC_nvmlInit_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    int request_id = rpc_start_request(RPC_nvmlInitWithFlags);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlShutdown(void)
{
    int request_id = rpc_start_request(RPC_nvmlShutdown);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetNVMLVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetProcessName);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetCount);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(unitCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unitCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetHandleByIndex);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetUnitInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetLedState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetPsuInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetTemperature);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&type, sizeof(unsigned int)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetFanSpeedInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetDevices);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_read(devices, *deviceCount * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_read(hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCount_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAttributes_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleBySerial);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(serial, sizeof(char)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByUUID);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(uuid, sizeof(char)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByPciBusId_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(pciBusId, sizeof(char)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetName);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBrand);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetIndex);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(index, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSerial);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(serial, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryAffinity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&nodeSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeSet, nodeSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinityWithinScope);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetCpuAffinity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceClearCpuAffinity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyNearestGpus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetTopologyGpuSet);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(deviceArray, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetP2PStatus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        rpc_write(p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetUUID);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMdevUUID);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mdevUuid, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinorNumber);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardPartNumber);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(partNumber, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&object, sizeof(nvmlInforomObject_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomImageVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int *checksum)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomConfigurationChecksum);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(checksum, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(checksum, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceValidateInforom);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayActive);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPersistenceMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPciInfo_v3);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGenDevice)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkWidth);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkWidth);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieThroughput);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_write(value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieReplayCounter);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClockInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clock, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxClockInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clock, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetApplicationsClock);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultApplicationsClock);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceResetApplicationsClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClock);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxCustomerBoostClock);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedMemoryClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedGraphicsClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memoryClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetAutoBoostedClocksEnabled);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int *speed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int *targetSpeed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTargetFanSpeed);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultFanSpeed_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int *minSpeed, unsigned int *maxSpeed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxFanSpeed);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_write(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_read(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t *policy)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanControlPolicy_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetFanControlPolicy);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(&policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumFans);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(numFans, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numFans, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperature);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperatureThreshold);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetTemperatureThreshold);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(temp, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetThermalSettings);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPerformanceState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimit);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minLimit, sizeof(unsigned int)) < 0 ||
        rpc_write(maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minLimit, sizeof(unsigned int)) < 0 ||
        rpc_read(maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerUsage);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(power, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(power, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEnergyConsumption);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(energy, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(energy, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEnforcedPowerLimit);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuOperationMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_write(pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCudaComputeCapability);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(major, sizeof(int)) < 0 ||
        rpc_write(minor, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(major, sizeof(int)) < 0 ||
        rpc_read(minor, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEccMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t *defaultMode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultEccMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardId);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(boardId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(boardId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMultiGpuBoard);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEccErrors);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDetailedEccErrors);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryErrorCounter);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(&locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        rpc_write(count, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetUtilizationRates);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderUtilization);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderCapacity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_write(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderStats);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderSessions);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfos, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDecoderUtilization);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCStats);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCSessions);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDriverModel);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVbiosVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBridgeChipInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeRunningProcesses_v3);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGraphicsRunningProcesses_v3);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceOnSameBoard);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(onSameBoard, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(onSameBoard, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAPIRestriction);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_write(isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSamples);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlSamplingType_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_read(samples, *sampleCount * sizeof(nvmlSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBAR1MemoryInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetViolationStatus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        rpc_write(violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int *irqNum)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetIrqNum);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(irqNum, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(irqNum, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int *numCores)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumGpuCores);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(numCores, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numCores, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t *powerSource)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerSource);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int *busWidth)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryBusWidth);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(busWidth, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(busWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int *maxSpeed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieLinkMaxSpeed);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int *pcieSpeed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieSpeed);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int *adaptiveClockStatus)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAdaptiveClockInfoStatus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingStats);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingPids);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingBufferSize);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(timestamps, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_read(timestamps, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRemappedRows);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(corrRows, sizeof(unsigned int)) < 0 ||
        rpc_write(uncRows, sizeof(unsigned int)) < 0 ||
        rpc_write(isPending, sizeof(unsigned int)) < 0 ||
        rpc_write(failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(corrRows, sizeof(unsigned int)) < 0 ||
        rpc_read(uncRows, sizeof(unsigned int)) < 0 ||
        rpc_read(isPending, sizeof(unsigned int)) < 0 ||
        rpc_read(failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRowRemapperHistogram);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetArchitecture);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color)
{
    int request_id = rpc_start_request(RPC_nvmlUnitSetLedState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&color, sizeof(nvmlLedColor_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetPersistenceMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetComputeMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetEccMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&ecc, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceClearEccErrorCounts);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetDriverModel);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&driverModel, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuLockedClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&minGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&maxGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceResetGpuLockedClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemoryLockedClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&minMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&maxMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceResetMemoryLockedClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetApplicationsClocks);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&graphicsClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClkMonStatus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetPowerManagementLimit);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuOperationMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetAPIRestriction);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_write(&isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetAccountingMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceClearAccountingPids);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(version, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkCapability);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkErrorCounter);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        rpc_write(counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkErrorCounters);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkUtilizationControl);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_write(&reset, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationControl);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_write(txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_read(txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceFreezeNvLinkUtilizationCounter);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(&freeze, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkUtilizationCounter);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetCreate);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceRegisterEvents);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedEventTypes);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetWait_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_write(data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_write(&timeoutms, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetFree);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceModifyDrainState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(&newState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceQueryDrainState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_read(currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceRemoveGpu_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(&gpuState, sizeof(nvmlDetachGpuState_t)) < 0 ||
        rpc_write(&linkState, sizeof(nvmlPcieLinkState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceDiscoverGpus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFieldValues);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceClearFieldValues);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVirtualizationMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHostVgpuMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetVirtualizationMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGridLicensableFeatures_v4);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetProcessUtilization);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_write(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_read(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char *version)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(version, sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultMode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_write(defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_read(defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int *capResult)
{
    int request_id = rpc_start_request(RPC_nvmlGetVgpuDriverCapabilities);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int *capResult)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuCapabilities);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedVgpus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCreatableVgpus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetClass);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuTypeClass, sizeof(char)) < 0 ||
        rpc_write(size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeClass, sizeof(char)) < 0 ||
        rpc_read(size, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetName);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeName, *size * sizeof(char)) < 0 ||
        rpc_read(size, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetDeviceID);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_write(subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_read(subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFramebufferSize);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetNumDisplayHeads);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetResolution);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&displayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(xdim, sizeof(unsigned int)) < 0 ||
        rpc_write(ydim, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(xdim, sizeof(unsigned int)) < 0 ||
        rpc_read(ydim, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetLicense);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeLicenseString, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFrameRateLimit);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstances);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetActiveVgpus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuInstances, *vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmID);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_write(vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vmId, size * sizeof(char)) < 0 ||
        rpc_read(vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetUUID);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmDriverVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFbUsage);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseStatus);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(licensed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(licensed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetType);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFrameRateLimit);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEccMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderCapacity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceSetEncoderCapacity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderStats);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderSessions);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCStats);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCSessions);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuInstanceId);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuPciId);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuPciId, *length * sizeof(char)) < 0 ||
        rpc_read(length, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int *capResult)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetCapabilities);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMetadata);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuMetadata);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo)
{
    int request_id = rpc_start_request(RPC_nvmlGetVgpuCompatibility);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPgpuMetadataString);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pgpuMetadata, *bufferSize * sizeof(char)) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t *pSchedulerLog)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerLog);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t *pSchedulerState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t *pCapabilities)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerCapabilities);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current)
{
    int request_id = rpc_start_request(RPC_nvmlGetVgpuVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_write(current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_read(current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion)
{
    int request_id = rpc_start_request(RPC_nvmlSetVgpuVersion);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuUtilization);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilizationSamples, *vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuProcessUtilization);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilizationSamples, *vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingPids);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingStats);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceClearAccountingPids);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo)
{
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount)
{
    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceCount);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceInfoByIndex);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetMigMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(unsigned int)) < 0 ||
        rpc_write(activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigMode);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(currentMode, sizeof(unsigned int)) < 0 ||
        rpc_write(pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currentMode, sizeof(unsigned int)) < 0 ||
        rpc_read(pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfoV);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(placements, *count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstance);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(placement, sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_write(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceDestroy);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstances, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstances);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstances, *count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceById);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_write(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t *placements, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstance);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t *placement, nvmlComputeInstance_t *computeInstance)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstanceWithPlacement);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(placement, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_write(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance)
{
    int request_id = rpc_start_request(RPC_nvmlComputeInstanceDestroy);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstances);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstances, *count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance)
{
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceById);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_write(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlComputeInstanceGetInfo_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_write(info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceIsMigDeviceHandle);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceId);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(id, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeInstanceId);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(id, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxMigDeviceCount);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t *type)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBusType);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDynamicPstatesInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetFanSpeed_v2);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(&speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int *offset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkVfOffset);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(offset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpcClkVfOffset);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int *offset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkVfOffset);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(offset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemClkVfOffset);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int *minClockMHz, unsigned int *maxClockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxClockOfPState);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&pstate, sizeof(nvmlPstates_t)) < 0 ||
        rpc_write(minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedPerformanceStates);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pstates, sizeof(nvmlPstates_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pstates, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkMinMaxVfOffset);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minOffset, sizeof(int)) < 0 ||
        rpc_write(maxOffset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minOffset, sizeof(int)) < 0 ||
        rpc_read(maxOffset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkMinMaxVfOffset);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minOffset, sizeof(int)) < 0 ||
        rpc_write(maxOffset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minOffset, sizeof(int)) < 0 ||
        rpc_read(maxOffset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t *gpuFabricInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuFabricInfo);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t *metricsGet)
{
    int request_id = rpc_start_request(RPC_nvmlGpmMetricsGet);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample)
{
    int request_id = rpc_start_request(RPC_nvmlGpmSampleFree);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t *gpmSample)
{
    int request_id = rpc_start_request(RPC_nvmlGpmSampleAlloc);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample)
{
    int request_id = rpc_start_request(RPC_nvmlGpmSampleGet);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample)
{
    int request_id = rpc_start_request(RPC_nvmlGpmMigSampleGet);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_write(&gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t *gpmSupport)
{
    int request_id = rpc_start_request(RPC_nvmlGpmQueryDeviceSupport);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    nvmlReturn_t return_value;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

using cuInitPtr = CUresult (*)(unsigned int);

void *libCudaHandle = dlopen("libcuda.so.1", RTLD_NOW);

cudaError_t cudaGetDeviceCount(int *deviceCount) {
    std::cout << "Client: Calling cudaGetDeviceCount shim" << std::endl;

    if (!deviceCount) return cudaErrorInvalidValue;

    int request_id = rpc_start_request(RPC_cudaGetDeviceCount);
    cudaError_t result;

    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&result, sizeof(cudaError_t)) < 0 ||
        result != cudaSuccess ||
        rpc_read(deviceCount, sizeof(int)) < 0 ||
        rpc_end_request(&result, request_id) < 0)
        return cudaErrorUnknown;

    std::cout << "Client: Received device count from server: " << *deviceCount << std::endl;
    return cudaSuccess;
}

CUresult cuDevicePrimaryCtxGetStateShim(CUdevice dev, unsigned int *flags, int *active) {
    std::cout << "Client: Calling cuDevicePrimaryCtxGetStateShim" << std::endl;

    if (!flags || !active) return CUDA_ERROR_INVALID_VALUE;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxGetState);
    CUresult result;

    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&result, sizeof(CUresult)) < 0 ||
        result != CUDA_SUCCESS ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_read(active, sizeof(int)) < 0 ||
        rpc_end_request(&result, request_id) < 0)
        return CUDA_ERROR_UNKNOWN;

    std::cout << "Client: Received primary context state from server. Flags: " << *flags << ", Active: " << *active << std::endl;
    return CUDA_SUCCESS;
}

std::unordered_map<std::string, void *> functionMap = {
    {"nvmlInit_v2", (void *)nvmlInit_v2},
    {"nvmlInit", (void *)nvmlInit_v2},
    {"nvmlInitWithFlags", (void *)nvmlInitWithFlags},
    {"nvmlShutdown", (void *)nvmlShutdown},
    {"nvmlSystemGetDriverVersion", (void *)nvmlSystemGetDriverVersion},
    {"nvmlSystemGetNVMLVersion", (void *)nvmlSystemGetNVMLVersion},
    {"nvmlSystemGetCudaDriverVersion", (void *)nvmlSystemGetCudaDriverVersion},
    {"nvmlSystemGetCudaDriverVersion_v2", (void *)nvmlSystemGetCudaDriverVersion_v2},
    {"nvmlSystemGetProcessName", (void *)nvmlSystemGetProcessName},
    {"nvmlUnitGetCount", (void *)nvmlUnitGetCount},
    {"nvmlUnitGetHandleByIndex", (void *)nvmlUnitGetHandleByIndex},
    {"nvmlUnitGetUnitInfo", (void *)nvmlUnitGetUnitInfo},
    {"nvmlUnitGetLedState", (void *)nvmlUnitGetLedState},
    {"nvmlUnitGetPsuInfo", (void *)nvmlUnitGetPsuInfo},
    {"nvmlUnitGetTemperature", (void *)nvmlUnitGetTemperature},
    {"nvmlUnitGetFanSpeedInfo", (void *)nvmlUnitGetFanSpeedInfo},
    {"nvmlUnitGetDevices", (void *)nvmlUnitGetDevices},
    {"nvmlSystemGetHicVersion", (void *)nvmlSystemGetHicVersion},
    {"nvmlDeviceGetCount_v2", (void *)nvmlDeviceGetCount_v2},
    {"nvmlDeviceGetAttributes_v2", (void *)nvmlDeviceGetAttributes_v2},
    {"nvmlDeviceGetHandleByIndex_v2", (void *)nvmlDeviceGetHandleByIndex_v2},
    {"nvmlDeviceGetHandleBySerial", (void *)nvmlDeviceGetHandleBySerial},
    {"nvmlDeviceGetHandleByUUID", (void *)nvmlDeviceGetHandleByUUID},
    {"nvmlDeviceGetHandleByPciBusId_v2", (void *)nvmlDeviceGetHandleByPciBusId_v2},
    {"nvmlDeviceGetName", (void *)nvmlDeviceGetName},
    {"nvmlDeviceGetBrand", (void *)nvmlDeviceGetBrand},
    {"nvmlDeviceGetIndex", (void *)nvmlDeviceGetIndex},
    {"nvmlDeviceGetSerial", (void *)nvmlDeviceGetSerial},
    {"nvmlDeviceGetMemoryAffinity", (void *)nvmlDeviceGetMemoryAffinity},
    {"nvmlDeviceGetCpuAffinityWithinScope", (void *)nvmlDeviceGetCpuAffinityWithinScope},
    {"nvmlDeviceGetCpuAffinity", (void *)nvmlDeviceGetCpuAffinity},
    {"nvmlDeviceSetCpuAffinity", (void *)nvmlDeviceSetCpuAffinity},
    {"nvmlDeviceClearCpuAffinity", (void *)nvmlDeviceClearCpuAffinity},
    {"nvmlDeviceGetTopologyCommonAncestor", (void *)nvmlDeviceGetTopologyCommonAncestor},
    {"nvmlDeviceGetTopologyNearestGpus", (void *)nvmlDeviceGetTopologyNearestGpus},
    {"nvmlSystemGetTopologyGpuSet", (void *)nvmlSystemGetTopologyGpuSet},
    {"nvmlDeviceGetP2PStatus", (void *)nvmlDeviceGetP2PStatus},
    {"nvmlDeviceGetUUID", (void *)nvmlDeviceGetUUID},
    {"nvmlVgpuInstanceGetMdevUUID", (void *)nvmlVgpuInstanceGetMdevUUID},
    {"nvmlDeviceGetMinorNumber", (void *)nvmlDeviceGetMinorNumber},
    {"nvmlDeviceGetBoardPartNumber", (void *)nvmlDeviceGetBoardPartNumber},
    {"nvmlDeviceGetInforomVersion", (void *)nvmlDeviceGetInforomVersion},
    {"nvmlDeviceGetInforomImageVersion", (void *)nvmlDeviceGetInforomImageVersion},
    {"nvmlDeviceGetInforomConfigurationChecksum", (void *)nvmlDeviceGetInforomConfigurationChecksum},
    {"nvmlDeviceValidateInforom", (void *)nvmlDeviceValidateInforom},
    {"nvmlDeviceGetDisplayMode", (void *)nvmlDeviceGetDisplayMode},
    {"nvmlDeviceGetDisplayActive", (void *)nvmlDeviceGetDisplayActive},
    {"nvmlDeviceGetPersistenceMode", (void *)nvmlDeviceGetPersistenceMode},
    {"nvmlDeviceGetPciInfo_v3", (void *)nvmlDeviceGetPciInfo_v3},
    {"nvmlDeviceGetMaxPcieLinkGeneration", (void *)nvmlDeviceGetMaxPcieLinkGeneration},
    {"nvmlDeviceGetGpuMaxPcieLinkGeneration", (void *)nvmlDeviceGetGpuMaxPcieLinkGeneration},
    {"nvmlDeviceGetMaxPcieLinkWidth", (void *)nvmlDeviceGetMaxPcieLinkWidth},
    {"nvmlDeviceGetCurrPcieLinkGeneration", (void *)nvmlDeviceGetCurrPcieLinkGeneration},
    {"nvmlDeviceGetCurrPcieLinkWidth", (void *)nvmlDeviceGetCurrPcieLinkWidth},
    {"nvmlDeviceGetPcieThroughput", (void *)nvmlDeviceGetPcieThroughput},
    {"nvmlDeviceGetPcieReplayCounter", (void *)nvmlDeviceGetPcieReplayCounter},
    {"nvmlDeviceGetClockInfo", (void *)nvmlDeviceGetClockInfo},
    {"nvmlDeviceGetMaxClockInfo", (void *)nvmlDeviceGetMaxClockInfo},
    {"nvmlDeviceGetApplicationsClock", (void *)nvmlDeviceGetApplicationsClock},
    {"nvmlDeviceGetDefaultApplicationsClock", (void *)nvmlDeviceGetDefaultApplicationsClock},
    {"nvmlDeviceResetApplicationsClocks", (void *)nvmlDeviceResetApplicationsClocks},
    {"nvmlDeviceGetClock", (void *)nvmlDeviceGetClock},
    {"nvmlDeviceGetMaxCustomerBoostClock", (void *)nvmlDeviceGetMaxCustomerBoostClock},
    {"nvmlDeviceGetSupportedMemoryClocks", (void *)nvmlDeviceGetSupportedMemoryClocks},
    {"nvmlDeviceGetSupportedGraphicsClocks", (void *)nvmlDeviceGetSupportedGraphicsClocks},
    {"nvmlDeviceGetAutoBoostedClocksEnabled", (void *)nvmlDeviceGetAutoBoostedClocksEnabled},
    {"nvmlDeviceSetAutoBoostedClocksEnabled", (void *)nvmlDeviceSetAutoBoostedClocksEnabled},
    {"nvmlDeviceSetDefaultAutoBoostedClocksEnabled", (void *)nvmlDeviceSetDefaultAutoBoostedClocksEnabled},
    {"nvmlDeviceGetFanSpeed", (void *)nvmlDeviceGetFanSpeed},
    {"nvmlDeviceGetFanSpeed_v2", (void *)nvmlDeviceGetFanSpeed_v2},
    {"nvmlDeviceGetTargetFanSpeed", (void *)nvmlDeviceGetTargetFanSpeed},
    {"nvmlDeviceSetDefaultFanSpeed_v2", (void *)nvmlDeviceSetDefaultFanSpeed_v2},
    {"nvmlDeviceGetMinMaxFanSpeed", (void *)nvmlDeviceGetMinMaxFanSpeed},
    {"nvmlDeviceGetFanControlPolicy_v2", (void *)nvmlDeviceGetFanControlPolicy_v2},
    {"nvmlDeviceSetFanControlPolicy", (void *)nvmlDeviceSetFanControlPolicy},
    {"nvmlDeviceGetNumFans", (void *)nvmlDeviceGetNumFans},
    {"nvmlDeviceGetTemperature", (void *)nvmlDeviceGetTemperature},
    {"nvmlDeviceGetTemperatureThreshold", (void *)nvmlDeviceGetTemperatureThreshold},
    {"nvmlDeviceSetTemperatureThreshold", (void *)nvmlDeviceSetTemperatureThreshold},
    {"nvmlDeviceGetThermalSettings", (void *)nvmlDeviceGetThermalSettings},
    {"nvmlDeviceGetPerformanceState", (void *)nvmlDeviceGetPerformanceState},
    {"nvmlDeviceGetCurrentClocksThrottleReasons", (void *)nvmlDeviceGetCurrentClocksThrottleReasons},
    {"nvmlDeviceGetSupportedClocksThrottleReasons", (void *)nvmlDeviceGetSupportedClocksThrottleReasons},
    {"nvmlDeviceGetPowerState", (void *)nvmlDeviceGetPowerState},
    {"nvmlDeviceGetPowerManagementMode", (void *)nvmlDeviceGetPowerManagementMode},
    {"nvmlDeviceGetPowerManagementLimit", (void *)nvmlDeviceGetPowerManagementLimit},
    {"nvmlDeviceGetPowerManagementLimitConstraints", (void *)nvmlDeviceGetPowerManagementLimitConstraints},
    {"nvmlDeviceGetPowerManagementDefaultLimit", (void *)nvmlDeviceGetPowerManagementDefaultLimit},
    {"nvmlDeviceGetPowerUsage", (void *)nvmlDeviceGetPowerUsage},
    {"nvmlDeviceGetTotalEnergyConsumption", (void *)nvmlDeviceGetTotalEnergyConsumption},
    {"nvmlDeviceGetEnforcedPowerLimit", (void *)nvmlDeviceGetEnforcedPowerLimit},
    {"nvmlDeviceGetGpuOperationMode", (void *)nvmlDeviceGetGpuOperationMode},
    {"nvmlDeviceGetMemoryInfo", (void *)nvmlDeviceGetMemoryInfo},
    {"nvmlDeviceGetMemoryInfo_v2", (void *)nvmlDeviceGetMemoryInfo_v2},
    {"nvmlDeviceGetComputeMode", (void *)nvmlDeviceGetComputeMode},
    {"nvmlDeviceGetCudaComputeCapability", (void *)nvmlDeviceGetCudaComputeCapability},
    {"nvmlDeviceGetEccMode", (void *)nvmlDeviceGetEccMode},
    {"nvmlDeviceGetDefaultEccMode", (void *)nvmlDeviceGetDefaultEccMode},
    {"nvmlDeviceGetBoardId", (void *)nvmlDeviceGetBoardId},
    {"nvmlDeviceGetMultiGpuBoard", (void *)nvmlDeviceGetMultiGpuBoard},
    {"nvmlDeviceGetTotalEccErrors", (void *)nvmlDeviceGetTotalEccErrors},
    {"nvmlDeviceGetDetailedEccErrors", (void *)nvmlDeviceGetDetailedEccErrors},
    {"nvmlDeviceGetMemoryErrorCounter", (void *)nvmlDeviceGetMemoryErrorCounter},
    {"nvmlDeviceGetUtilizationRates", (void *)nvmlDeviceGetUtilizationRates},
    {"nvmlDeviceGetEncoderUtilization", (void *)nvmlDeviceGetEncoderUtilization},
    {"nvmlDeviceGetEncoderCapacity", (void *)nvmlDeviceGetEncoderCapacity},
    {"nvmlDeviceGetEncoderStats", (void *)nvmlDeviceGetEncoderStats},
    {"nvmlDeviceGetEncoderSessions", (void *)nvmlDeviceGetEncoderSessions},
    {"nvmlDeviceGetDecoderUtilization", (void *)nvmlDeviceGetDecoderUtilization},
    {"nvmlDeviceGetFBCStats", (void *)nvmlDeviceGetFBCStats},
    {"nvmlDeviceGetFBCSessions", (void *)nvmlDeviceGetFBCSessions},
    {"nvmlDeviceGetDriverModel", (void *)nvmlDeviceGetDriverModel},
    {"nvmlDeviceGetVbiosVersion", (void *)nvmlDeviceGetVbiosVersion},
    {"nvmlDeviceGetBridgeChipInfo", (void *)nvmlDeviceGetBridgeChipInfo},
    {"nvmlDeviceGetComputeRunningProcesses_v3", (void *)nvmlDeviceGetComputeRunningProcesses_v3},
    {"nvmlDeviceGetGraphicsRunningProcesses_v3", (void *)nvmlDeviceGetGraphicsRunningProcesses_v3},
    {"nvmlDeviceGetMPSComputeRunningProcesses_v3", (void *)nvmlDeviceGetMPSComputeRunningProcesses_v3},
    {"nvmlDeviceOnSameBoard", (void *)nvmlDeviceOnSameBoard},
    {"nvmlDeviceGetAPIRestriction", (void *)nvmlDeviceGetAPIRestriction},
    {"nvmlDeviceGetSamples", (void *)nvmlDeviceGetSamples},
    {"nvmlDeviceGetBAR1MemoryInfo", (void *)nvmlDeviceGetBAR1MemoryInfo},
    {"nvmlDeviceGetViolationStatus", (void *)nvmlDeviceGetViolationStatus},
    {"nvmlDeviceGetIrqNum", (void *)nvmlDeviceGetIrqNum},
    {"nvmlDeviceGetNumGpuCores", (void *)nvmlDeviceGetNumGpuCores},
    {"nvmlDeviceGetPowerSource", (void *)nvmlDeviceGetPowerSource},
    {"nvmlDeviceGetMemoryBusWidth", (void *)nvmlDeviceGetMemoryBusWidth},
    {"nvmlDeviceGetPcieLinkMaxSpeed", (void *)nvmlDeviceGetPcieLinkMaxSpeed},
    {"nvmlDeviceGetPcieSpeed", (void *)nvmlDeviceGetPcieSpeed},
    {"nvmlDeviceGetAdaptiveClockInfoStatus", (void *)nvmlDeviceGetAdaptiveClockInfoStatus},
    {"nvmlDeviceGetAccountingMode", (void *)nvmlDeviceGetAccountingMode},
    {"nvmlDeviceGetAccountingStats", (void *)nvmlDeviceGetAccountingStats},
    {"nvmlDeviceGetAccountingPids", (void *)nvmlDeviceGetAccountingPids},
    {"nvmlDeviceGetAccountingBufferSize", (void *)nvmlDeviceGetAccountingBufferSize},
    {"nvmlDeviceGetRetiredPages", (void *)nvmlDeviceGetRetiredPages},
    {"nvmlDeviceGetRetiredPages_v2", (void *)nvmlDeviceGetRetiredPages_v2},
    {"nvmlDeviceGetRetiredPagesPendingStatus", (void *)nvmlDeviceGetRetiredPagesPendingStatus},
    {"nvmlDeviceGetRemappedRows", (void *)nvmlDeviceGetRemappedRows},
    {"nvmlDeviceGetRowRemapperHistogram", (void *)nvmlDeviceGetRowRemapperHistogram},
    {"nvmlDeviceGetArchitecture", (void *)nvmlDeviceGetArchitecture},
    {"nvmlUnitSetLedState", (void *)nvmlUnitSetLedState},
    {"nvmlDeviceSetPersistenceMode", (void *)nvmlDeviceSetPersistenceMode},
    {"nvmlDeviceSetComputeMode", (void *)nvmlDeviceSetComputeMode},
    {"nvmlDeviceSetEccMode", (void *)nvmlDeviceSetEccMode},
    {"nvmlDeviceClearEccErrorCounts", (void *)nvmlDeviceClearEccErrorCounts},
    {"nvmlDeviceSetDriverModel", (void *)nvmlDeviceSetDriverModel},
    {"nvmlDeviceSetGpuLockedClocks", (void *)nvmlDeviceSetGpuLockedClocks},
    {"nvmlDeviceResetGpuLockedClocks", (void *)nvmlDeviceResetGpuLockedClocks},
    {"nvmlDeviceSetMemoryLockedClocks", (void *)nvmlDeviceSetMemoryLockedClocks},
    {"nvmlDeviceResetMemoryLockedClocks", (void *)nvmlDeviceResetMemoryLockedClocks},
    {"nvmlDeviceSetApplicationsClocks", (void *)nvmlDeviceSetApplicationsClocks},
    {"nvmlDeviceGetClkMonStatus", (void *)nvmlDeviceGetClkMonStatus},
    {"nvmlDeviceSetPowerManagementLimit", (void *)nvmlDeviceSetPowerManagementLimit},
    {"nvmlDeviceSetGpuOperationMode", (void *)nvmlDeviceSetGpuOperationMode},
    {"nvmlDeviceSetAPIRestriction", (void *)nvmlDeviceSetAPIRestriction},
    {"nvmlDeviceSetAccountingMode", (void *)nvmlDeviceSetAccountingMode},
    {"nvmlDeviceClearAccountingPids", (void *)nvmlDeviceClearAccountingPids},
    {"nvmlDeviceGetNvLinkState", (void *)nvmlDeviceGetNvLinkState},
    {"nvmlDeviceGetNvLinkVersion", (void *)nvmlDeviceGetNvLinkVersion},
    {"nvmlDeviceGetNvLinkCapability", (void *)nvmlDeviceGetNvLinkCapability},
    {"nvmlDeviceGetNvLinkRemotePciInfo_v2", (void *)nvmlDeviceGetNvLinkRemotePciInfo_v2},
    {"nvmlDeviceGetNvLinkErrorCounter", (void *)nvmlDeviceGetNvLinkErrorCounter},
    {"nvmlDeviceResetNvLinkErrorCounters", (void *)nvmlDeviceResetNvLinkErrorCounters},
    {"nvmlDeviceSetNvLinkUtilizationControl", (void *)nvmlDeviceSetNvLinkUtilizationControl},
    {"nvmlDeviceGetNvLinkUtilizationControl", (void *)nvmlDeviceGetNvLinkUtilizationControl},
    {"nvmlDeviceGetNvLinkUtilizationCounter", (void *)nvmlDeviceGetNvLinkUtilizationCounter},
    {"nvmlDeviceFreezeNvLinkUtilizationCounter", (void *)nvmlDeviceFreezeNvLinkUtilizationCounter},
    {"nvmlDeviceResetNvLinkUtilizationCounter", (void *)nvmlDeviceResetNvLinkUtilizationCounter},
    {"nvmlDeviceGetNvLinkRemoteDeviceType", (void *)nvmlDeviceGetNvLinkRemoteDeviceType},
    {"nvmlEventSetCreate", (void *)nvmlEventSetCreate},
    {"nvmlDeviceRegisterEvents", (void *)nvmlDeviceRegisterEvents},
    {"nvmlDeviceGetSupportedEventTypes", (void *)nvmlDeviceGetSupportedEventTypes},
    {"nvmlEventSetWait_v2", (void *)nvmlEventSetWait_v2},
    {"nvmlEventSetFree", (void *)nvmlEventSetFree},
    {"nvmlDeviceModifyDrainState", (void *)nvmlDeviceModifyDrainState},
    {"nvmlDeviceQueryDrainState", (void *)nvmlDeviceQueryDrainState},
    {"nvmlDeviceRemoveGpu_v2", (void *)nvmlDeviceRemoveGpu_v2},
    {"nvmlDeviceDiscoverGpus", (void *)nvmlDeviceDiscoverGpus},
    {"nvmlDeviceGetFieldValues", (void *)nvmlDeviceGetFieldValues},
    {"nvmlDeviceClearFieldValues", (void *)nvmlDeviceClearFieldValues},
    {"nvmlDeviceGetVirtualizationMode", (void *)nvmlDeviceGetVirtualizationMode},
    {"nvmlDeviceGetHostVgpuMode", (void *)nvmlDeviceGetHostVgpuMode},
    {"nvmlDeviceSetVirtualizationMode", (void *)nvmlDeviceSetVirtualizationMode},
    {"nvmlDeviceGetGridLicensableFeatures_v4", (void *)nvmlDeviceGetGridLicensableFeatures_v4},
    {"nvmlDeviceGetProcessUtilization", (void *)nvmlDeviceGetProcessUtilization},
    {"nvmlDeviceGetGspFirmwareVersion", (void *)nvmlDeviceGetGspFirmwareVersion},
    {"nvmlDeviceGetGspFirmwareMode", (void *)nvmlDeviceGetGspFirmwareMode},
    {"nvmlGetVgpuDriverCapabilities", (void *)nvmlGetVgpuDriverCapabilities},
    {"nvmlDeviceGetVgpuCapabilities", (void *)nvmlDeviceGetVgpuCapabilities},
    {"nvmlDeviceGetSupportedVgpus", (void *)nvmlDeviceGetSupportedVgpus},
    {"nvmlDeviceGetCreatableVgpus", (void *)nvmlDeviceGetCreatableVgpus},
    {"nvmlVgpuTypeGetClass", (void *)nvmlVgpuTypeGetClass},
    {"nvmlVgpuTypeGetName", (void *)nvmlVgpuTypeGetName},
    {"nvmlVgpuTypeGetGpuInstanceProfileId", (void *)nvmlVgpuTypeGetGpuInstanceProfileId},
    {"nvmlVgpuTypeGetDeviceID", (void *)nvmlVgpuTypeGetDeviceID},
    {"nvmlVgpuTypeGetFramebufferSize", (void *)nvmlVgpuTypeGetFramebufferSize},
    {"nvmlVgpuTypeGetNumDisplayHeads", (void *)nvmlVgpuTypeGetNumDisplayHeads},
    {"nvmlVgpuTypeGetResolution", (void *)nvmlVgpuTypeGetResolution},
    {"nvmlVgpuTypeGetLicense", (void *)nvmlVgpuTypeGetLicense},
    {"nvmlVgpuTypeGetFrameRateLimit", (void *)nvmlVgpuTypeGetFrameRateLimit},
    {"nvmlVgpuTypeGetMaxInstances", (void *)nvmlVgpuTypeGetMaxInstances},
    {"nvmlVgpuTypeGetMaxInstancesPerVm", (void *)nvmlVgpuTypeGetMaxInstancesPerVm},
    {"nvmlDeviceGetActiveVgpus", (void *)nvmlDeviceGetActiveVgpus},
    {"nvmlVgpuInstanceGetVmID", (void *)nvmlVgpuInstanceGetVmID},
    {"nvmlVgpuInstanceGetUUID", (void *)nvmlVgpuInstanceGetUUID},
    {"nvmlVgpuInstanceGetVmDriverVersion", (void *)nvmlVgpuInstanceGetVmDriverVersion},
    {"nvmlVgpuInstanceGetFbUsage", (void *)nvmlVgpuInstanceGetFbUsage},
    {"nvmlVgpuInstanceGetLicenseStatus", (void *)nvmlVgpuInstanceGetLicenseStatus},
    {"nvmlVgpuInstanceGetType", (void *)nvmlVgpuInstanceGetType},
    {"nvmlVgpuInstanceGetFrameRateLimit", (void *)nvmlVgpuInstanceGetFrameRateLimit},
    {"nvmlVgpuInstanceGetEccMode", (void *)nvmlVgpuInstanceGetEccMode},
    {"nvmlVgpuInstanceGetEncoderCapacity", (void *)nvmlVgpuInstanceGetEncoderCapacity},
    {"nvmlVgpuInstanceSetEncoderCapacity", (void *)nvmlVgpuInstanceSetEncoderCapacity},
    {"nvmlVgpuInstanceGetEncoderStats", (void *)nvmlVgpuInstanceGetEncoderStats},
    {"nvmlVgpuInstanceGetEncoderSessions", (void *)nvmlVgpuInstanceGetEncoderSessions},
    {"nvmlVgpuInstanceGetFBCStats", (void *)nvmlVgpuInstanceGetFBCStats},
    {"nvmlVgpuInstanceGetFBCSessions", (void *)nvmlVgpuInstanceGetFBCSessions},
    {"nvmlVgpuInstanceGetGpuInstanceId", (void *)nvmlVgpuInstanceGetGpuInstanceId},
    {"nvmlVgpuInstanceGetGpuPciId", (void *)nvmlVgpuInstanceGetGpuPciId},
    {"nvmlVgpuTypeGetCapabilities", (void *)nvmlVgpuTypeGetCapabilities},
    {"nvmlVgpuInstanceGetMetadata", (void *)nvmlVgpuInstanceGetMetadata},
    {"nvmlDeviceGetVgpuMetadata", (void *)nvmlDeviceGetVgpuMetadata},
    {"nvmlGetVgpuCompatibility", (void *)nvmlGetVgpuCompatibility},
    {"nvmlDeviceGetPgpuMetadataString", (void *)nvmlDeviceGetPgpuMetadataString},
    {"nvmlDeviceGetVgpuSchedulerLog", (void *)nvmlDeviceGetVgpuSchedulerLog},
    {"nvmlDeviceGetVgpuSchedulerState", (void *)nvmlDeviceGetVgpuSchedulerState},
    {"nvmlDeviceGetVgpuSchedulerCapabilities", (void *)nvmlDeviceGetVgpuSchedulerCapabilities},
    {"nvmlGetVgpuVersion", (void *)nvmlGetVgpuVersion},
    {"nvmlSetVgpuVersion", (void *)nvmlSetVgpuVersion},
    {"nvmlDeviceGetVgpuUtilization", (void *)nvmlDeviceGetVgpuUtilization},
    {"nvmlDeviceGetVgpuProcessUtilization", (void *)nvmlDeviceGetVgpuProcessUtilization},
    {"nvmlVgpuInstanceGetAccountingMode", (void *)nvmlVgpuInstanceGetAccountingMode},
    {"nvmlVgpuInstanceGetAccountingPids", (void *)nvmlVgpuInstanceGetAccountingPids},
    {"nvmlVgpuInstanceGetAccountingStats", (void *)nvmlVgpuInstanceGetAccountingStats},
    {"nvmlVgpuInstanceClearAccountingPids", (void *)nvmlVgpuInstanceClearAccountingPids},
    {"nvmlVgpuInstanceGetLicenseInfo_v2", (void *)nvmlVgpuInstanceGetLicenseInfo_v2},
    {"nvmlGetExcludedDeviceCount", (void *)nvmlGetExcludedDeviceCount},
    {"nvmlGetExcludedDeviceInfoByIndex", (void *)nvmlGetExcludedDeviceInfoByIndex},
    {"nvmlDeviceSetMigMode", (void *)nvmlDeviceSetMigMode},
    {"nvmlDeviceGetMigMode", (void *)nvmlDeviceGetMigMode},
    {"nvmlDeviceGetGpuInstanceProfileInfo", (void *)nvmlDeviceGetGpuInstanceProfileInfo},
    {"nvmlDeviceGetGpuInstanceProfileInfoV", (void *)nvmlDeviceGetGpuInstanceProfileInfoV},
    {"nvmlDeviceGetGpuInstancePossiblePlacements_v2", (void *)nvmlDeviceGetGpuInstancePossiblePlacements_v2},
    {"nvmlDeviceGetGpuInstanceRemainingCapacity", (void *)nvmlDeviceGetGpuInstanceRemainingCapacity},
    {"nvmlDeviceCreateGpuInstance", (void *)nvmlDeviceCreateGpuInstance},
    {"nvmlDeviceCreateGpuInstanceWithPlacement", (void *)nvmlDeviceCreateGpuInstanceWithPlacement},
    {"nvmlGpuInstanceDestroy", (void *)nvmlGpuInstanceDestroy},
    {"nvmlDeviceGetGpuInstances", (void *)nvmlDeviceGetGpuInstances},
    {"nvmlDeviceGetGpuInstanceById", (void *)nvmlDeviceGetGpuInstanceById},
    {"nvmlGpuInstanceGetInfo", (void *)nvmlGpuInstanceGetInfo},
    {"nvmlGpuInstanceGetComputeInstanceProfileInfo", (void *)nvmlGpuInstanceGetComputeInstanceProfileInfo},
    {"nvmlGpuInstanceGetComputeInstanceProfileInfoV", (void *)nvmlGpuInstanceGetComputeInstanceProfileInfoV},
    {"nvmlGpuInstanceGetComputeInstanceRemainingCapacity", (void *)nvmlGpuInstanceGetComputeInstanceRemainingCapacity},
    {"nvmlGpuInstanceGetComputeInstancePossiblePlacements", (void *)nvmlGpuInstanceGetComputeInstancePossiblePlacements},
    {"nvmlGpuInstanceCreateComputeInstance", (void *)nvmlGpuInstanceCreateComputeInstance},
    {"nvmlGpuInstanceCreateComputeInstanceWithPlacement", (void *)nvmlGpuInstanceCreateComputeInstanceWithPlacement},
    {"nvmlComputeInstanceDestroy", (void *)nvmlComputeInstanceDestroy},
    {"nvmlGpuInstanceGetComputeInstances", (void *)nvmlGpuInstanceGetComputeInstances},
    {"nvmlGpuInstanceGetComputeInstanceById", (void *)nvmlGpuInstanceGetComputeInstanceById},
    {"nvmlComputeInstanceGetInfo_v2", (void *)nvmlComputeInstanceGetInfo_v2},
    {"nvmlDeviceIsMigDeviceHandle", (void *)nvmlDeviceIsMigDeviceHandle},
    {"nvmlDeviceGetGpuInstanceId", (void *)nvmlDeviceGetGpuInstanceId},
    {"nvmlDeviceGetComputeInstanceId", (void *)nvmlDeviceGetComputeInstanceId},
    {"nvmlDeviceGetMaxMigDeviceCount", (void *)nvmlDeviceGetMaxMigDeviceCount},
    {"nvmlDeviceGetMigDeviceHandleByIndex", (void *)nvmlDeviceGetMigDeviceHandleByIndex},
    {"nvmlDeviceGetDeviceHandleFromMigDeviceHandle", (void *)nvmlDeviceGetDeviceHandleFromMigDeviceHandle},
    {"nvmlDeviceGetBusType", (void *)nvmlDeviceGetBusType},
    {"nvmlDeviceGetDynamicPstatesInfo", (void *)nvmlDeviceGetDynamicPstatesInfo},
    {"nvmlDeviceSetFanSpeed_v2", (void *)nvmlDeviceSetFanSpeed_v2},
    {"nvmlDeviceGetGpcClkVfOffset", (void *)nvmlDeviceGetGpcClkVfOffset},
    {"nvmlDeviceSetGpcClkVfOffset", (void *)nvmlDeviceSetGpcClkVfOffset},
    {"nvmlDeviceGetMemClkVfOffset", (void *)nvmlDeviceGetMemClkVfOffset},
    {"nvmlDeviceSetMemClkVfOffset", (void *)nvmlDeviceSetMemClkVfOffset},
    {"nvmlDeviceGetMinMaxClockOfPState", (void *)nvmlDeviceGetMinMaxClockOfPState},
    {"nvmlDeviceGetSupportedPerformanceStates", (void *)nvmlDeviceGetSupportedPerformanceStates},
    {"nvmlDeviceGetGpcClkMinMaxVfOffset", (void *)nvmlDeviceGetGpcClkMinMaxVfOffset},
    {"nvmlDeviceGetMemClkMinMaxVfOffset", (void *)nvmlDeviceGetMemClkMinMaxVfOffset},
    {"nvmlDeviceGetGpuFabricInfo", (void *)nvmlDeviceGetGpuFabricInfo},
    {"nvmlGpmMetricsGet", (void *)nvmlGpmMetricsGet},
    {"nvmlGpmSampleFree", (void *)nvmlGpmSampleFree},
    {"nvmlGpmSampleAlloc", (void *)nvmlGpmSampleAlloc},
    {"nvmlGpmSampleGet", (void *)nvmlGpmSampleGet},
    {"nvmlGpmMigSampleGet", (void *)nvmlGpmMigSampleGet},
    {"nvmlGpmQueryDeviceSupport", (void *)nvmlGpmQueryDeviceSupport},
    {"nvmlDeviceSetNvLinkDeviceLowPowerThreshold", (void *)nvmlDeviceSetNvLinkDeviceLowPowerThreshold},
    {"cuDevicePrimaryCtxGetState", (void *)cuDevicePrimaryCtxGetStateShim}
};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it != functionMap.end())
        return it->second;
    return nullptr;
}
