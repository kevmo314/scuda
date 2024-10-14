#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "manual_client.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const std::size_t size);
extern int rpc_read(void *data, const std::size_t size);
extern int rpc_wait_for_response(const unsigned int request_id);
extern int rpc_end_request(void *return_value, const unsigned int request_id);
nvmlReturn_t nvmlInit_v2()
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlInit_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlInitWithFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlShutdown()
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlShutdown);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetNVMLVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetProcessName);
    if (request_id < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unitCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetUnitInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetLedState);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetPsuInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetTemperature);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&type, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetFanSpeedInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetDevices);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_read(devices, *deviceCount * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    if (request_id < 0 ||
        rpc_write(&hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_read(hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCount_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAttributes_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleBySerial);
    std::size_t serial_len = std::strlen(serial) + 1;
    if (request_id < 0 ||
        rpc_write(&serial_len, sizeof(std::size_t)) < 0 ||
        rpc_write(serial, serial_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByUUID);
    std::size_t uuid_len = std::strlen(uuid) + 1;
    if (request_id < 0 ||
        rpc_write(&uuid_len, sizeof(std::size_t)) < 0 ||
        rpc_write(uuid, uuid_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByPciBusId_v2);
    std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
    if (request_id < 0 ||
        rpc_write(&pciBusId_len, sizeof(std::size_t)) < 0 ||
        rpc_write(pciBusId, pciBusId_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetName);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBrand);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(index, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSerial);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(serial, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryAffinity);
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

nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinityWithinScope);
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

nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinity);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyNearestGpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetTopologyGpuSet);
    if (request_id < 0 ||
        rpc_write(&cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetP2PStatus);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetUUID);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMdevUUID);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mdevUuid, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinorNumber);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardPartNumber);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(partNumber, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomVersion);
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

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomImageVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomConfigurationChecksum);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(checksum, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceValidateInforom);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayActive);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPersistenceMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPciInfo_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieThroughput);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieReplayCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetApplicationsClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultApplicationsClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetApplicationsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxCustomerBoostClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedMemoryClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(clocksMHz, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedGraphicsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memoryClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(clocksMHz, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetAutoBoostedClocksEnabled);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTargetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_read(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanControlPolicy_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetFanControlPolicy);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(&policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumFans);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numFans, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperature);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(&temp, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetThermalSettings);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPerformanceState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long* clocksThrottleReasons)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long* supportedClocksThrottleReasons)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minLimit, sizeof(unsigned int)) < 0 ||
        rpc_read(maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerUsage);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(power, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEnergyConsumption);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(energy, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEnforcedPowerLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuOperationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCudaComputeCapability);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(major, sizeof(int)) < 0 ||
        rpc_read(minor, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(boardId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMultiGpuBoard);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEccErrors);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDetailedEccErrors);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryErrorCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(&locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetUtilizationRates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfos, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDecoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCSessions);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, *sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDriverModel);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVbiosVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBridgeChipInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGraphicsRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceOnSameBoard);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(onSameBoard, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAPIRestriction);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSamples);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlSamplingType_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(&sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_read(samples, *sampleCount * sizeof(nvmlSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBAR1MemoryInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t* violTime)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetViolationStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetIrqNum);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(irqNum, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumGpuCores);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numCores, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerSource);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryBusWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(busWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieLinkMaxSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAdaptiveClockInfoStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingPids);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingBufferSize);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(&pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(&pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_read(timestamps, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRemappedRows);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(corrRows, sizeof(unsigned int)) < 0 ||
        rpc_read(uncRows, sizeof(unsigned int)) < 0 ||
        rpc_read(isPending, sizeof(unsigned int)) < 0 ||
        rpc_read(failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRowRemapperHistogram);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetArchitecture);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitSetLedState);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetPersistenceMode);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetComputeMode);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetEccMode);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearEccErrorCounts);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetDriverModel);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuLockedClocks);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetGpuLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemoryLockedClocks);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetMemoryLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetApplicationsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&graphicsClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClkMonStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetPowerManagementLimit);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuOperationMode);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetAPIRestriction);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetAccountingMode);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkCapability);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkErrorCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkErrorCounters);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control, unsigned int reset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(&control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_write(&reset, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long* rxcounter, unsigned long long* txcounter)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_read(txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceFreezeNvLinkUtilizationCounter);
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
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlEventSetCreate);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceRegisterEvents);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedEventTypes);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlEventSetWait_v2);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_write(&timeoutms, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlEventSetFree);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceModifyDrainState);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(&newState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceQueryDrainState);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceRemoveGpu_v2);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(&gpuState, sizeof(nvmlDetachGpuState_t)) < 0 ||
        rpc_write(&linkState, sizeof(nvmlPcieLinkState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceDiscoverGpus);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFieldValues);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearFieldValues);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVirtualizationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHostVgpuMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetVirtualizationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGridLicensableFeatures_v4);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetProcessUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilization, *processSamplesCount * sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_read(defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetVgpuDriverCapabilities);
    if (request_id < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuCapabilities);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedVgpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCreatableVgpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetClass);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(size, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeClass, *size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetName);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(size, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeName, *size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetDeviceID);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_read(subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFramebufferSize);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetNumDisplayHeads);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetResolution);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&displayIndex, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(xdim, sizeof(unsigned int)) < 0 ||
        rpc_read(ydim, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetLicense);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeLicenseString, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFrameRateLimit);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstances);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetActiveVgpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuInstances, *vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmID);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vmId, size * sizeof(char)) < 0 ||
        rpc_read(vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetUUID);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmDriverVersion);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFbUsage);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseStatus);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(licensed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetType);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFrameRateLimit);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceSetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCSessions);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, *sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuInstanceId);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuPciId);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(length, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuPciId, *length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetCapabilities);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMetadata);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuMetadata, *bufferSize * sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuMetadata);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(pgpuMetadata, *bufferSize * sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetVgpuCompatibility);
    if (request_id < 0 ||
        rpc_write(&vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPgpuMetadataString);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(pgpuMetadata, *bufferSize * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerLog);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerCapabilities);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetVgpuVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_read(current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSetVgpuVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(&sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(&vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilizationSamples, *vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuProcessUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(&vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilizationSamples, *vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingPids);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceInfoByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetMigMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currentMode, sizeof(unsigned int)) < 0 ||
        rpc_read(pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfoV);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(placements, *count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstance);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstances);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(gpuInstances, *count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceById);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetInfo);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(placements, *count * sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstance);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlComputeInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstances);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(computeInstances, *count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceById);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlComputeInstanceGetInfo_v2);
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceIsMigDeviceHandle);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(id, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeInstanceId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(id, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxMigDeviceCount);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    if (request_id < 0 ||
        rpc_write(&migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBusType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDynamicPstatesInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(&speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(offset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpcClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(offset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxClockOfPState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&pstate, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedPerformanceStates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pstates, size * sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkMinMaxVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minOffset, sizeof(int)) < 0 ||
        rpc_read(maxOffset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkMinMaxVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minOffset, sizeof(int)) < 0 ||
        rpc_read(maxOffset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* gpuFabricInfo)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuFabricInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t* metricsGet)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmMetricsGet);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmSampleFree);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t* gpmSample)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmSampleAlloc);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmSampleGet);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmMigSampleGet);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t* gpmSupport)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmQueryDeviceSupport);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info)
{
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuInit(unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuInit);
    if (request_id < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDriverGetVersion(int* driverVersion)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDriverGetVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(driverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGet);
    if (request_id < 0 ||
        rpc_write(&ordinal, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(CUdevice)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetCount(int* count)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetName);
    if (request_id < 0 ||
        rpc_write(&len, sizeof(int)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, len * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetUuid);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, 16) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetUuid_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, 16) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetLuid);
    std::size_t luid_len = std::strlen(luid) + 1;
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&luid_len, sizeof(luid_len)) < 0 ||
        rpc_read(luid, luid_len) < 0 ||
        rpc_read(deviceNodeMask, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceTotalMem_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bytes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetTexture1DLinearMaxWidth);
    if (request_id < 0 ||
        rpc_write(&format, sizeof(CUarray_format)) < 0 ||
        rpc_write(&numChannels, sizeof(unsigned)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxWidthInElements, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetAttribute);
    if (request_id < 0 ||
        rpc_write(&attrib, sizeof(CUdevice_attribute)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pi, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceSetMemPool);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetMemPool);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetDefaultMemPool);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetExecAffinitySupport);
    if (request_id < 0 ||
        rpc_write(&type, sizeof(CUexecAffinityType)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pi, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFlushGPUDirectRDMAWrites);
    if (request_id < 0 ||
        rpc_write(&target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0 ||
        rpc_write(&scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetProperties);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(prop, sizeof(CUdevprop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceComputeCapability);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(major, sizeof(int)) < 0 ||
        rpc_read(minor, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxRetain);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxRelease_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxSetFlags_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxGetState);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_read(active, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxReset_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxCreate_v2);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxCreate_v3);
    if (request_id < 0 ||
        rpc_write(&numParams, sizeof(int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_read(paramsArray, numParams * sizeof(CUexecAffinityParam)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDestroy_v2(CUcontext ctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxDestroy_v2);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxPushCurrent_v2);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxPopCurrent_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetCurrent);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetCurrent(CUcontext* pctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetCurrent);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetDevice(CUdevice* device)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetDevice);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(CUdevice)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetFlags(unsigned int* flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetFlags);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetId);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ctxId, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSynchronize()
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSynchronize);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(CUlimit)) < 0 ||
        rpc_write(&value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(CUlimit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pvalue, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetCacheConfig);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pconfig, sizeof(CUfunc_cache)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(CUfunc_cache)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetSharedMemConfig);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pConfig, sizeof(CUsharedconfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(CUsharedconfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetApiVersion);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetStreamPriorityRange);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(leastPriority, sizeof(int)) < 0 ||
        rpc_read(greatestPriority, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxResetPersistingL2Cache()
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxResetPersistingL2Cache);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetExecAffinity);
    if (request_id < 0 ||
        rpc_write(&type, sizeof(CUexecAffinityType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pExecAffinity, sizeof(CUexecAffinityParam)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxAttach);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDetach(CUcontext ctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxDetach);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleLoad);
    std::size_t fname_len = std::strlen(fname) + 1;
    if (request_id < 0 ||
        rpc_write(&fname_len, sizeof(std::size_t)) < 0 ||
        rpc_write(fname, fname_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(module, sizeof(CUmodule)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleUnload(CUmodule hmod)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleUnload);
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetLoadingMode);
    if (request_id < 0 ||
        rpc_write(&mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetFunction);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hfunc, sizeof(CUfunction)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetGlobal_v2);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(bytes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkCreate_v2);
    if (request_id < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(&options, sizeof(CUjit_option)) < 0 ||
        rpc_write(&optionValues, sizeof(void*)) < 0 ||
        rpc_write(&stateOut, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(options, sizeof(CUjit_option)) < 0 ||
        rpc_read(optionValues, sizeof(void*)) < 0 ||
        rpc_read(stateOut, sizeof(CUlinkState)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkAddData_v2);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_write(&type, sizeof(CUjitInputType)) < 0 ||
        rpc_write(&data, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(optionValues, numOptions * sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkAddFile_v2);
    std::size_t path_len = std::strlen(path) + 1;
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_write(&type, sizeof(CUjitInputType)) < 0 ||
        rpc_write(&path_len, sizeof(std::size_t)) < 0 ||
        rpc_write(path, path_len) < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(optionValues, numOptions * sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkComplete);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cubinOut, sizeof(void*)) < 0 ||
        rpc_read(sizeOut, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkDestroy(CUlinkState state)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkDestroy);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetTexRef);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetSurfRef);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryLoadFromFile);
    std::size_t fileName_len = std::strlen(fileName) + 1;
    if (request_id < 0 ||
        rpc_write(&fileName_len, sizeof(std::size_t)) < 0 ||
        rpc_write(fileName, fileName_len) < 0 ||
        rpc_write(&numJitOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(jitOptions, numJitOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(jitOptionsValues, numJitOptions * sizeof(void*)) < 0 ||
        rpc_write(&numLibraryOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(libraryOptions, numLibraryOptions * sizeof(CUlibraryOption)) < 0 ||
        rpc_write(libraryOptionValues, numLibraryOptions * sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(library, sizeof(CUlibrary)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryUnload(CUlibrary library)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryUnload);
    if (request_id < 0 ||
        rpc_write(&library, sizeof(CUlibrary)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetKernel);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&library, sizeof(CUlibrary)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pKernel, sizeof(CUkernel)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetModule);
    if (request_id < 0 ||
        rpc_write(&library, sizeof(CUlibrary)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pMod, sizeof(CUmodule)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelGetFunction);
    if (request_id < 0 ||
        rpc_write(&kernel, sizeof(CUkernel)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pFunc, sizeof(CUfunction)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetGlobal);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&library, sizeof(CUlibrary)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(bytes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetManaged);
    std::size_t name_len = std::strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&library, sizeof(CUlibrary)) < 0 ||
        rpc_write(&name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(name, name_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(bytes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetUnifiedFunction);
    std::size_t symbol_len = std::strlen(symbol) + 1;
    if (request_id < 0 ||
        rpc_write(&library, sizeof(CUlibrary)) < 0 ||
        rpc_write(&symbol_len, sizeof(std::size_t)) < 0 ||
        rpc_write(symbol, symbol_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelGetAttribute);
    if (request_id < 0 ||
        rpc_write(&pi, sizeof(int)) < 0 ||
        rpc_write(&attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(&kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pi, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelSetAttribute);
    if (request_id < 0 ||
        rpc_write(&attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(&val, sizeof(int)) < 0 ||
        rpc_write(&kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(&config, sizeof(CUfunc_cache)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetInfo_v2);
    if (request_id < 0 ||
        rpc_write(&free, sizeof(size_t)) < 0 ||
        rpc_write(&total, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(free, sizeof(size_t)) < 0 ||
        rpc_read(total, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAlloc_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocPitch_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&pPitch, sizeof(size_t)) < 0 ||
        rpc_write(&WidthInBytes, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_write(&ElementSizeBytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(pPitch, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemFree_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetAddressRange_v2);
    if (request_id < 0 ||
        rpc_write(&pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&psize, sizeof(size_t)) < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(psize, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocHost_v2(void** pp, size_t bytesize)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocHost_v2);
    if (request_id < 0 ||
        rpc_write(&pp, sizeof(void*)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pp, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFreeHost(void* p)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemFreeHost);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(p, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostAlloc);
    if (request_id < 0 ||
        rpc_write(&pp, sizeof(void*)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pp, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostGetDevicePointer_v2);
    if (request_id < 0 ||
        rpc_write(&pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&p, sizeof(void*)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(p, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostGetFlags);
    if (request_id < 0 ||
        rpc_write(&pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(&p, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pFlags, sizeof(unsigned int)) < 0 ||
        rpc_read(p, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocManaged);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetByPCIBusId);
    std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_write(&pciBusId_len, sizeof(std::size_t)) < 0 ||
        rpc_write(pciBusId, pciBusId_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dev, sizeof(CUdevice)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetPCIBusId);
    if (request_id < 0 ||
        rpc_write(&len, sizeof(int)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciBusId, len * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcGetEventHandle);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_write(&event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcOpenEventHandle);
    if (request_id < 0 ||
        rpc_write(&phEvent, sizeof(CUevent)) < 0 ||
        rpc_write(&handle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phEvent, sizeof(CUevent)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcGetMemHandle);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcOpenMemHandle_v2);
    if (request_id < 0 ||
        rpc_write(&pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&handle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcCloseMemHandle);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostRegister_v2);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(void*)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(p, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostUnregister(void* p)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostUnregister);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(p, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&src, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyPeer);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstContext, sizeof(CUcontext)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&srcContext, sizeof(CUcontext)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoH_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(void*)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dstHost, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoD_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoA_v2);
    if (request_id < 0 ||
        rpc_write(&dstArray, sizeof(CUarray)) < 0 ||
        rpc_write(&dstOffset, sizeof(size_t)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoD_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(&srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoH_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(void*)) < 0 ||
        rpc_write(&srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(&srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoA_v2);
    if (request_id < 0 ||
        rpc_write(&dstArray, sizeof(CUarray)) < 0 ||
        rpc_write(&dstOffset, sizeof(size_t)) < 0 ||
        rpc_write(&srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(&srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&src, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyPeerAsync);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstContext, sizeof(CUcontext)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&srcContext, sizeof(CUcontext)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoHAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(void*)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dstHost, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoDAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoHAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(void*)) < 0 ||
        rpc_write(&srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(&srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dstHost, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD8_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&uc, sizeof(unsigned char)) < 0 ||
        rpc_write(&N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD16_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&us, sizeof(unsigned short)) < 0 ||
        rpc_write(&N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD32_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ui, sizeof(unsigned int)) < 0 ||
        rpc_write(&N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D8_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(&uc, sizeof(unsigned char)) < 0 ||
        rpc_write(&Width, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D16_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(&us, sizeof(unsigned short)) < 0 ||
        rpc_write(&Width, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D32_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(&ui, sizeof(unsigned int)) < 0 ||
        rpc_write(&Width, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD8Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&uc, sizeof(unsigned char)) < 0 ||
        rpc_write(&N, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD16Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&us, sizeof(unsigned short)) < 0 ||
        rpc_write(&N, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD32Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ui, sizeof(unsigned int)) < 0 ||
        rpc_write(&N, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D8Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(&uc, sizeof(unsigned char)) < 0 ||
        rpc_write(&Width, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D16Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(&us, sizeof(unsigned short)) < 0 ||
        rpc_write(&Width, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D32Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(&ui, sizeof(unsigned int)) < 0 ||
        rpc_write(&Width, sizeof(size_t)) < 0 ||
        rpc_write(&Height, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetDescriptor_v2);
    if (request_id < 0 ||
        rpc_write(&pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_write(&hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_write(&array, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_write(&mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_write(&array, sizeof(CUarray)) < 0 ||
        rpc_write(&device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_write(&mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(&device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetPlane);
    if (request_id < 0 ||
        rpc_write(&pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_write(&hArray, sizeof(CUarray)) < 0 ||
        rpc_write(&planeIdx, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayDestroy(CUarray hArray)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayDestroy);
    if (request_id < 0 ||
        rpc_write(&hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArray3DGetDescriptor_v2);
    if (request_id < 0 ||
        rpc_write(&pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_write(&hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayGetLevel);
    if (request_id < 0 ||
        rpc_write(&pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_write(&hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(&level, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayDestroy);
    if (request_id < 0 ||
        rpc_write(&hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetHandleForAddressRange);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(void*)) < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&handleType, sizeof(CUmemRangeHandleType)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(handle, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAddressReserve);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&alignment, sizeof(size_t)) < 0 ||
        rpc_write(&addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAddressFree);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRelease);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemMap);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&offset, sizeof(size_t)) < 0 ||
        rpc_write(&handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemMapArrayAsync);
    if (request_id < 0 ||
        rpc_write(&mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemUnmap);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemExportToShareableHandle);
    if (request_id < 0 ||
        rpc_write(&shareableHandle, sizeof(void*)) < 0 ||
        rpc_write(&handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(&handleType, sizeof(CUmemAllocationHandleType)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(shareableHandle, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemImportFromShareableHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(&osHandle, sizeof(void*)) < 0 ||
        rpc_write(&shHandleType, sizeof(CUmemAllocationHandleType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_read(osHandle, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetAllocationPropertiesFromHandle);
    if (request_id < 0 ||
        rpc_write(&prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_write(&handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRetainAllocationHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(&addr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_read(addr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemFreeAsync);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocAsync);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolTrimTo);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&minBytesToKeep, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolSetAttribute);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&attr, sizeof(CUmemPool_attribute)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolGetAttribute);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&attr, sizeof(CUmemPool_attribute)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolGetAccess);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_write(&memPool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&location, sizeof(CUmemLocation)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_read(location, sizeof(CUmemLocation)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolDestroy);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocFromPoolAsync);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&bytesize, sizeof(size_t)) < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolExportToShareableHandle);
    if (request_id < 0 ||
        rpc_write(&handle_out, sizeof(void*)) < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&handleType, sizeof(CUmemAllocationHandleType)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(handle_out, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolImportFromShareableHandle);
    if (request_id < 0 ||
        rpc_write(&pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&handle, sizeof(void*)) < 0 ||
        rpc_write(&handleType, sizeof(CUmemAllocationHandleType)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_read(handle, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolExportPointer);
    if (request_id < 0 ||
        rpc_write(&shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolImportPointer);
    if (request_id < 0 ||
        rpc_write(&ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(&shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuPointerGetAttribute);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(void*)) < 0 ||
        rpc_write(&attribute, sizeof(CUpointer_attribute)) < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPrefetchAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_write(&dstDevice, sizeof(CUdevice)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAdvise);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_write(&advice, sizeof(CUmem_advise)) < 0 ||
        rpc_write(&device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRangeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(void*)) < 0 ||
        rpc_write(&dataSize, sizeof(size_t)) < 0 ||
        rpc_write(&attribute, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_write(&devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRangeGetAttributes);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(void*)) < 0 ||
        rpc_write(&dataSizes, sizeof(size_t)) < 0 ||
        rpc_write(&attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_write(&numAttributes, sizeof(size_t)) < 0 ||
        rpc_write(&devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(void*)) < 0 ||
        rpc_read(dataSizes, sizeof(size_t)) < 0 ||
        rpc_read(attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuPointerGetAttributes);
    if (request_id < 0 ||
        rpc_write(&numAttributes, sizeof(unsigned int)) < 0 ||
        rpc_write(&attributes, sizeof(CUpointer_attribute)) < 0 ||
        rpc_write(&data, sizeof(void*)) < 0 ||
        rpc_write(&ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(attributes, sizeof(CUpointer_attribute)) < 0 ||
        rpc_read(data, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamCreate);
    if (request_id < 0 ||
        rpc_write(&phStream, sizeof(CUstream)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phStream, sizeof(CUstream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamCreateWithPriority);
    if (request_id < 0 ||
        rpc_write(&phStream, sizeof(CUstream)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_write(&priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phStream, sizeof(CUstream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetPriority(CUstream hStream, int* priority)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetPriority);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(priority, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetFlags);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetId);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&streamId, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(streamId, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetCtx);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&pctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWaitEvent);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamAddCallback);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&callback, sizeof(CUstreamCallback)) < 0 ||
        rpc_write(&userData, sizeof(void*)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(userData, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamBeginCapture_v2);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuThreadExchangeStreamCaptureMode);
    if (request_id < 0 ||
        rpc_write(&mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamEndCapture);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&phGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamIsCapturing);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamUpdateCaptureDependencies);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamAttachMemAsync);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&length, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamQuery(CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamQuery);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamSynchronize(CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamSynchronize);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamDestroy_v2(CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamDestroy_v2);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCopyAttributes(CUstream dst, CUstream src)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(CUstream)) < 0 ||
        rpc_write(&src, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&attr, sizeof(CUstreamAttrID)) < 0 ||
        rpc_write(&value_out, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value_out, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventCreate);
    if (request_id < 0 ||
        rpc_write(&phEvent, sizeof(CUevent)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phEvent, sizeof(CUevent)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventRecord);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventRecordWithFlags);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventQuery(CUevent hEvent)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventQuery);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventSynchronize(CUevent hEvent)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventSynchronize);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventDestroy_v2(CUevent hEvent)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventDestroy_v2);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventElapsedTime);
    if (request_id < 0 ||
        rpc_write(&pMilliseconds, sizeof(float)) < 0 ||
        rpc_write(&hStart, sizeof(CUevent)) < 0 ||
        rpc_write(&hEnd, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pMilliseconds, sizeof(float)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDestroyExternalMemory);
    if (request_id < 0 ||
        rpc_write(&extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDestroyExternalSemaphore);
    if (request_id < 0 ||
        rpc_write(&extSem, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWaitValue32_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(CUstream)) < 0 ||
        rpc_write(&addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&value, sizeof(cuuint32_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWaitValue64_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(CUstream)) < 0 ||
        rpc_write(&addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&value, sizeof(cuuint64_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWriteValue32_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(CUstream)) < 0 ||
        rpc_write(&addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&value, sizeof(cuuint32_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWriteValue64_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(CUstream)) < 0 ||
        rpc_write(&addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&value, sizeof(cuuint64_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamBatchMemOp_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(CUstream)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_write(&paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncGetAttribute);
    if (request_id < 0 ||
        rpc_write(&pi, sizeof(int)) < 0 ||
        rpc_write(&attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pi, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&config, sizeof(CUfunc_cache)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&config, sizeof(CUsharedconfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncGetModule);
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hmod, sizeof(CUmodule)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchKernel);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(CUfunction)) < 0 ||
        rpc_write(&gridDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(&gridDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(&gridDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(&blockDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(&blockDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(&blockDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(&sharedMemBytes, sizeof(unsigned int)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&kernelParams, sizeof(void*)) < 0 ||
        rpc_write(&extra, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchCooperativeKernel);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(CUfunction)) < 0 ||
        rpc_write(&gridDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(&gridDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(&gridDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(&blockDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(&blockDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(&blockDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(&sharedMemBytes, sizeof(unsigned int)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&kernelParams, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(kernelParams, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchCooperativeKernelMultiDevice);
    if (request_id < 0 ||
        rpc_write(&launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_write(&numDevices, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchHostFunc);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_write(&fn, sizeof(CUhostFn)) < 0 ||
        rpc_write(&userData, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(userData, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetBlockShape);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&x, sizeof(int)) < 0 ||
        rpc_write(&y, sizeof(int)) < 0 ||
        rpc_write(&z, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetSharedSize);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&bytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetSize);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&numbytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSeti);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_write(&value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetf);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_write(&value, sizeof(float)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetv);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&numbytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunch(CUfunction f)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunch);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchGrid);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(CUfunction)) < 0 ||
        rpc_write(&grid_width, sizeof(int)) < 0 ||
        rpc_write(&grid_height, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchGridAsync);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(CUfunction)) < 0 ||
        rpc_write(&grid_width, sizeof(int)) < 0 ||
        rpc_write(&grid_height, sizeof(int)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetTexRef);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&texunit, sizeof(int)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphCreate);
    if (request_id < 0 ||
        rpc_write(&phGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeGetParams_v2);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemcpyNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemsetNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphHostNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphChildGraphNodeGetGraph);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&phGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventRecordNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&event_out, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event_out, sizeof(CUevent)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventWaitNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&event_out, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event_out, sizeof(CUevent)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExternalSemaphoresSignalNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExternalSemaphoresWaitNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphBatchMemOpNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemAllocNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemFreeNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGraphMemTrim(CUdevice device)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGraphMemTrim);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(CUdevice)) < 0 ||
        rpc_write(&attr, sizeof(CUgraphMem_attribute)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceSetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(CUdevice)) < 0 ||
        rpc_write(&attr, sizeof(CUgraphMem_attribute)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphClone);
    if (request_id < 0 ||
        rpc_write(&phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_write(&originalGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeFindInClone);
    if (request_id < 0 ||
        rpc_write(&phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&hOriginalNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&hClonedGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetType);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphGetNodes);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&nodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&numNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(numNodes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphGetRootNodes);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&numRootNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(numRootNodes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphGetEdges);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&from, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&to, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&numEdges, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(from, sizeof(CUgraphNode)) < 0 ||
        rpc_read(to, sizeof(CUgraphNode)) < 0 ||
        rpc_read(numEdges, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetDependencies);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_read(numDependencies, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetDependentNodes);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&dependentNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&numDependentNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dependentNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(numDependentNodes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDestroyNode(CUgraphNode hNode)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphDestroyNode);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphInstantiateWithFlags);
    if (request_id < 0 ||
        rpc_write(&phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphInstantiateWithParams);
    if (request_id < 0 ||
        rpc_write(&phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_read(instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecGetFlags);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&flags, sizeof(cuuint64_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(cuuint64_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecChildGraphNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&childGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeSetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphUpload);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphLaunch);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecDestroy);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDestroy(CUgraph hGraph)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphDestroy);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecUpdate_v2);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(&resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&src, sizeof(CUgraphNode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(&attr, sizeof(CUkernelNodeAttrID)) < 0 ||
        rpc_write(&value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuUserObjectCreate);
    if (request_id < 0 ||
        rpc_write(&object_out, sizeof(CUuserObject)) < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&destroy, sizeof(CUhostFn)) < 0 ||
        rpc_write(&initialRefcount, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(object_out, sizeof(CUuserObject)) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectRetain(CUuserObject object, unsigned int count)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuUserObjectRetain);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(CUuserObject)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectRelease(CUuserObject object, unsigned int count)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuUserObjectRelease);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(CUuserObject)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphRetainUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(CUgraph)) < 0 ||
        rpc_write(&object, sizeof(CUuserObject)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphReleaseUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(CUgraph)) < 0 ||
        rpc_write(&object, sizeof(CUuserObject)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor);
    if (request_id < 0 ||
        rpc_write(&numBlocks, sizeof(int)) < 0 ||
        rpc_write(&func, sizeof(CUfunction)) < 0 ||
        rpc_write(&blockSize, sizeof(int)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numBlocks, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    if (request_id < 0 ||
        rpc_write(&numBlocks, sizeof(int)) < 0 ||
        rpc_write(&func, sizeof(CUfunction)) < 0 ||
        rpc_write(&blockSize, sizeof(int)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numBlocks, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyAvailableDynamicSMemPerBlock);
    if (request_id < 0 ||
        rpc_write(&dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_write(&func, sizeof(CUfunction)) < 0 ||
        rpc_write(&numBlocks, sizeof(int)) < 0 ||
        rpc_write(&blockSize, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetArray);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&hArray, sizeof(CUarray)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetAddress_v2);
    if (request_id < 0 ||
        rpc_write(&ByteOffset, sizeof(size_t)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&bytes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ByteOffset, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetFormat);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&fmt, sizeof(CUarray_format)) < 0 ||
        rpc_write(&NumPackedComponents, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetAddressMode);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&dim, sizeof(int)) < 0 ||
        rpc_write(&am, sizeof(CUaddress_mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetFilterMode);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&fm, sizeof(CUfilter_mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmapFilterMode);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&fm, sizeof(CUfilter_mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmapLevelBias);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&bias, sizeof(float)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmapLevelClamp);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&minMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(&maxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMaxAnisotropy);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&maxAniso, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetBorderColor);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&pBorderColor, sizeof(float)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pBorderColor, sizeof(float)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetFlags);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetAddress_v2);
    if (request_id < 0 ||
        rpc_write(&pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetArray);
    if (request_id < 0 ||
        rpc_write(&phArray, sizeof(CUarray)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phArray, sizeof(CUarray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetAddressMode);
    if (request_id < 0 ||
        rpc_write(&pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(&dim, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetFilterMode);
    if (request_id < 0 ||
        rpc_write(&pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetFormat);
    if (request_id < 0 ||
        rpc_write(&pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_write(&pNumChannels, sizeof(int)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_read(pNumChannels, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmapFilterMode);
    if (request_id < 0 ||
        rpc_write(&pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmapLevelBias);
    if (request_id < 0 ||
        rpc_write(&pbias, sizeof(float)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pbias, sizeof(float)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmapLevelClamp);
    if (request_id < 0 ||
        rpc_write(&pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(&pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_read(pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMaxAnisotropy);
    if (request_id < 0 ||
        rpc_write(&pmaxAniso, sizeof(int)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pmaxAniso, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetBorderColor);
    if (request_id < 0 ||
        rpc_write(&pBorderColor, sizeof(float)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pBorderColor, sizeof(float)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetFlags);
    if (request_id < 0 ||
        rpc_write(&pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pFlags, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefCreate(CUtexref* pTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefCreate);
    if (request_id < 0 ||
        rpc_write(&pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefDestroy(CUtexref hTexRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefDestroy);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfRefSetArray);
    if (request_id < 0 ||
        rpc_write(&hSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_write(&hArray, sizeof(CUarray)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfRefGetArray);
    if (request_id < 0 ||
        rpc_write(&phArray, sizeof(CUarray)) < 0 ||
        rpc_write(&hSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(phArray, sizeof(CUarray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectDestroy(CUtexObject texObject)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectDestroy);
    if (request_id < 0 ||
        rpc_write(&texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectGetResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(&texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectGetTextureDesc);
    if (request_id < 0 ||
        rpc_write(&pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_write(&texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectGetResourceViewDesc);
    if (request_id < 0 ||
        rpc_write(&pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_write(&texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfObjectDestroy);
    if (request_id < 0 ||
        rpc_write(&surfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfObjectGetResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(&surfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTensorMapReplaceAddress);
    if (request_id < 0 ||
        rpc_write(&tensorMap, sizeof(CUtensorMap)) < 0 ||
        rpc_write(&globalAddress, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(tensorMap, sizeof(CUtensorMap)) < 0 ||
        rpc_read(globalAddress, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceCanAccessPeer);
    if (request_id < 0 ||
        rpc_write(&canAccessPeer, sizeof(int)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_write(&peerDev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(canAccessPeer, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxEnablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerContext, sizeof(CUcontext)) < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxDisablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerContext, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetP2PAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&attrib, sizeof(CUdevice_P2PAttribute)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdevice)) < 0 ||
        rpc_write(&dstDevice, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsUnregisterResource);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsSubResourceGetMappedArray);
    if (request_id < 0 ||
        rpc_write(&pArray, sizeof(CUarray)) < 0 ||
        rpc_write(&resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(&arrayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(&mipLevel, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pArray, sizeof(CUarray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsResourceGetMappedMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(&resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsResourceGetMappedPointer_v2);
    if (request_id < 0 ||
        rpc_write(&pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&pSize, sizeof(size_t)) < 0 ||
        rpc_write(&resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(pSize, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsResourceSetMapFlags_v2);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsMapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_write(&resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsUnmapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_write(&resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

cudaError_t cudaDeviceReset()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceReset);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSynchronize()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSynchronize);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_write(&value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetLimit);
    if (request_id < 0 ||
        rpc_write(&pValue, sizeof(size_t)) < 0 ||
        rpc_write(&limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pValue, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetStreamPriorityRange);
    if (request_id < 0 ||
        rpc_write(&leastPriority, sizeof(int)) < 0 ||
        rpc_write(&greatestPriority, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(leastPriority, sizeof(int)) < 0 ||
        rpc_read(greatestPriority, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&cacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&pConfig, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pConfig, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetPCIBusId);
    if (request_id < 0 ||
        rpc_write(&pciBusId, sizeof(char)) < 0 ||
        rpc_write(&len, sizeof(int)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciBusId, sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcGetEventHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcOpenEventHandle);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(&handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcGetMemHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(cudaIpcMemHandle_t)) < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(handle, sizeof(cudaIpcMemHandle_t)) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcOpenMemHandle);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&handle, sizeof(cudaIpcMemHandle_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcCloseMemHandle);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceFlushGPUDirectRDMAWrites);
    if (request_id < 0 ||
        rpc_write(&target, sizeof(enum cudaFlushGPUDirectRDMAWritesTarget)) < 0 ||
        rpc_write(&scope, sizeof(enum cudaFlushGPUDirectRDMAWritesScope)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadExit()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadExit);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadSynchronize()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadSynchronize);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadSetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_write(&value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadGetLimit(size_t* pValue, enum cudaLimit limit)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadGetLimit);
    if (request_id < 0 ||
        rpc_write(&pValue, sizeof(size_t)) < 0 ||
        rpc_write(&limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pValue, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadGetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&cacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetLastError()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetLastError);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaPeekAtLastError()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaPeekAtLastError);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDeviceCount(int* count)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDeviceCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp* prop, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDeviceProperties_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(prop, sizeof(struct cudaDeviceProp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&attr, sizeof(enum cudaDeviceAttr)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetDefaultMemPool);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetMemPool);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetMemPool);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetNvSciSyncAttributes);
    if (request_id < 0 ||
        rpc_write(&nvSciSyncAttrList, sizeof(void*)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_write(&flags, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nvSciSyncAttrList, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetP2PAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&attr, sizeof(enum cudaDeviceP2PAttr)) < 0 ||
        rpc_write(&srcDevice, sizeof(int)) < 0 ||
        rpc_write(&dstDevice, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaInitDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_write(&deviceFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDevice(int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDevice(int* device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetValidDevices(int* device_arr, int len)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetValidDevices);
    if (request_id < 0 ||
        rpc_write(&device_arr, sizeof(int)) < 0 ||
        rpc_write(&len, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device_arr, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDeviceFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDeviceFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCreate);
    if (request_id < 0 ||
        rpc_write(&pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCreateWithFlags);
    if (request_id < 0 ||
        rpc_write(&pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCreateWithPriority);
    if (request_id < 0 ||
        rpc_write(&pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_write(&priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetPriority);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(priority, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetFlags);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetId);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&streamId, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(streamId, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaCtxResetPersistingL2Cache()
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaCtxResetPersistingL2Cache);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&src, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&attr, sizeof(cudaLaunchAttributeID)) < 0 ||
        rpc_write(&value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamDestroy);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamWaitEvent);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamAddCallback);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&callback, sizeof(cudaStreamCallback_t)) < 0 ||
        rpc_write(&userData, sizeof(void*)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(userData, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamSynchronize);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamQuery);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamAttachMemAsync);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&length, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamBeginCapture);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadExchangeStreamCaptureMode);
    if (request_id < 0 ||
        rpc_write(&mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamEndCapture);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamIsCapturing);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetCaptureInfo_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(captureStatus_out, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_read(id_out, sizeof(unsigned long long)) < 0 ||
        rpc_read(graph_out, sizeof(cudaGraph_t)) < 0 ||
        rpc_read(numDependencies_out, sizeof(size_t)) < 0 ||
        rpc_read(dependencies_out, *numDependencies_out * sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamUpdateCaptureDependencies);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(dependencies, numDependencies * sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventCreate);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventCreateWithFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventRecord);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventRecordWithFlags);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventQuery);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventSynchronize);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventDestroy);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventElapsedTime);
    if (request_id < 0 ||
        rpc_write(&start, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(&end, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ms, sizeof(float)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroyExternalMemory);
    if (request_id < 0 ||
        rpc_write(&extMem, sizeof(cudaExternalMemory_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroyExternalSemaphore);
    if (request_id < 0 ||
        rpc_write(&extSem, sizeof(cudaExternalSemaphore_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchCooperativeKernelMultiDevice);
    if (request_id < 0 ||
        rpc_write(&launchParamsList, sizeof(struct cudaLaunchParams)) < 0 ||
        rpc_write(&numDevices, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(launchParamsList, sizeof(struct cudaLaunchParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDoubleForDevice(double* d)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDoubleForDevice);
    if (request_id < 0 ||
        rpc_write(&d, sizeof(double)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(d, sizeof(double)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDoubleForHost(double* d)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDoubleForHost);
    if (request_id < 0 ||
        rpc_write(&d, sizeof(double)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(d, sizeof(double)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchHostFunc);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(&fn, sizeof(cudaHostFn_t)) < 0 ||
        rpc_write(&userData, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(userData, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocManaged);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMalloc);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocHost(void** ptr, size_t size)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocHost);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocPitch);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&pitch, sizeof(size_t)) < 0 ||
        rpc_write(&width, sizeof(size_t)) < 0 ||
        rpc_write(&height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_read(pitch, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFree(void* devPtr)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFree);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeHost(void* ptr)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeHost);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeArray(cudaArray_t array)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeArray);
    if (request_id < 0 ||
        rpc_write(&array, sizeof(cudaArray_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostAlloc);
    if (request_id < 0 ||
        rpc_write(&pHost, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pHost, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostRegister);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaHostUnregister(void* ptr)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostUnregister);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostGetDevicePointer);
    if (request_id < 0 ||
        rpc_write(&pDevice, sizeof(void*)) < 0 ||
        rpc_write(&pHost, sizeof(void*)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDevice, sizeof(void*)) < 0 ||
        rpc_read(pHost, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostGetFlags);
    if (request_id < 0 ||
        rpc_write(&pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(&pHost, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pFlags, sizeof(unsigned int)) < 0 ||
        rpc_read(pHost, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMalloc3D);
    if (request_id < 0 ||
        rpc_write(&pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_write(&extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetMipmappedArrayLevel);
    if (request_id < 0 ||
        rpc_write(&levelArray, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&mipmappedArray, sizeof(cudaMipmappedArray_const_t)) < 0 ||
        rpc_write(&level, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(levelArray, sizeof(cudaArray_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemGetInfo);
    if (request_id < 0 ||
        rpc_write(&free, sizeof(size_t)) < 0 ||
        rpc_write(&total, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(free, sizeof(size_t)) < 0 ||
        rpc_read(total, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetInfo);
    if (request_id < 0 ||
        rpc_write(&desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_write(&extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_write(&array, sizeof(cudaArray_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_read(extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_read(flags, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetPlane);
    if (request_id < 0 ||
        rpc_write(&pPlaneArray, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&hArray, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&planeIdx, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pPlaneArray, sizeof(cudaArray_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_write(&array, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMipmappedArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_write(&mipmap, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaArray_t array)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_write(&array, sizeof(cudaArray_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMipmappedArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_write(&mipmap, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DFromArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(void*)) < 0 ||
        rpc_write(&dpitch, sizeof(size_t)) < 0 ||
        rpc_write(&src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(&wOffset, sizeof(size_t)) < 0 ||
        rpc_write(&hOffset, sizeof(size_t)) < 0 ||
        rpc_write(&width, sizeof(size_t)) < 0 ||
        rpc_write(&height, sizeof(size_t)) < 0 ||
        rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dst, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DArrayToArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&wOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(&hOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(&src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(&wOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(&hOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(&width, sizeof(size_t)) < 0 ||
        rpc_write(&height, sizeof(size_t)) < 0 ||
        rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DFromArrayAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(void*)) < 0 ||
        rpc_write(&dpitch, sizeof(size_t)) < 0 ||
        rpc_write(&src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(&wOffset, sizeof(size_t)) < 0 ||
        rpc_write(&hOffset, sizeof(size_t)) < 0 ||
        rpc_write(&width, sizeof(size_t)) < 0 ||
        rpc_write(&height, sizeof(size_t)) < 0 ||
        rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dst, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset2D);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&pitch, sizeof(size_t)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&width, sizeof(size_t)) < 0 ||
        rpc_write(&height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset3D);
    if (request_id < 0 ||
        rpc_write(&pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemsetAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset2DAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&pitch, sizeof(size_t)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&width, sizeof(size_t)) < 0 ||
        rpc_write(&height, sizeof(size_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset3DAsync);
    if (request_id < 0 ||
        rpc_write(&pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_write(&extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyFromArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(void*)) < 0 ||
        rpc_write(&src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(&wOffset, sizeof(size_t)) < 0 ||
        rpc_write(&hOffset, sizeof(size_t)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dst, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyArrayToArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&wOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(&hOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(&src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(&wOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(&hOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyFromArrayAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(void*)) < 0 ||
        rpc_write(&src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(&wOffset, sizeof(size_t)) < 0 ||
        rpc_write(&hOffset, sizeof(size_t)) < 0 ||
        rpc_write(&count, sizeof(size_t)) < 0 ||
        rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dst, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolTrimTo);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&minBytesToKeep, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolSetAttribute);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&attr, sizeof(enum cudaMemPoolAttr)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolGetAttribute);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&attr, sizeof(enum cudaMemPoolAttr)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags* flags, cudaMemPool_t memPool, struct cudaMemLocation* location)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolGetAccess);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(enum cudaMemAccessFlags)) < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&location, sizeof(struct cudaMemLocation)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(enum cudaMemAccessFlags)) < 0 ||
        rpc_read(location, sizeof(struct cudaMemLocation)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolDestroy);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocFromPoolAsync);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, enum cudaMemAllocationHandleType handleType, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolExportToShareableHandle);
    if (request_id < 0 ||
        rpc_write(&shareableHandle, sizeof(void*)) < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&handleType, sizeof(enum cudaMemAllocationHandleType)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(shareableHandle, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolImportFromShareableHandle);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&shareableHandle, sizeof(void*)) < 0 ||
        rpc_write(&handleType, sizeof(enum cudaMemAllocationHandleType)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_read(shareableHandle, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData* exportData, void* ptr)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolExportPointer);
    if (request_id < 0 ||
        rpc_write(&exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData* exportData)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolImportPointer);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(&exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_read(exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceCanAccessPeer);
    if (request_id < 0 ||
        rpc_write(&canAccessPeer, sizeof(int)) < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_write(&peerDevice, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(canAccessPeer, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceEnablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerDevice, sizeof(int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceDisablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerDevice, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsUnregisterResource);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsResourceSetMapFlags);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsMapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(int)) < 0 ||
        rpc_write(&resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsUnmapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(int)) < 0 ||
        rpc_write(&resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsResourceGetMappedPointer);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(void*)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(&resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(devPtr, sizeof(void*)) < 0 ||
        rpc_read(size, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsSubResourceGetMappedArray);
    if (request_id < 0 ||
        rpc_write(&array, sizeof(cudaArray_t)) < 0 ||
        rpc_write(&resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(&arrayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(&mipLevel, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(array, sizeof(cudaArray_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsResourceGetMappedMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_write(&resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc* desc, cudaArray_const_t array)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetChannelDesc);
    if (request_id < 0 ||
        rpc_write(&desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_write(&array, sizeof(cudaArray_const_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroyTextureObject);
    if (request_id < 0 ||
        rpc_write(&texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaTextureObject_t texObject)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetTextureObjectResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_write(&texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetTextureObjectTextureDesc);
    if (request_id < 0 ||
        rpc_write(&pTexDesc, sizeof(struct cudaTextureDesc)) < 0 ||
        rpc_write(&texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pTexDesc, sizeof(struct cudaTextureDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetTextureObjectResourceViewDesc);
    if (request_id < 0 ||
        rpc_write(&pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0 ||
        rpc_write(&texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroySurfaceObject);
    if (request_id < 0 ||
        rpc_write(&surfObject, sizeof(cudaSurfaceObject_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetSurfaceObjectResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_write(&surfObject, sizeof(cudaSurfaceObject_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDriverGetVersion(int* driverVersion)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDriverGetVersion);
    if (request_id < 0 ||
        rpc_write(&driverVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(driverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaRuntimeGetVersion);
    if (request_id < 0 ||
        rpc_write(&runtimeVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(runtimeVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphCreate);
    if (request_id < 0 ||
        rpc_write(&pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&hSrc, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&hDst, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&attr, sizeof(cudaLaunchAttributeID)) < 0 ||
        rpc_write(&value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms* pNodeParams)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemcpyNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams* pNodeParams)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemsetNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNodeParams, sizeof(struct cudaMemsetParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNodeParams, sizeof(struct cudaMemsetParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams* pNodeParams)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphHostNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNodeParams, sizeof(struct cudaHostNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNodeParams, sizeof(struct cudaHostNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphChildGraphNodeGetGraph);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventRecordNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventWaitNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams* params_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExternalSemaphoresSignalNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams* params_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExternalSemaphoresWaitNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams* params_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemAllocNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&params_out, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(params_out, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemFreeNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&dptr_out, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr_out, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGraphMemTrim(int device)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGraphMemTrim);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_write(&attr, sizeof(enum cudaGraphMemAttributeType)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(int)) < 0 ||
        rpc_write(&attr, sizeof(enum cudaGraphMemAttributeType)) < 0 ||
        rpc_write(&value, sizeof(void*)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphClone);
    if (request_id < 0 ||
        rpc_write(&pGraphClone, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&originalGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraphClone, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeFindInClone);
    if (request_id < 0 ||
        rpc_write(&pNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&originalNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&clonedGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* pType)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetType);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pType, sizeof(enum cudaGraphNodeType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pType, sizeof(enum cudaGraphNodeType)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphGetNodes);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&nodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&numNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(numNodes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphGetRootNodes);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&pRootNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNumRootNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pRootNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(pNumRootNodes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphGetEdges);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&from, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&to, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&numEdges, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(from, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(to, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(numEdges, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetDependencies);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pDependencies, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNumDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDependencies, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(pNumDependencies, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetDependentNodes);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pDependentNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&pNumDependentNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDependentNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(pNumDependentNodes, sizeof(size_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphDestroyNode);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphInstantiate);
    if (request_id < 0 ||
        rpc_write(&pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphInstantiateWithFlags);
    if (request_id < 0 ||
        rpc_write(&pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphInstantiateWithParams);
    if (request_id < 0 ||
        rpc_write(&pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_read(instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecGetFlags);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(flags, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecChildGraphNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&childGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeSetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(&isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecUpdate);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&hGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphUpload);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphLaunch);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(&stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecDestroy);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphDestroy);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaUserObjectCreate);
    if (request_id < 0 ||
        rpc_write(&object_out, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(&ptr, sizeof(void*)) < 0 ||
        rpc_write(&destroy, sizeof(cudaHostFn_t)) < 0 ||
        rpc_write(&initialRefcount, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(object_out, sizeof(cudaUserObject_t)) < 0 ||
        rpc_read(ptr, sizeof(void*)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaUserObjectRetain);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaUserObjectRelease);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphRetainUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphReleaseUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(&object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

std::unordered_map<std::string, void *> functionMap = {
    {"nvmlInit", (void *)nvmlInit_v2},
    {"nvmlInit_v2", (void *)nvmlInit_v2},
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
    {"nvmlGpuInstanceDestroy", (void *)nvmlGpuInstanceDestroy},
    {"nvmlDeviceGetGpuInstances", (void *)nvmlDeviceGetGpuInstances},
    {"nvmlDeviceGetGpuInstanceById", (void *)nvmlDeviceGetGpuInstanceById},
    {"nvmlGpuInstanceGetInfo", (void *)nvmlGpuInstanceGetInfo},
    {"nvmlGpuInstanceGetComputeInstanceProfileInfo", (void *)nvmlGpuInstanceGetComputeInstanceProfileInfo},
    {"nvmlGpuInstanceGetComputeInstanceProfileInfoV", (void *)nvmlGpuInstanceGetComputeInstanceProfileInfoV},
    {"nvmlGpuInstanceGetComputeInstanceRemainingCapacity", (void *)nvmlGpuInstanceGetComputeInstanceRemainingCapacity},
    {"nvmlGpuInstanceGetComputeInstancePossiblePlacements", (void *)nvmlGpuInstanceGetComputeInstancePossiblePlacements},
    {"nvmlGpuInstanceCreateComputeInstance", (void *)nvmlGpuInstanceCreateComputeInstance},
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
    {"cuInit", (void *)cuInit},
    {"cuDriverGetVersion", (void *)cuDriverGetVersion},
    {"cuDeviceGet", (void *)cuDeviceGet},
    {"cuDeviceGetCount", (void *)cuDeviceGetCount},
    {"cuDeviceGetName", (void *)cuDeviceGetName},
    {"cuDeviceGetUuid", (void *)cuDeviceGetUuid},
    {"cuDeviceGetUuid_v2", (void *)cuDeviceGetUuid_v2},
    {"cuDeviceGetLuid", (void *)cuDeviceGetLuid},
    {"cuDeviceTotalMem_v2", (void *)cuDeviceTotalMem_v2},
    {"cuDeviceGetTexture1DLinearMaxWidth", (void *)cuDeviceGetTexture1DLinearMaxWidth},
    {"cuDeviceGetAttribute", (void *)cuDeviceGetAttribute},
    {"cuDeviceSetMemPool", (void *)cuDeviceSetMemPool},
    {"cuDeviceGetMemPool", (void *)cuDeviceGetMemPool},
    {"cuDeviceGetDefaultMemPool", (void *)cuDeviceGetDefaultMemPool},
    {"cuDeviceGetExecAffinitySupport", (void *)cuDeviceGetExecAffinitySupport},
    {"cuFlushGPUDirectRDMAWrites", (void *)cuFlushGPUDirectRDMAWrites},
    {"cuDeviceGetProperties", (void *)cuDeviceGetProperties},
    {"cuDeviceComputeCapability", (void *)cuDeviceComputeCapability},
    {"cuDevicePrimaryCtxRetain", (void *)cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxGetState", (void *)cuDevicePrimaryCtxGetState},
    {"cuDevicePrimaryCtxReset_v2", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuCtxCreate_v2", (void *)cuCtxCreate_v2},
    {"cuCtxCreate_v3", (void *)cuCtxCreate_v3},
    {"cuCtxDestroy_v2", (void *)cuCtxDestroy_v2},
    {"cuCtxPushCurrent_v2", (void *)cuCtxPushCurrent_v2},
    {"cuCtxPopCurrent_v2", (void *)cuCtxPopCurrent_v2},
    {"cuCtxSetCurrent", (void *)cuCtxSetCurrent},
    {"cuCtxGetCurrent", (void *)cuCtxGetCurrent},
    {"cuCtxGetDevice", (void *)cuCtxGetDevice},
    {"cuCtxGetFlags", (void *)cuCtxGetFlags},
    {"cuCtxGetId", (void *)cuCtxGetId},
    {"cuCtxSynchronize", (void *)cuCtxSynchronize},
    {"cuCtxSetLimit", (void *)cuCtxSetLimit},
    {"cuCtxGetLimit", (void *)cuCtxGetLimit},
    {"cuCtxGetCacheConfig", (void *)cuCtxGetCacheConfig},
    {"cuCtxSetCacheConfig", (void *)cuCtxSetCacheConfig},
    {"cuCtxGetSharedMemConfig", (void *)cuCtxGetSharedMemConfig},
    {"cuCtxSetSharedMemConfig", (void *)cuCtxSetSharedMemConfig},
    {"cuCtxGetApiVersion", (void *)cuCtxGetApiVersion},
    {"cuCtxGetStreamPriorityRange", (void *)cuCtxGetStreamPriorityRange},
    {"cuCtxResetPersistingL2Cache", (void *)cuCtxResetPersistingL2Cache},
    {"cuCtxGetExecAffinity", (void *)cuCtxGetExecAffinity},
    {"cuCtxAttach", (void *)cuCtxAttach},
    {"cuCtxDetach", (void *)cuCtxDetach},
    {"cuModuleLoad", (void *)cuModuleLoad},
    {"cuModuleUnload", (void *)cuModuleUnload},
    {"cuModuleGetLoadingMode", (void *)cuModuleGetLoadingMode},
    {"cuModuleGetFunction", (void *)cuModuleGetFunction},
    {"cuModuleGetGlobal_v2", (void *)cuModuleGetGlobal_v2},
    {"cuLinkCreate_v2", (void *)cuLinkCreate_v2},
    {"cuLinkAddData_v2", (void *)cuLinkAddData_v2},
    {"cuLinkAddFile_v2", (void *)cuLinkAddFile_v2},
    {"cuLinkComplete", (void *)cuLinkComplete},
    {"cuLinkDestroy", (void *)cuLinkDestroy},
    {"cuModuleGetTexRef", (void *)cuModuleGetTexRef},
    {"cuModuleGetSurfRef", (void *)cuModuleGetSurfRef},
    {"cuLibraryLoadFromFile", (void *)cuLibraryLoadFromFile},
    {"cuLibraryUnload", (void *)cuLibraryUnload},
    {"cuLibraryGetKernel", (void *)cuLibraryGetKernel},
    {"cuLibraryGetModule", (void *)cuLibraryGetModule},
    {"cuKernelGetFunction", (void *)cuKernelGetFunction},
    {"cuLibraryGetGlobal", (void *)cuLibraryGetGlobal},
    {"cuLibraryGetManaged", (void *)cuLibraryGetManaged},
    {"cuLibraryGetUnifiedFunction", (void *)cuLibraryGetUnifiedFunction},
    {"cuKernelGetAttribute", (void *)cuKernelGetAttribute},
    {"cuKernelSetAttribute", (void *)cuKernelSetAttribute},
    {"cuKernelSetCacheConfig", (void *)cuKernelSetCacheConfig},
    {"cuMemGetInfo_v2", (void *)cuMemGetInfo_v2},
    {"cuMemAlloc_v2", (void *)cuMemAlloc_v2},
    {"cuMemAllocPitch_v2", (void *)cuMemAllocPitch_v2},
    {"cuMemFree_v2", (void *)cuMemFree_v2},
    {"cuMemGetAddressRange_v2", (void *)cuMemGetAddressRange_v2},
    {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
    {"cuMemFreeHost", (void *)cuMemFreeHost},
    {"cuMemHostAlloc", (void *)cuMemHostAlloc},
    {"cuMemHostGetDevicePointer_v2", (void *)cuMemHostGetDevicePointer_v2},
    {"cuMemHostGetFlags", (void *)cuMemHostGetFlags},
    {"cuMemAllocManaged", (void *)cuMemAllocManaged},
    {"cuDeviceGetByPCIBusId", (void *)cuDeviceGetByPCIBusId},
    {"cuDeviceGetPCIBusId", (void *)cuDeviceGetPCIBusId},
    {"cuIpcGetEventHandle", (void *)cuIpcGetEventHandle},
    {"cuIpcOpenEventHandle", (void *)cuIpcOpenEventHandle},
    {"cuIpcGetMemHandle", (void *)cuIpcGetMemHandle},
    {"cuIpcOpenMemHandle_v2", (void *)cuIpcOpenMemHandle_v2},
    {"cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle},
    {"cuMemHostRegister_v2", (void *)cuMemHostRegister_v2},
    {"cuMemHostUnregister", (void *)cuMemHostUnregister},
    {"cuMemcpy", (void *)cuMemcpy},
    {"cuMemcpyPeer", (void *)cuMemcpyPeer},
    {"cuMemcpyDtoH_v2", (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoD_v2", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyAsync", (void *)cuMemcpyAsync},
    {"cuMemcpyPeerAsync", (void *)cuMemcpyPeerAsync},
    {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
    {"cuMemcpyDtoDAsync_v2", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemcpyAtoHAsync_v2", (void *)cuMemcpyAtoHAsync_v2},
    {"cuMemsetD8_v2", (void *)cuMemsetD8_v2},
    {"cuMemsetD16_v2", (void *)cuMemsetD16_v2},
    {"cuMemsetD32_v2", (void *)cuMemsetD32_v2},
    {"cuMemsetD2D8_v2", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D16_v2", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32_v2", (void *)cuMemsetD2D32_v2},
    {"cuMemsetD8Async", (void *)cuMemsetD8Async},
    {"cuMemsetD16Async", (void *)cuMemsetD16Async},
    {"cuMemsetD32Async", (void *)cuMemsetD32Async},
    {"cuMemsetD2D8Async", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D16Async", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D32Async", (void *)cuMemsetD2D32Async},
    {"cuArrayGetDescriptor_v2", (void *)cuArrayGetDescriptor_v2},
    {"cuArrayGetSparseProperties", (void *)cuArrayGetSparseProperties},
    {"cuMipmappedArrayGetSparseProperties", (void *)cuMipmappedArrayGetSparseProperties},
    {"cuArrayGetMemoryRequirements", (void *)cuArrayGetMemoryRequirements},
    {"cuMipmappedArrayGetMemoryRequirements", (void *)cuMipmappedArrayGetMemoryRequirements},
    {"cuArrayGetPlane", (void *)cuArrayGetPlane},
    {"cuArrayDestroy", (void *)cuArrayDestroy},
    {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
    {"cuMipmappedArrayGetLevel", (void *)cuMipmappedArrayGetLevel},
    {"cuMipmappedArrayDestroy", (void *)cuMipmappedArrayDestroy},
    {"cuMemGetHandleForAddressRange", (void *)cuMemGetHandleForAddressRange},
    {"cuMemAddressReserve", (void *)cuMemAddressReserve},
    {"cuMemAddressFree", (void *)cuMemAddressFree},
    {"cuMemRelease", (void *)cuMemRelease},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemMapArrayAsync", (void *)cuMemMapArrayAsync},
    {"cuMemUnmap", (void *)cuMemUnmap},
    {"cuMemExportToShareableHandle", (void *)cuMemExportToShareableHandle},
    {"cuMemImportFromShareableHandle", (void *)cuMemImportFromShareableHandle},
    {"cuMemGetAllocationPropertiesFromHandle", (void *)cuMemGetAllocationPropertiesFromHandle},
    {"cuMemRetainAllocationHandle", (void *)cuMemRetainAllocationHandle},
    {"cuMemFreeAsync", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync", (void *)cuMemAllocAsync},
    {"cuMemPoolTrimTo", (void *)cuMemPoolTrimTo},
    {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
    {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
    {"cuMemPoolGetAccess", (void *)cuMemPoolGetAccess},
    {"cuMemPoolDestroy", (void *)cuMemPoolDestroy},
    {"cuMemAllocFromPoolAsync", (void *)cuMemAllocFromPoolAsync},
    {"cuMemPoolExportToShareableHandle", (void *)cuMemPoolExportToShareableHandle},
    {"cuMemPoolImportFromShareableHandle", (void *)cuMemPoolImportFromShareableHandle},
    {"cuMemPoolExportPointer", (void *)cuMemPoolExportPointer},
    {"cuMemPoolImportPointer", (void *)cuMemPoolImportPointer},
    {"cuPointerGetAttribute", (void *)cuPointerGetAttribute},
    {"cuMemPrefetchAsync", (void *)cuMemPrefetchAsync},
    {"cuMemAdvise", (void *)cuMemAdvise},
    {"cuMemRangeGetAttribute", (void *)cuMemRangeGetAttribute},
    {"cuMemRangeGetAttributes", (void *)cuMemRangeGetAttributes},
    {"cuPointerGetAttributes", (void *)cuPointerGetAttributes},
    {"cuStreamCreate", (void *)cuStreamCreate},
    {"cuStreamCreateWithPriority", (void *)cuStreamCreateWithPriority},
    {"cuStreamGetPriority", (void *)cuStreamGetPriority},
    {"cuStreamGetFlags", (void *)cuStreamGetFlags},
    {"cuStreamGetId", (void *)cuStreamGetId},
    {"cuStreamGetCtx", (void *)cuStreamGetCtx},
    {"cuStreamWaitEvent", (void *)cuStreamWaitEvent},
    {"cuStreamAddCallback", (void *)cuStreamAddCallback},
    {"cuStreamBeginCapture_v2", (void *)cuStreamBeginCapture_v2},
    {"cuThreadExchangeStreamCaptureMode", (void *)cuThreadExchangeStreamCaptureMode},
    {"cuStreamEndCapture", (void *)cuStreamEndCapture},
    {"cuStreamIsCapturing", (void *)cuStreamIsCapturing},
    {"cuStreamUpdateCaptureDependencies", (void *)cuStreamUpdateCaptureDependencies},
    {"cuStreamAttachMemAsync", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery", (void *)cuStreamQuery},
    {"cuStreamSynchronize", (void *)cuStreamSynchronize},
    {"cuStreamDestroy_v2", (void *)cuStreamDestroy_v2},
    {"cuStreamCopyAttributes", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute", (void *)cuStreamGetAttribute},
    {"cuEventCreate", (void *)cuEventCreate},
    {"cuEventRecord", (void *)cuEventRecord},
    {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
    {"cuEventQuery", (void *)cuEventQuery},
    {"cuEventSynchronize", (void *)cuEventSynchronize},
    {"cuEventDestroy_v2", (void *)cuEventDestroy_v2},
    {"cuEventElapsedTime", (void *)cuEventElapsedTime},
    {"cuDestroyExternalMemory", (void *)cuDestroyExternalMemory},
    {"cuDestroyExternalSemaphore", (void *)cuDestroyExternalSemaphore},
    {"cuStreamWaitValue32_v2", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWaitValue64_v2", (void *)cuStreamWaitValue64_v2},
    {"cuStreamWriteValue32_v2", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWriteValue64_v2", (void *)cuStreamWriteValue64_v2},
    {"cuStreamBatchMemOp_v2", (void *)cuStreamBatchMemOp_v2},
    {"cuFuncGetAttribute", (void *)cuFuncGetAttribute},
    {"cuFuncSetAttribute", (void *)cuFuncSetAttribute},
    {"cuFuncSetCacheConfig", (void *)cuFuncSetCacheConfig},
    {"cuFuncSetSharedMemConfig", (void *)cuFuncSetSharedMemConfig},
    {"cuFuncGetModule", (void *)cuFuncGetModule},
    {"cuLaunchKernel", (void *)cuLaunchKernel},
    {"cuLaunchCooperativeKernel", (void *)cuLaunchCooperativeKernel},
    {"cuLaunchCooperativeKernelMultiDevice", (void *)cuLaunchCooperativeKernelMultiDevice},
    {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
    {"cuFuncSetBlockShape", (void *)cuFuncSetBlockShape},
    {"cuFuncSetSharedSize", (void *)cuFuncSetSharedSize},
    {"cuParamSetSize", (void *)cuParamSetSize},
    {"cuParamSeti", (void *)cuParamSeti},
    {"cuParamSetf", (void *)cuParamSetf},
    {"cuParamSetv", (void *)cuParamSetv},
    {"cuLaunch", (void *)cuLaunch},
    {"cuLaunchGrid", (void *)cuLaunchGrid},
    {"cuLaunchGridAsync", (void *)cuLaunchGridAsync},
    {"cuParamSetTexRef", (void *)cuParamSetTexRef},
    {"cuGraphCreate", (void *)cuGraphCreate},
    {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
    {"cuGraphMemcpyNodeGetParams", (void *)cuGraphMemcpyNodeGetParams},
    {"cuGraphMemsetNodeGetParams", (void *)cuGraphMemsetNodeGetParams},
    {"cuGraphHostNodeGetParams", (void *)cuGraphHostNodeGetParams},
    {"cuGraphChildGraphNodeGetGraph", (void *)cuGraphChildGraphNodeGetGraph},
    {"cuGraphEventRecordNodeGetEvent", (void *)cuGraphEventRecordNodeGetEvent},
    {"cuGraphEventRecordNodeSetEvent", (void *)cuGraphEventRecordNodeSetEvent},
    {"cuGraphEventWaitNodeGetEvent", (void *)cuGraphEventWaitNodeGetEvent},
    {"cuGraphEventWaitNodeSetEvent", (void *)cuGraphEventWaitNodeSetEvent},
    {"cuGraphExternalSemaphoresSignalNodeGetParams", (void *)cuGraphExternalSemaphoresSignalNodeGetParams},
    {"cuGraphExternalSemaphoresWaitNodeGetParams", (void *)cuGraphExternalSemaphoresWaitNodeGetParams},
    {"cuGraphBatchMemOpNodeGetParams", (void *)cuGraphBatchMemOpNodeGetParams},
    {"cuGraphMemAllocNodeGetParams", (void *)cuGraphMemAllocNodeGetParams},
    {"cuGraphMemFreeNodeGetParams", (void *)cuGraphMemFreeNodeGetParams},
    {"cuDeviceGraphMemTrim", (void *)cuDeviceGraphMemTrim},
    {"cuDeviceGetGraphMemAttribute", (void *)cuDeviceGetGraphMemAttribute},
    {"cuDeviceSetGraphMemAttribute", (void *)cuDeviceSetGraphMemAttribute},
    {"cuGraphClone", (void *)cuGraphClone},
    {"cuGraphNodeFindInClone", (void *)cuGraphNodeFindInClone},
    {"cuGraphNodeGetType", (void *)cuGraphNodeGetType},
    {"cuGraphGetNodes", (void *)cuGraphGetNodes},
    {"cuGraphGetRootNodes", (void *)cuGraphGetRootNodes},
    {"cuGraphGetEdges", (void *)cuGraphGetEdges},
    {"cuGraphNodeGetDependencies", (void *)cuGraphNodeGetDependencies},
    {"cuGraphNodeGetDependentNodes", (void *)cuGraphNodeGetDependentNodes},
    {"cuGraphDestroyNode", (void *)cuGraphDestroyNode},
    {"cuGraphInstantiateWithFlags", (void *)cuGraphInstantiateWithFlags},
    {"cuGraphInstantiateWithParams", (void *)cuGraphInstantiateWithParams},
    {"cuGraphExecGetFlags", (void *)cuGraphExecGetFlags},
    {"cuGraphExecChildGraphNodeSetParams", (void *)cuGraphExecChildGraphNodeSetParams},
    {"cuGraphExecEventRecordNodeSetEvent", (void *)cuGraphExecEventRecordNodeSetEvent},
    {"cuGraphExecEventWaitNodeSetEvent", (void *)cuGraphExecEventWaitNodeSetEvent},
    {"cuGraphNodeSetEnabled", (void *)cuGraphNodeSetEnabled},
    {"cuGraphNodeGetEnabled", (void *)cuGraphNodeGetEnabled},
    {"cuGraphUpload", (void *)cuGraphUpload},
    {"cuGraphLaunch", (void *)cuGraphLaunch},
    {"cuGraphExecDestroy", (void *)cuGraphExecDestroy},
    {"cuGraphDestroy", (void *)cuGraphDestroy},
    {"cuGraphExecUpdate_v2", (void *)cuGraphExecUpdate_v2},
    {"cuGraphKernelNodeCopyAttributes", (void *)cuGraphKernelNodeCopyAttributes},
    {"cuGraphKernelNodeGetAttribute", (void *)cuGraphKernelNodeGetAttribute},
    {"cuUserObjectCreate", (void *)cuUserObjectCreate},
    {"cuUserObjectRetain", (void *)cuUserObjectRetain},
    {"cuUserObjectRelease", (void *)cuUserObjectRelease},
    {"cuGraphRetainUserObject", (void *)cuGraphRetainUserObject},
    {"cuGraphReleaseUserObject", (void *)cuGraphReleaseUserObject},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessor", (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", (void *)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {"cuOccupancyAvailableDynamicSMemPerBlock", (void *)cuOccupancyAvailableDynamicSMemPerBlock},
    {"cuTexRefSetArray", (void *)cuTexRefSetArray},
    {"cuTexRefSetMipmappedArray", (void *)cuTexRefSetMipmappedArray},
    {"cuTexRefSetAddress_v2", (void *)cuTexRefSetAddress_v2},
    {"cuTexRefSetFormat", (void *)cuTexRefSetFormat},
    {"cuTexRefSetAddressMode", (void *)cuTexRefSetAddressMode},
    {"cuTexRefSetFilterMode", (void *)cuTexRefSetFilterMode},
    {"cuTexRefSetMipmapFilterMode", (void *)cuTexRefSetMipmapFilterMode},
    {"cuTexRefSetMipmapLevelBias", (void *)cuTexRefSetMipmapLevelBias},
    {"cuTexRefSetMipmapLevelClamp", (void *)cuTexRefSetMipmapLevelClamp},
    {"cuTexRefSetMaxAnisotropy", (void *)cuTexRefSetMaxAnisotropy},
    {"cuTexRefSetBorderColor", (void *)cuTexRefSetBorderColor},
    {"cuTexRefSetFlags", (void *)cuTexRefSetFlags},
    {"cuTexRefGetAddress_v2", (void *)cuTexRefGetAddress_v2},
    {"cuTexRefGetArray", (void *)cuTexRefGetArray},
    {"cuTexRefGetMipmappedArray", (void *)cuTexRefGetMipmappedArray},
    {"cuTexRefGetAddressMode", (void *)cuTexRefGetAddressMode},
    {"cuTexRefGetFilterMode", (void *)cuTexRefGetFilterMode},
    {"cuTexRefGetFormat", (void *)cuTexRefGetFormat},
    {"cuTexRefGetMipmapFilterMode", (void *)cuTexRefGetMipmapFilterMode},
    {"cuTexRefGetMipmapLevelBias", (void *)cuTexRefGetMipmapLevelBias},
    {"cuTexRefGetMipmapLevelClamp", (void *)cuTexRefGetMipmapLevelClamp},
    {"cuTexRefGetMaxAnisotropy", (void *)cuTexRefGetMaxAnisotropy},
    {"cuTexRefGetBorderColor", (void *)cuTexRefGetBorderColor},
    {"cuTexRefGetFlags", (void *)cuTexRefGetFlags},
    {"cuTexRefCreate", (void *)cuTexRefCreate},
    {"cuTexRefDestroy", (void *)cuTexRefDestroy},
    {"cuSurfRefSetArray", (void *)cuSurfRefSetArray},
    {"cuSurfRefGetArray", (void *)cuSurfRefGetArray},
    {"cuTexObjectDestroy", (void *)cuTexObjectDestroy},
    {"cuTexObjectGetResourceDesc", (void *)cuTexObjectGetResourceDesc},
    {"cuTexObjectGetTextureDesc", (void *)cuTexObjectGetTextureDesc},
    {"cuTexObjectGetResourceViewDesc", (void *)cuTexObjectGetResourceViewDesc},
    {"cuSurfObjectDestroy", (void *)cuSurfObjectDestroy},
    {"cuSurfObjectGetResourceDesc", (void *)cuSurfObjectGetResourceDesc},
    {"cuTensorMapReplaceAddress", (void *)cuTensorMapReplaceAddress},
    {"cuDeviceCanAccessPeer", (void *)cuDeviceCanAccessPeer},
    {"cuCtxEnablePeerAccess", (void *)cuCtxEnablePeerAccess},
    {"cuCtxDisablePeerAccess", (void *)cuCtxDisablePeerAccess},
    {"cuDeviceGetP2PAttribute", (void *)cuDeviceGetP2PAttribute},
    {"cuGraphicsUnregisterResource", (void *)cuGraphicsUnregisterResource},
    {"cuGraphicsSubResourceGetMappedArray", (void *)cuGraphicsSubResourceGetMappedArray},
    {"cuGraphicsResourceGetMappedMipmappedArray", (void *)cuGraphicsResourceGetMappedMipmappedArray},
    {"cuGraphicsResourceGetMappedPointer_v2", (void *)cuGraphicsResourceGetMappedPointer_v2},
    {"cuGraphicsResourceSetMapFlags_v2", (void *)cuGraphicsResourceSetMapFlags_v2},
    {"cuGraphicsMapResources", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources", (void *)cuGraphicsUnmapResources},
    {"cudaDeviceReset", (void *)cudaDeviceReset},
    {"cudaDeviceSynchronize", (void *)cudaDeviceSynchronize},
    {"cudaDeviceSetLimit", (void *)cudaDeviceSetLimit},
    {"cudaDeviceGetLimit", (void *)cudaDeviceGetLimit},
    {"cudaDeviceGetCacheConfig", (void *)cudaDeviceGetCacheConfig},
    {"cudaDeviceGetStreamPriorityRange", (void *)cudaDeviceGetStreamPriorityRange},
    {"cudaDeviceSetCacheConfig", (void *)cudaDeviceSetCacheConfig},
    {"cudaDeviceGetSharedMemConfig", (void *)cudaDeviceGetSharedMemConfig},
    {"cudaDeviceSetSharedMemConfig", (void *)cudaDeviceSetSharedMemConfig},
    {"cudaDeviceGetPCIBusId", (void *)cudaDeviceGetPCIBusId},
    {"cudaIpcGetEventHandle", (void *)cudaIpcGetEventHandle},
    {"cudaIpcOpenEventHandle", (void *)cudaIpcOpenEventHandle},
    {"cudaIpcGetMemHandle", (void *)cudaIpcGetMemHandle},
    {"cudaIpcOpenMemHandle", (void *)cudaIpcOpenMemHandle},
    {"cudaIpcCloseMemHandle", (void *)cudaIpcCloseMemHandle},
    {"cudaDeviceFlushGPUDirectRDMAWrites", (void *)cudaDeviceFlushGPUDirectRDMAWrites},
    {"cudaThreadExit", (void *)cudaThreadExit},
    {"cudaThreadSynchronize", (void *)cudaThreadSynchronize},
    {"cudaThreadSetLimit", (void *)cudaThreadSetLimit},
    {"cudaThreadGetLimit", (void *)cudaThreadGetLimit},
    {"cudaThreadGetCacheConfig", (void *)cudaThreadGetCacheConfig},
    {"cudaThreadSetCacheConfig", (void *)cudaThreadSetCacheConfig},
    {"cudaGetLastError", (void *)cudaGetLastError},
    {"cudaPeekAtLastError", (void *)cudaPeekAtLastError},
    {"cudaGetDeviceCount", (void *)cudaGetDeviceCount},
    {"cudaGetDeviceProperties_v2", (void *)cudaGetDeviceProperties_v2},
    {"cudaDeviceGetAttribute", (void *)cudaDeviceGetAttribute},
    {"cudaDeviceGetDefaultMemPool", (void *)cudaDeviceGetDefaultMemPool},
    {"cudaDeviceSetMemPool", (void *)cudaDeviceSetMemPool},
    {"cudaDeviceGetMemPool", (void *)cudaDeviceGetMemPool},
    {"cudaDeviceGetNvSciSyncAttributes", (void *)cudaDeviceGetNvSciSyncAttributes},
    {"cudaDeviceGetP2PAttribute", (void *)cudaDeviceGetP2PAttribute},
    {"cudaInitDevice", (void *)cudaInitDevice},
    {"cudaSetDevice", (void *)cudaSetDevice},
    {"cudaGetDevice", (void *)cudaGetDevice},
    {"cudaSetValidDevices", (void *)cudaSetValidDevices},
    {"cudaSetDeviceFlags", (void *)cudaSetDeviceFlags},
    {"cudaGetDeviceFlags", (void *)cudaGetDeviceFlags},
    {"cudaStreamCreate", (void *)cudaStreamCreate},
    {"cudaStreamCreateWithFlags", (void *)cudaStreamCreateWithFlags},
    {"cudaStreamCreateWithPriority", (void *)cudaStreamCreateWithPriority},
    {"cudaStreamGetPriority", (void *)cudaStreamGetPriority},
    {"cudaStreamGetFlags", (void *)cudaStreamGetFlags},
    {"cudaStreamGetId", (void *)cudaStreamGetId},
    {"cudaCtxResetPersistingL2Cache", (void *)cudaCtxResetPersistingL2Cache},
    {"cudaStreamCopyAttributes", (void *)cudaStreamCopyAttributes},
    {"cudaStreamGetAttribute", (void *)cudaStreamGetAttribute},
    {"cudaStreamDestroy", (void *)cudaStreamDestroy},
    {"cudaStreamWaitEvent", (void *)cudaStreamWaitEvent},
    {"cudaStreamAddCallback", (void *)cudaStreamAddCallback},
    {"cudaStreamSynchronize", (void *)cudaStreamSynchronize},
    {"cudaStreamQuery", (void *)cudaStreamQuery},
    {"cudaStreamAttachMemAsync", (void *)cudaStreamAttachMemAsync},
    {"cudaStreamBeginCapture", (void *)cudaStreamBeginCapture},
    {"cudaThreadExchangeStreamCaptureMode", (void *)cudaThreadExchangeStreamCaptureMode},
    {"cudaStreamEndCapture", (void *)cudaStreamEndCapture},
    {"cudaStreamIsCapturing", (void *)cudaStreamIsCapturing},
    {"cudaStreamGetCaptureInfo_v2", (void *)cudaStreamGetCaptureInfo_v2},
    {"cudaStreamUpdateCaptureDependencies", (void *)cudaStreamUpdateCaptureDependencies},
    {"cudaEventCreate", (void *)cudaEventCreate},
    {"cudaEventCreateWithFlags", (void *)cudaEventCreateWithFlags},
    {"cudaEventRecord", (void *)cudaEventRecord},
    {"cudaEventRecordWithFlags", (void *)cudaEventRecordWithFlags},
    {"cudaEventQuery", (void *)cudaEventQuery},
    {"cudaEventSynchronize", (void *)cudaEventSynchronize},
    {"cudaEventDestroy", (void *)cudaEventDestroy},
    {"cudaEventElapsedTime", (void *)cudaEventElapsedTime},
    {"cudaDestroyExternalMemory", (void *)cudaDestroyExternalMemory},
    {"cudaDestroyExternalSemaphore", (void *)cudaDestroyExternalSemaphore},
    {"cudaLaunchCooperativeKernelMultiDevice", (void *)cudaLaunchCooperativeKernelMultiDevice},
    {"cudaSetDoubleForDevice", (void *)cudaSetDoubleForDevice},
    {"cudaSetDoubleForHost", (void *)cudaSetDoubleForHost},
    {"cudaLaunchHostFunc", (void *)cudaLaunchHostFunc},
    {"cudaMallocManaged", (void *)cudaMallocManaged},
    {"cudaMalloc", (void *)cudaMalloc},
    {"cudaMallocHost", (void *)cudaMallocHost},
    {"cudaMallocPitch", (void *)cudaMallocPitch},
    {"cudaFree", (void *)cudaFree},
    {"cudaFreeHost", (void *)cudaFreeHost},
    {"cudaFreeArray", (void *)cudaFreeArray},
    {"cudaFreeMipmappedArray", (void *)cudaFreeMipmappedArray},
    {"cudaHostAlloc", (void *)cudaHostAlloc},
    {"cudaHostRegister", (void *)cudaHostRegister},
    {"cudaHostUnregister", (void *)cudaHostUnregister},
    {"cudaHostGetDevicePointer", (void *)cudaHostGetDevicePointer},
    {"cudaHostGetFlags", (void *)cudaHostGetFlags},
    {"cudaMalloc3D", (void *)cudaMalloc3D},
    {"cudaGetMipmappedArrayLevel", (void *)cudaGetMipmappedArrayLevel},
    {"cudaMemGetInfo", (void *)cudaMemGetInfo},
    {"cudaArrayGetInfo", (void *)cudaArrayGetInfo},
    {"cudaArrayGetPlane", (void *)cudaArrayGetPlane},
    {"cudaArrayGetMemoryRequirements", (void *)cudaArrayGetMemoryRequirements},
    {"cudaMipmappedArrayGetMemoryRequirements", (void *)cudaMipmappedArrayGetMemoryRequirements},
    {"cudaArrayGetSparseProperties", (void *)cudaArrayGetSparseProperties},
    {"cudaMipmappedArrayGetSparseProperties", (void *)cudaMipmappedArrayGetSparseProperties},
    {"cudaMemcpy2DFromArray", (void *)cudaMemcpy2DFromArray},
    {"cudaMemcpy2DArrayToArray", (void *)cudaMemcpy2DArrayToArray},
    {"cudaMemcpy2DFromArrayAsync", (void *)cudaMemcpy2DFromArrayAsync},
    {"cudaMemset", (void *)cudaMemset},
    {"cudaMemset2D", (void *)cudaMemset2D},
    {"cudaMemset3D", (void *)cudaMemset3D},
    {"cudaMemsetAsync", (void *)cudaMemsetAsync},
    {"cudaMemset2DAsync", (void *)cudaMemset2DAsync},
    {"cudaMemset3DAsync", (void *)cudaMemset3DAsync},
    {"cudaMemcpyFromArray", (void *)cudaMemcpyFromArray},
    {"cudaMemcpyArrayToArray", (void *)cudaMemcpyArrayToArray},
    {"cudaMemcpyFromArrayAsync", (void *)cudaMemcpyFromArrayAsync},
    {"cudaMallocAsync", (void *)cudaMallocAsync},
    {"cudaFreeAsync", (void *)cudaFreeAsync},
    {"cudaMemPoolTrimTo", (void *)cudaMemPoolTrimTo},
    {"cudaMemPoolSetAttribute", (void *)cudaMemPoolSetAttribute},
    {"cudaMemPoolGetAttribute", (void *)cudaMemPoolGetAttribute},
    {"cudaMemPoolGetAccess", (void *)cudaMemPoolGetAccess},
    {"cudaMemPoolDestroy", (void *)cudaMemPoolDestroy},
    {"cudaMallocFromPoolAsync", (void *)cudaMallocFromPoolAsync},
    {"cudaMemPoolExportToShareableHandle", (void *)cudaMemPoolExportToShareableHandle},
    {"cudaMemPoolImportFromShareableHandle", (void *)cudaMemPoolImportFromShareableHandle},
    {"cudaMemPoolExportPointer", (void *)cudaMemPoolExportPointer},
    {"cudaMemPoolImportPointer", (void *)cudaMemPoolImportPointer},
    {"cudaDeviceCanAccessPeer", (void *)cudaDeviceCanAccessPeer},
    {"cudaDeviceEnablePeerAccess", (void *)cudaDeviceEnablePeerAccess},
    {"cudaDeviceDisablePeerAccess", (void *)cudaDeviceDisablePeerAccess},
    {"cudaGraphicsUnregisterResource", (void *)cudaGraphicsUnregisterResource},
    {"cudaGraphicsResourceSetMapFlags", (void *)cudaGraphicsResourceSetMapFlags},
    {"cudaGraphicsMapResources", (void *)cudaGraphicsMapResources},
    {"cudaGraphicsUnmapResources", (void *)cudaGraphicsUnmapResources},
    {"cudaGraphicsResourceGetMappedPointer", (void *)cudaGraphicsResourceGetMappedPointer},
    {"cudaGraphicsSubResourceGetMappedArray", (void *)cudaGraphicsSubResourceGetMappedArray},
    {"cudaGraphicsResourceGetMappedMipmappedArray", (void *)cudaGraphicsResourceGetMappedMipmappedArray},
    {"cudaGetChannelDesc", (void *)cudaGetChannelDesc},
    {"cudaDestroyTextureObject", (void *)cudaDestroyTextureObject},
    {"cudaGetTextureObjectResourceDesc", (void *)cudaGetTextureObjectResourceDesc},
    {"cudaGetTextureObjectTextureDesc", (void *)cudaGetTextureObjectTextureDesc},
    {"cudaGetTextureObjectResourceViewDesc", (void *)cudaGetTextureObjectResourceViewDesc},
    {"cudaDestroySurfaceObject", (void *)cudaDestroySurfaceObject},
    {"cudaGetSurfaceObjectResourceDesc", (void *)cudaGetSurfaceObjectResourceDesc},
    {"cudaDriverGetVersion", (void *)cudaDriverGetVersion},
    {"cudaRuntimeGetVersion", (void *)cudaRuntimeGetVersion},
    {"cudaGraphCreate", (void *)cudaGraphCreate},
    {"cudaGraphKernelNodeGetParams", (void *)cudaGraphKernelNodeGetParams},
    {"cudaGraphKernelNodeCopyAttributes", (void *)cudaGraphKernelNodeCopyAttributes},
    {"cudaGraphKernelNodeGetAttribute", (void *)cudaGraphKernelNodeGetAttribute},
    {"cudaGraphMemcpyNodeGetParams", (void *)cudaGraphMemcpyNodeGetParams},
    {"cudaGraphMemsetNodeGetParams", (void *)cudaGraphMemsetNodeGetParams},
    {"cudaGraphHostNodeGetParams", (void *)cudaGraphHostNodeGetParams},
    {"cudaGraphChildGraphNodeGetGraph", (void *)cudaGraphChildGraphNodeGetGraph},
    {"cudaGraphEventRecordNodeGetEvent", (void *)cudaGraphEventRecordNodeGetEvent},
    {"cudaGraphEventRecordNodeSetEvent", (void *)cudaGraphEventRecordNodeSetEvent},
    {"cudaGraphEventWaitNodeGetEvent", (void *)cudaGraphEventWaitNodeGetEvent},
    {"cudaGraphEventWaitNodeSetEvent", (void *)cudaGraphEventWaitNodeSetEvent},
    {"cudaGraphExternalSemaphoresSignalNodeGetParams", (void *)cudaGraphExternalSemaphoresSignalNodeGetParams},
    {"cudaGraphExternalSemaphoresWaitNodeGetParams", (void *)cudaGraphExternalSemaphoresWaitNodeGetParams},
    {"cudaGraphMemAllocNodeGetParams", (void *)cudaGraphMemAllocNodeGetParams},
    {"cudaGraphMemFreeNodeGetParams", (void *)cudaGraphMemFreeNodeGetParams},
    {"cudaDeviceGraphMemTrim", (void *)cudaDeviceGraphMemTrim},
    {"cudaDeviceGetGraphMemAttribute", (void *)cudaDeviceGetGraphMemAttribute},
    {"cudaDeviceSetGraphMemAttribute", (void *)cudaDeviceSetGraphMemAttribute},
    {"cudaGraphClone", (void *)cudaGraphClone},
    {"cudaGraphNodeFindInClone", (void *)cudaGraphNodeFindInClone},
    {"cudaGraphNodeGetType", (void *)cudaGraphNodeGetType},
    {"cudaGraphGetNodes", (void *)cudaGraphGetNodes},
    {"cudaGraphGetRootNodes", (void *)cudaGraphGetRootNodes},
    {"cudaGraphGetEdges", (void *)cudaGraphGetEdges},
    {"cudaGraphNodeGetDependencies", (void *)cudaGraphNodeGetDependencies},
    {"cudaGraphNodeGetDependentNodes", (void *)cudaGraphNodeGetDependentNodes},
    {"cudaGraphDestroyNode", (void *)cudaGraphDestroyNode},
    {"cudaGraphInstantiate", (void *)cudaGraphInstantiate},
    {"cudaGraphInstantiateWithFlags", (void *)cudaGraphInstantiateWithFlags},
    {"cudaGraphInstantiateWithParams", (void *)cudaGraphInstantiateWithParams},
    {"cudaGraphExecGetFlags", (void *)cudaGraphExecGetFlags},
    {"cudaGraphExecChildGraphNodeSetParams", (void *)cudaGraphExecChildGraphNodeSetParams},
    {"cudaGraphExecEventRecordNodeSetEvent", (void *)cudaGraphExecEventRecordNodeSetEvent},
    {"cudaGraphExecEventWaitNodeSetEvent", (void *)cudaGraphExecEventWaitNodeSetEvent},
    {"cudaGraphNodeSetEnabled", (void *)cudaGraphNodeSetEnabled},
    {"cudaGraphNodeGetEnabled", (void *)cudaGraphNodeGetEnabled},
    {"cudaGraphExecUpdate", (void *)cudaGraphExecUpdate},
    {"cudaGraphUpload", (void *)cudaGraphUpload},
    {"cudaGraphLaunch", (void *)cudaGraphLaunch},
    {"cudaGraphExecDestroy", (void *)cudaGraphExecDestroy},
    {"cudaGraphDestroy", (void *)cudaGraphDestroy},
    {"cudaUserObjectCreate", (void *)cudaUserObjectCreate},
    {"cudaUserObjectRetain", (void *)cudaUserObjectRetain},
    {"cudaUserObjectRelease", (void *)cudaUserObjectRelease},
    {"cudaGraphRetainUserObject", (void *)cudaGraphRetainUserObject},
    {"cudaGraphReleaseUserObject", (void *)cudaGraphReleaseUserObject},
    {"cuMemcpy_ptds", (void *)cuMemcpy},
    {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
    {"cuMemcpyPeer_ptds", (void *)cuMemcpyPeer},
    {"cuMemcpyPeerAsync_ptsz", (void *)cuMemcpyPeerAsync},
    {"cuMemPrefetchAsync_ptsz", (void *)cuMemPrefetchAsync},
    {"cuMemsetD8Async_ptsz", (void *)cuMemsetD8Async},
    {"cuMemsetD16Async_ptsz", (void *)cuMemsetD16Async},
    {"cuMemsetD32Async_ptsz", (void *)cuMemsetD32Async},
    {"cuMemsetD2D8Async_ptsz", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D16Async_ptsz", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D32Async_ptsz", (void *)cuMemsetD2D32Async},
    {"cuStreamGetPriority_ptsz", (void *)cuStreamGetPriority},
    {"cuStreamGetId_ptsz", (void *)cuStreamGetId},
    {"cuStreamGetFlags_ptsz", (void *)cuStreamGetFlags},
    {"cuStreamGetCtx_ptsz", (void *)cuStreamGetCtx},
    {"cuStreamWaitEvent_ptsz", (void *)cuStreamWaitEvent},
    {"cuStreamEndCapture_ptsz", (void *)cuStreamEndCapture},
    {"cuStreamIsCapturing_ptsz", (void *)cuStreamIsCapturing},
    {"cuStreamUpdateCaptureDependencies_ptsz", (void *)cuStreamUpdateCaptureDependencies},
    {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
    {"cuStreamAttachMemAsync_ptsz", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery_ptsz", (void *)cuStreamQuery},
    {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize},
    {"cuEventRecord_ptsz", (void *)cuEventRecord},
    {"cuEventRecordWithFlags_ptsz", (void *)cuEventRecordWithFlags},
    {"cuLaunchKernel_ptsz", (void *)cuLaunchKernel},
    {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
    {"cuGraphicsMapResources_ptsz", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources_ptsz", (void *)cuGraphicsUnmapResources},
    {"cuLaunchCooperativeKernel_ptsz", (void *)cuLaunchCooperativeKernel},
    {"cuGraphInstantiateWithParams_ptsz", (void *)cuGraphInstantiateWithParams},
    {"cuGraphUpload_ptsz", (void *)cuGraphUpload},
    {"cuGraphLaunch_ptsz", (void *)cuGraphLaunch},
    {"cuStreamCopyAttributes_ptsz", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute_ptsz", (void *)cuStreamGetAttribute},
    {"cuMemMapArrayAsync_ptsz", (void *)cuMemMapArrayAsync},
    {"cuMemFreeAsync_ptsz", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync_ptsz", (void *)cuMemAllocAsync},
    {"cuMemAllocFromPoolAsync_ptsz", (void *)cuMemAllocFromPoolAsync},
    {"cudaMemcpy", (void *)cudaMemcpy},
    {"cudaMemcpyAsync", (void *)cudaMemcpyAsync},
    {"cudaLaunchKernel", (void *)cudaLaunchKernel},
};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it == functionMap.end())
        return nullptr;
    return it->second;
}
