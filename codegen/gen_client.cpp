#include <nvml.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "manual_client.h"

extern int rpc_size();
extern int rpc_start_request(const int index, const unsigned int request);
extern int rpc_write(const int index, const void *data, const std::size_t size);
extern int rpc_end_request(const int index);
extern int rpc_wait_for_response(const int index);
extern int rpc_read(const int index, void *data, const std::size_t size);
extern int rpc_end_response(const int index, void *return_value);
extern int rpc_close();

nvmlReturn_t nvmlInit_v2()
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlInit_v2) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlInitWithFlags) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlShutdown()
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlShutdown) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    if (rpc_close() < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetDriverVersion) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetNVMLVersion) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetCudaDriverVersion) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetCudaDriverVersion_v2) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetProcessName) < 0 ||
        rpc_write(0, &pid, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, name, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetCount) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, unitCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetHandleByIndex) < 0 ||
        rpc_write(0, &index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetUnitInfo) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetLedState) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetPsuInfo) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetTemperature) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(0, &type, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, temp, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetFanSpeedInfo) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitGetDevices) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(0, deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, devices, *deviceCount * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetHicVersion) < 0 ||
        rpc_write(0, hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCount_v2) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAttributes_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetHandleByIndex_v2) < 0 ||
        rpc_write(0, &index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;
    std::size_t serial_len = std::strlen(serial) + 1;
    if (rpc_start_request(0, RPC_nvmlDeviceGetHandleBySerial) < 0 ||
        rpc_write(0, &serial_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, serial, serial_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;
    std::size_t uuid_len = std::strlen(uuid) + 1;
    if (rpc_start_request(0, RPC_nvmlDeviceGetHandleByUUID) < 0 ||
        rpc_write(0, &uuid_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, uuid, uuid_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;
    std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
    if (rpc_start_request(0, RPC_nvmlDeviceGetHandleByPciBusId_v2) < 0 ||
        rpc_write(0, &pciBusId_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, pciBusId, pciBusId_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetName) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, name, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetBrand) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetIndex) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, index, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSerial) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, serial, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemoryAffinity) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &nodeSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodeSet, nodeSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCpuAffinityWithinScope) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCpuAffinity) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetCpuAffinity) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceClearCpuAffinity) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTopologyCommonAncestor) < 0 ||
        rpc_write(0, &device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTopologyNearestGpus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSystemGetTopologyGpuSet) < 0 ||
        rpc_write(0, &cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetP2PStatus) < 0 ||
        rpc_write(0, &device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetUUID) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, uuid, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetMdevUUID) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, &size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mdevUuid, size * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMinorNumber) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetBoardPartNumber) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, partNumber, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetInforomVersion) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &object, sizeof(nvmlInforomObject_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetInforomImageVersion) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetInforomConfigurationChecksum) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, checksum, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceValidateInforom) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDisplayMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDisplayActive) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPersistenceMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPciInfo_v3) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMaxPcieLinkGeneration) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMaxPcieLinkWidth) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCurrPcieLinkGeneration) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCurrPcieLinkWidth) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPcieThroughput) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPcieReplayCounter) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetClockInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clock, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMaxClockInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clock, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetApplicationsClock) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDefaultApplicationsClock) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceResetApplicationsClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetClock) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(0, &clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMaxCustomerBoostClock) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSupportedMemoryClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, clocksMHz, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSupportedGraphicsClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &memoryClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, clocksMHz, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAutoBoostedClocksEnabled) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(0, defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetAutoBoostedClocksEnabled) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetFanSpeed) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, speed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetFanSpeed_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, speed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTargetFanSpeed) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetDefaultFanSpeed_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMinMaxFanSpeed) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_read(0, maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetFanControlPolicy_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetFanControlPolicy) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &fan, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNumFans) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numFans, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTemperature) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, temp, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTemperatureThreshold) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, temp, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetTemperatureThreshold) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(0, temp, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, temp, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetThermalSettings) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPerformanceState) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long* clocksThrottleReasons)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCurrentClocksThrottleReasons) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long* supportedClocksThrottleReasons)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSupportedClocksThrottleReasons) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerState) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerManagementMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerManagementLimit) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, limit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerManagementLimitConstraints) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, minLimit, sizeof(unsigned int)) < 0 ||
        rpc_read(0, maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerManagementDefaultLimit) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerUsage) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, power, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTotalEnergyConsumption) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, energy, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetEnforcedPowerLimit) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, limit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuOperationMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_read(0, pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemoryInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemoryInfo_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetComputeMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCudaComputeCapability) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, major, sizeof(int)) < 0 ||
        rpc_read(0, minor, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetEccMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(0, pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDefaultEccMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetBoardId) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, boardId, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMultiGpuBoard) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetTotalEccErrors) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(0, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDetailedEccErrors) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(0, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemoryErrorCounter) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(0, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(0, &locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetUtilizationRates) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetEncoderUtilization) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(0, samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetEncoderCapacity) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetEncoderStats) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(0, averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetEncoderSessions) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, sessionInfos, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDecoderUtilization) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(0, samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetFBCStats) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetFBCSessions) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, sessionInfo, *sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDriverModel) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(0, pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVbiosVersion) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetBridgeChipInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetComputeRunningProcesses_v3) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGraphicsRunningProcesses_v3) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceOnSameBoard) < 0 ||
        rpc_write(0, &device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, onSameBoard, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAPIRestriction) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSamples) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &type, sizeof(nvmlSamplingType_t)) < 0 ||
        rpc_write(0, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(0, sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(0, sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, samples, *sampleCount * sizeof(nvmlSample_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetBAR1MemoryInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t* violTime)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetViolationStatus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetIrqNum) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, irqNum, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNumGpuCores) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numCores, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPowerSource) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemoryBusWidth) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, busWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPcieLinkMaxSpeed) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPcieSpeed) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAdaptiveClockInfoStatus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAccountingMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAccountingStats) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &pid, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAccountingPids) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetAccountingBufferSize) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetRetiredPages) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(0, pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetRetiredPages_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(0, pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_read(0, timestamps, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetRetiredPagesPendingStatus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetRemappedRows) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, corrRows, sizeof(unsigned int)) < 0 ||
        rpc_read(0, uncRows, sizeof(unsigned int)) < 0 ||
        rpc_read(0, isPending, sizeof(unsigned int)) < 0 ||
        rpc_read(0, failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetRowRemapperHistogram) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetArchitecture) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlUnitSetLedState) < 0 ||
        rpc_write(0, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(0, &color, sizeof(nvmlLedColor_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetPersistenceMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetComputeMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetEccMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &ecc, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceClearEccErrorCounts) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetDriverModel) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &driverModel, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetGpuLockedClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &minGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &maxGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceResetGpuLockedClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetMemoryLockedClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &minMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &maxMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceResetMemoryLockedClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetApplicationsClocks) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &memClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &graphicsClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetClkMonStatus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetPowerManagementLimit) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetGpuOperationMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &mode, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetAPIRestriction) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_write(0, &isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetAccountingMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceClearAccountingPids) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkState) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkVersion) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkCapability) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkErrorCounter) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceResetNvLinkErrorCounters) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control, unsigned int reset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetNvLinkUtilizationControl) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &counter, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &control, sizeof(nvmlNvLinkUtilizationControl_t*)) < 0 ||
        rpc_write(0, &reset, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkUtilizationControl) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long* rxcounter, unsigned long long* txcounter)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkUtilizationCounter) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_read(0, txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceFreezeNvLinkUtilizationCounter) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &counter, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &freeze, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceResetNvLinkUtilizationCounter) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetNvLinkRemoteDeviceType) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlEventSetCreate) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceRegisterEvents) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_write(0, &set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSupportedEventTypes) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlEventSetWait_v2) < 0 ||
        rpc_write(0, &set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_write(0, &timeoutms, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlEventSetFree) < 0 ||
        rpc_write(0, &set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceModifyDrainState) < 0 ||
        rpc_write(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(0, &newState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceQueryDrainState) < 0 ||
        rpc_write(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_read(0, currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceRemoveGpu_v2) < 0 ||
        rpc_write(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(0, &gpuState, sizeof(nvmlDetachGpuState_t)) < 0 ||
        rpc_write(0, &linkState, sizeof(nvmlPcieLinkState_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceDiscoverGpus) < 0 ||
        rpc_write(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetFieldValues) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceClearFieldValues) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVirtualizationMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetHostVgpuMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetVirtualizationMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGridLicensableFeatures_v4) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetProcessUtilization) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, utilization, *processSamplesCount * sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGspFirmwareVersion) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGspFirmwareMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_read(0, defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGetVgpuDriverCapabilities) < 0 ||
        rpc_write(0, &capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuCapabilities) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSupportedVgpus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetCreatableVgpus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetClass) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, size, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuTypeClass, *size * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetName) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(0, size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, size, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuTypeName, *size * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetGpuInstanceProfileId) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetDeviceID) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_read(0, subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetFramebufferSize) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetNumDisplayHeads) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetResolution) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(0, &displayIndex, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, xdim, sizeof(unsigned int)) < 0 ||
        rpc_read(0, ydim, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetLicense) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(0, &size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuTypeLicenseString, size * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetFrameRateLimit) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetMaxInstances) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetMaxInstancesPerVm) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetActiveVgpus) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuInstances, *vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetVmID) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, &size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vmId, size * sizeof(char)) < 0 ||
        rpc_read(0, vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetUUID) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, &size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, uuid, size * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetVmDriverVersion) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, &length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetFbUsage) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetLicenseStatus) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, licensed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetType) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetFrameRateLimit) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetEccMode) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetEncoderCapacity) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceSetEncoderCapacity) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, &encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetEncoderStats) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(0, averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetEncoderSessions) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, sessionInfo, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetFBCStats) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetFBCSessions) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, sessionInfo, *sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetGpuInstanceId) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetGpuPciId) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, length, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuPciId, *length * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuTypeGetCapabilities) < 0 ||
        rpc_write(0, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(0, &capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetMetadata) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(0, vgpuMetadata, *bufferSize * sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuMetadata) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(0, pgpuMetadata, *bufferSize * sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGetVgpuCompatibility) < 0 ||
        rpc_write(0, vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(0, pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(0, compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetPgpuMetadataString) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(0, pgpuMetadata, *bufferSize * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuSchedulerLog) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuSchedulerState) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuSchedulerCapabilities) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGetVgpuVersion) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_read(0, current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlSetVgpuVersion) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuUtilization) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(0, sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(0, vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(0, vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, utilizationSamples, *vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetVgpuProcessUtilization) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(0, vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(0, utilizationSamples, *vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetAccountingMode) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetAccountingPids) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetAccountingStats) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(0, &pid, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceClearAccountingPids) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlVgpuInstanceGetLicenseInfo_v2) < 0 ||
        rpc_write(0, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGetExcludedDeviceCount) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGetExcludedDeviceInfoByIndex) < 0 ||
        rpc_write(0, &index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetMigMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &mode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMigMode) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, currentMode, sizeof(unsigned int)) < 0 ||
        rpc_read(0, pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstanceProfileInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &profile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstanceProfileInfoV) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &profile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, placements, *count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstanceRemainingCapacity) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceCreateGpuInstance) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceDestroy) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstances) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, gpuInstances, *count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstanceById) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetInfo) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &profile, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &engProfile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &profile, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &engProfile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, placements, *count * sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceCreateComputeInstance) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlComputeInstanceDestroy) < 0 ||
        rpc_write(0, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetComputeInstances) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(0, count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_read(0, computeInstances, *count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpuInstanceGetComputeInstanceById) < 0 ||
        rpc_write(0, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(0, &id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlComputeInstanceGetInfo_v2) < 0 ||
        rpc_write(0, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceIsMigDeviceHandle) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuInstanceId) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, id, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetComputeInstanceId) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, id, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMaxMigDeviceCount) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMigDeviceHandleByIndex) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle) < 0 ||
        rpc_write(0, &migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetBusType) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetDynamicPstatesInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetFanSpeed_v2) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &fan, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpcClkVfOffset) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, offset, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetGpcClkVfOffset) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemClkVfOffset) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, offset, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetMemClkVfOffset) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMinMaxClockOfPState) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(0, &pstate, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(0, maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetSupportedPerformanceStates) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pstates, size * sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpcClkMinMaxVfOffset) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, minOffset, sizeof(int)) < 0 ||
        rpc_read(0, maxOffset, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetMemClkMinMaxVfOffset) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, minOffset, sizeof(int)) < 0 ||
        rpc_read(0, maxOffset, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* gpuFabricInfo)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceGetGpuFabricInfo) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t* metricsGet)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpmMetricsGet) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpmSampleFree) < 0 ||
        rpc_write(0, &gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t* gpmSample)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpmSampleAlloc) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpmSampleGet) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpmMigSampleGet) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(0, &gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t* gpmSupport)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlGpmQueryDeviceSupport) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info)
{
    nvmlReturn_t return_value;
    if (rpc_start_request(0, RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold) < 0 ||
        rpc_write(0, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuInit(unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuInit) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDriverGetVersion(int* driverVersion)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDriverGetVersion) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, driverVersion, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGet) < 0 ||
        rpc_write(0, &ordinal, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(CUdevice)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetCount(int* count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetCount) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetName) < 0 ||
        rpc_write(0, &len, sizeof(int)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, name, len * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetUuid) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, uuid, 16) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetUuid_v2) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, uuid, 16) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev)
{
    CUresult return_value;
    std::size_t luid_len;
    if (rpc_start_request(0, RPC_cuDeviceGetLuid) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, &luid_len, sizeof(std::size_t)) < 0 ||
        rpc_read(0, luid, luid_len) < 0 ||
        rpc_read(0, deviceNodeMask, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceTotalMem_v2) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetTexture1DLinearMaxWidth) < 0 ||
        rpc_write(0, &format, sizeof(CUarray_format)) < 0 ||
        rpc_write(0, &numChannels, sizeof(unsigned)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, maxWidthInElements, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetAttribute) < 0 ||
        rpc_write(0, &attrib, sizeof(CUdevice_attribute)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pi, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceSetMemPool) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_write(0, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetMemPool) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetDefaultMemPool) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetExecAffinitySupport) < 0 ||
        rpc_write(0, &type, sizeof(CUexecAffinityType)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pi, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFlushGPUDirectRDMAWrites) < 0 ||
        rpc_write(0, &target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0 ||
        rpc_write(0, &scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetProperties) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, prop, sizeof(CUdevprop)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceComputeCapability) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, major, sizeof(int)) < 0 ||
        rpc_read(0, minor, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDevicePrimaryCtxRetain) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDevicePrimaryCtxRelease_v2) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDevicePrimaryCtxSetFlags_v2) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDevicePrimaryCtxGetState) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_read(0, active, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDevicePrimaryCtxReset_v2) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxCreate_v2) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxCreate_v3) < 0 ||
        rpc_write(0, &numParams, sizeof(int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_read(0, paramsArray, numParams * sizeof(CUexecAffinityParam)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDestroy_v2(CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxDestroy_v2) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxPushCurrent_v2) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxPopCurrent_v2) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxSetCurrent) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetCurrent(CUcontext* pctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetCurrent) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetDevice(CUdevice* device)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetDevice) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(CUdevice)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetFlags(unsigned int* flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetFlags) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetId) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ctxId, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSynchronize()
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxSynchronize) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxSetLimit) < 0 ||
        rpc_write(0, &limit, sizeof(CUlimit)) < 0 ||
        rpc_write(0, &value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetLimit) < 0 ||
        rpc_write(0, &limit, sizeof(CUlimit)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pvalue, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetCacheConfig) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pconfig, sizeof(CUfunc_cache)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxSetCacheConfig) < 0 ||
        rpc_write(0, &config, sizeof(CUfunc_cache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetSharedMemConfig) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pConfig, sizeof(CUsharedconfig)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxSetSharedMemConfig) < 0 ||
        rpc_write(0, &config, sizeof(CUsharedconfig)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetApiVersion) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, version, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetStreamPriorityRange) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, leastPriority, sizeof(int)) < 0 ||
        rpc_read(0, greatestPriority, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxResetPersistingL2Cache()
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxResetPersistingL2Cache) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxGetExecAffinity) < 0 ||
        rpc_write(0, &type, sizeof(CUexecAffinityType)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pExecAffinity, sizeof(CUexecAffinityParam)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxAttach) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDetach(CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxDetach) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname)
{
    CUresult return_value;
    std::size_t fname_len = std::strlen(fname) + 1;
    if (rpc_start_request(0, RPC_cuModuleLoad) < 0 ||
        rpc_write(0, &fname_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, fname, fname_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, module, sizeof(CUmodule)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleUnload(CUmodule hmod)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuModuleUnload) < 0 ||
        rpc_write(0, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuModuleGetLoadingMode) < 0 ||
        rpc_write(0, mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuModuleGetFunction) < 0 ||
        rpc_write(0, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, hfunc, sizeof(CUfunction)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuModuleGetGlobal_v2) < 0 ||
        rpc_write(0, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLinkCreate_v2) < 0 ||
        rpc_write(0, &numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(0, options, sizeof(CUjit_option)) < 0 ||
        rpc_write(0, optionValues, sizeof(void*)) < 0 ||
        rpc_write(0, stateOut, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, options, sizeof(CUjit_option)) < 0 ||
        rpc_read(0, optionValues, sizeof(void*)) < 0 ||
        rpc_read(0, stateOut, sizeof(CUlinkState)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues)
{
    CUresult return_value;
    std::size_t path_len = std::strlen(path) + 1;
    if (rpc_start_request(0, RPC_cuLinkAddFile_v2) < 0 ||
        rpc_write(0, &state, sizeof(CUlinkState)) < 0 ||
        rpc_write(0, &type, sizeof(CUjitInputType)) < 0 ||
        rpc_write(0, &path_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, path, path_len) < 0 ||
        rpc_write(0, &numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(0, options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(0, optionValues, numOptions * sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLinkComplete) < 0 ||
        rpc_write(0, &state, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, cubinOut, sizeof(void*)) < 0 ||
        rpc_read(0, sizeOut, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLinkDestroy(CUlinkState state)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLinkDestroy) < 0 ||
        rpc_write(0, &state, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuModuleGetTexRef) < 0 ||
        rpc_write(0, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuModuleGetSurfRef) < 0 ||
        rpc_write(0, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions)
{
    CUresult return_value;
    std::size_t fileName_len = std::strlen(fileName) + 1;
    if (rpc_start_request(0, RPC_cuLibraryLoadFromFile) < 0 ||
        rpc_write(0, &fileName_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, fileName, fileName_len) < 0 ||
        rpc_write(0, &numJitOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(0, jitOptions, numJitOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(0, jitOptionsValues, numJitOptions * sizeof(void*)) < 0 ||
        rpc_write(0, &numLibraryOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(0, libraryOptions, numLibraryOptions * sizeof(CUlibraryOption)) < 0 ||
        rpc_write(0, libraryOptionValues, numLibraryOptions * sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, library, sizeof(CUlibrary)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryUnload(CUlibrary library)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLibraryUnload) < 0 ||
        rpc_write(0, &library, sizeof(CUlibrary)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuLibraryGetKernel) < 0 ||
        rpc_write(0, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pKernel, sizeof(CUkernel)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLibraryGetModule) < 0 ||
        rpc_write(0, &library, sizeof(CUlibrary)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pMod, sizeof(CUmodule)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuKernelGetFunction) < 0 ||
        rpc_write(0, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pFunc, sizeof(CUfunction)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuLibraryGetGlobal) < 0 ||
        rpc_write(0, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name)
{
    CUresult return_value;
    std::size_t name_len = std::strlen(name) + 1;
    if (rpc_start_request(0, RPC_cuLibraryGetManaged) < 0 ||
        rpc_write(0, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(0, &name_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, name, name_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol)
{
    CUresult return_value;
    std::size_t symbol_len = std::strlen(symbol) + 1;
    if (rpc_start_request(0, RPC_cuLibraryGetUnifiedFunction) < 0 ||
        rpc_write(0, &library, sizeof(CUlibrary)) < 0 ||
        rpc_write(0, &symbol_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, symbol, symbol_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, fptr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuKernelGetAttribute) < 0 ||
        rpc_write(0, pi, sizeof(int)) < 0 ||
        rpc_write(0, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(0, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pi, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuKernelSetAttribute) < 0 ||
        rpc_write(0, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(0, &val, sizeof(int)) < 0 ||
        rpc_write(0, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuKernelSetCacheConfig) < 0 ||
        rpc_write(0, &kernel, sizeof(CUkernel)) < 0 ||
        rpc_write(0, &config, sizeof(CUfunc_cache)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemGetInfo_v2) < 0 ||
        rpc_write(0, free, sizeof(size_t)) < 0 ||
        rpc_write(0, total, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, free, sizeof(size_t)) < 0 ||
        rpc_read(0, total, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAlloc_v2) < 0 ||
        rpc_write(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &bytesize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAllocPitch_v2) < 0 ||
        rpc_write(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, pPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &WidthInBytes, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_write(0, &ElementSizeBytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, pPitch, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemFree_v2) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemGetAddressRange_v2) < 0 ||
        rpc_write(0, pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, psize, sizeof(size_t)) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, psize, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocHost_v2(void** pp, size_t bytesize)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAllocHost_v2) < 0 ||
        rpc_write(0, &bytesize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pp, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFreeHost(void* p)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemFreeHost) < 0 ||
        rpc_write(0, &p, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemHostAlloc) < 0 ||
        rpc_write(0, &bytesize, sizeof(size_t)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pp, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemHostGetDevicePointer_v2) < 0 ||
        rpc_write(0, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &p, sizeof(void*)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemHostGetFlags) < 0 ||
        rpc_write(0, pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &p, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pFlags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAllocManaged) < 0 ||
        rpc_write(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &bytesize, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId)
{
    CUresult return_value;
    std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
    if (rpc_start_request(0, RPC_cuDeviceGetByPCIBusId) < 0 ||
        rpc_write(0, dev, sizeof(CUdevice)) < 0 ||
        rpc_write(0, &pciBusId_len, sizeof(std::size_t)) < 0 ||
        rpc_write(0, pciBusId, pciBusId_len) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dev, sizeof(CUdevice)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetPCIBusId) < 0 ||
        rpc_write(0, &len, sizeof(int)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pciBusId, len * sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuIpcGetEventHandle) < 0 ||
        rpc_write(0, pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuIpcOpenEventHandle) < 0 ||
        rpc_write(0, phEvent, sizeof(CUevent)) < 0 ||
        rpc_write(0, &handle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phEvent, sizeof(CUevent)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuIpcGetMemHandle) < 0 ||
        rpc_write(0, pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuIpcOpenMemHandle_v2) < 0 ||
        rpc_write(0, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &handle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuIpcCloseMemHandle) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpy) < 0 ||
        rpc_write(0, &dst, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &src, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyPeer) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstContext, sizeof(CUcontext)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcContext, sizeof(CUcontext)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyHtoD_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcHost, sizeof(const void*)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyDtoD_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyDtoA_v2) < 0 ||
        rpc_write(0, &dstArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &dstOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyAtoD_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyAtoH_v2) < 0 ||
        rpc_write(0, &dstHost, sizeof(void*)) < 0 ||
        rpc_write(0, &srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyAtoA_v2) < 0 ||
        rpc_write(0, &dstArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &dstOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &srcArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &srcOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyAsync) < 0 ||
        rpc_write(0, &dst, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &src, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyPeerAsync) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstContext, sizeof(CUcontext)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcContext, sizeof(CUcontext)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyHtoDAsync_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcHost, sizeof(const void*)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemcpyDtoDAsync_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ByteCount, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD8_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(0, &N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD16_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(0, &N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD32_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &N, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD2D8_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(0, &Width, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD2D16_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(0, &Width, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD2D32_v2) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &Width, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD8Async) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(0, &N, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD16Async) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(0, &N, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD32Async) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &N, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD2D8Async) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &uc, sizeof(unsigned char)) < 0 ||
        rpc_write(0, &Width, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD2D16Async) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &us, sizeof(unsigned short)) < 0 ||
        rpc_write(0, &Width, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemsetD2D32Async) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &dstPitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &ui, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &Width, sizeof(size_t)) < 0 ||
        rpc_write(0, &Height, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArrayCreate_v2) < 0 ||
        rpc_write(0, pHandle, sizeof(CUarray)) < 0 ||
        rpc_write(0, &pAllocateArray, sizeof(const CUDA_ARRAY_DESCRIPTOR*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHandle, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArrayGetDescriptor_v2) < 0 ||
        rpc_write(0, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_write(0, &hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArrayGetSparseProperties) < 0 ||
        rpc_write(0, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_write(0, &array, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMipmappedArrayGetSparseProperties) < 0 ||
        rpc_write(0, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_write(0, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArrayGetMemoryRequirements) < 0 ||
        rpc_write(0, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_write(0, &array, sizeof(CUarray)) < 0 ||
        rpc_write(0, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMipmappedArrayGetMemoryRequirements) < 0 ||
        rpc_write(0, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_write(0, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArrayGetPlane) < 0 ||
        rpc_write(0, pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &hArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &planeIdx, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArrayDestroy(CUarray hArray)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArrayDestroy) < 0 ||
        rpc_write(0, &hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArray3DCreate_v2) < 0 ||
        rpc_write(0, pHandle, sizeof(CUarray)) < 0 ||
        rpc_write(0, &pAllocateArray, sizeof(const CUDA_ARRAY3D_DESCRIPTOR*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHandle, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuArray3DGetDescriptor_v2) < 0 ||
        rpc_write(0, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_write(0, &hArray, sizeof(CUarray)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMipmappedArrayCreate) < 0 ||
        rpc_write(0, pHandle, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &pMipmappedArrayDesc, sizeof(const CUDA_ARRAY3D_DESCRIPTOR*)) < 0 ||
        rpc_write(0, &numMipmapLevels, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHandle, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMipmappedArrayGetLevel) < 0 ||
        rpc_write(0, pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &level, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMipmappedArrayDestroy) < 0 ||
        rpc_write(0, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAddressReserve) < 0 ||
        rpc_write(0, ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &alignment, sizeof(size_t)) < 0 ||
        rpc_write(0, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAddressFree) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemCreate) < 0 ||
        rpc_write(0, handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &prop, sizeof(const CUmemAllocationProp*)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemRelease) < 0 ||
        rpc_write(0, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemMap) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &offset, sizeof(size_t)) < 0 ||
        rpc_write(0, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemMapArrayAsync) < 0 ||
        rpc_write(0, mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemUnmap) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemSetAccess) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &desc, sizeof(const CUmemAccessDesc*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemGetAccess) < 0 ||
        rpc_write(0, flags, sizeof(unsigned long long)) < 0 ||
        rpc_write(0, &location, sizeof(const CUmemLocation*)) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemGetAllocationGranularity) < 0 ||
        rpc_write(0, granularity, sizeof(size_t)) < 0 ||
        rpc_write(0, &prop, sizeof(const CUmemAllocationProp*)) < 0 ||
        rpc_write(0, &option, sizeof(CUmemAllocationGranularity_flags)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, granularity, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemGetAllocationPropertiesFromHandle) < 0 ||
        rpc_write(0, prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_write(0, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemFreeAsync) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAllocAsync) < 0 ||
        rpc_write(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &bytesize, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolTrimTo) < 0 ||
        rpc_write(0, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(0, &minBytesToKeep, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolSetAccess) < 0 ||
        rpc_write(0, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(0, &map, sizeof(const CUmemAccessDesc*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolGetAccess) < 0 ||
        rpc_write(0, flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_write(0, &memPool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(0, location, sizeof(CUmemLocation)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_read(0, location, sizeof(CUmemLocation)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolCreate) < 0 ||
        rpc_write(0, pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(0, &poolProps, sizeof(const CUmemPoolProps*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolDestroy) < 0 ||
        rpc_write(0, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAllocFromPoolAsync) < 0 ||
        rpc_write(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &bytesize, sizeof(size_t)) < 0 ||
        rpc_write(0, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolExportPointer) < 0 ||
        rpc_write(0, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPoolImportPointer) < 0 ||
        rpc_write(0, ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(0, shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemPrefetchAsync) < 0 ||
        rpc_write(0, &devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdevice)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemAdvise) < 0 ||
        rpc_write(0, &devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &advice, sizeof(CUmem_advise)) < 0 ||
        rpc_write(0, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuMemRangeGetAttributes) < 0 ||
        rpc_write(0, data, sizeof(void*)) < 0 ||
        rpc_write(0, dataSizes, sizeof(size_t)) < 0 ||
        rpc_write(0, attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_write(0, &numAttributes, sizeof(size_t)) < 0 ||
        rpc_write(0, &devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, data, sizeof(void*)) < 0 ||
        rpc_read(0, dataSizes, sizeof(size_t)) < 0 ||
        rpc_read(0, attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuPointerSetAttribute) < 0 ||
        rpc_write(0, &value, sizeof(const void*)) < 0 ||
        rpc_write(0, &attribute, sizeof(CUpointer_attribute)) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuPointerGetAttributes) < 0 ||
        rpc_write(0, &numAttributes, sizeof(unsigned int)) < 0 ||
        rpc_write(0, attributes, sizeof(CUpointer_attribute)) < 0 ||
        rpc_write(0, data, sizeof(void*)) < 0 ||
        rpc_write(0, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, attributes, sizeof(CUpointer_attribute)) < 0 ||
        rpc_read(0, data, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamCreate) < 0 ||
        rpc_write(0, phStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phStream, sizeof(CUstream)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamCreateWithPriority) < 0 ||
        rpc_write(0, phStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phStream, sizeof(CUstream)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetPriority(CUstream hStream, int* priority)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamGetPriority) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, priority, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamGetFlags) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamGetId) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, streamId, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, streamId, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamGetCtx) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamWaitEvent) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamBeginCapture_v2) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuThreadExchangeStreamCaptureMode) < 0 ||
        rpc_write(0, mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamEndCapture) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamIsCapturing) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamUpdateCaptureDependencies) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamAttachMemAsync) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &length, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamQuery(CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamQuery) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamSynchronize(CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamSynchronize) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamDestroy_v2(CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamDestroy_v2) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamCopyAttributes(CUstream dst, CUstream src)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamCopyAttributes) < 0 ||
        rpc_write(0, &dst, sizeof(CUstream)) < 0 ||
        rpc_write(0, &src, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamGetAttribute) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &attr, sizeof(CUstreamAttrID)) < 0 ||
        rpc_write(0, value_out, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value_out, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamSetAttribute) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &attr, sizeof(CUstreamAttrID)) < 0 ||
        rpc_write(0, &value, sizeof(const CUstreamAttrValue*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventCreate) < 0 ||
        rpc_write(0, phEvent, sizeof(CUevent)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phEvent, sizeof(CUevent)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventRecord) < 0 ||
        rpc_write(0, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventRecordWithFlags) < 0 ||
        rpc_write(0, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventQuery(CUevent hEvent)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventQuery) < 0 ||
        rpc_write(0, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventSynchronize(CUevent hEvent)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventSynchronize) < 0 ||
        rpc_write(0, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventDestroy_v2(CUevent hEvent)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventDestroy_v2) < 0 ||
        rpc_write(0, &hEvent, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuEventElapsedTime) < 0 ||
        rpc_write(0, pMilliseconds, sizeof(float)) < 0 ||
        rpc_write(0, &hStart, sizeof(CUevent)) < 0 ||
        rpc_write(0, &hEnd, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pMilliseconds, sizeof(float)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuImportExternalMemory) < 0 ||
        rpc_write(0, extMem_out, sizeof(CUexternalMemory)) < 0 ||
        rpc_write(0, &memHandleDesc, sizeof(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, extMem_out, sizeof(CUexternalMemory)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuExternalMemoryGetMappedBuffer) < 0 ||
        rpc_write(0, devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_write(0, &bufferDesc, sizeof(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuExternalMemoryGetMappedMipmappedArray) < 0 ||
        rpc_write(0, mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_write(0, &mipmapDesc, sizeof(const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mipmap, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDestroyExternalMemory) < 0 ||
        rpc_write(0, &extMem, sizeof(CUexternalMemory)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuImportExternalSemaphore) < 0 ||
        rpc_write(0, extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_write(0, &semHandleDesc, sizeof(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuSignalExternalSemaphoresAsync) < 0 ||
        rpc_write(0, &extSemArray, sizeof(const CUexternalSemaphore*)) < 0 ||
        rpc_write(0, &paramsArray, sizeof(const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*)) < 0 ||
        rpc_write(0, &numExtSems, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuWaitExternalSemaphoresAsync) < 0 ||
        rpc_write(0, &extSemArray, sizeof(const CUexternalSemaphore*)) < 0 ||
        rpc_write(0, &paramsArray, sizeof(const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*)) < 0 ||
        rpc_write(0, &numExtSems, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDestroyExternalSemaphore) < 0 ||
        rpc_write(0, &extSem, sizeof(CUexternalSemaphore)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamWaitValue32_v2) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &value, sizeof(cuuint32_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamWaitValue64_v2) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &value, sizeof(cuuint64_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamWriteValue32_v2) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &value, sizeof(cuuint32_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamWriteValue64_v2) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &addr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &value, sizeof(cuuint64_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuStreamBatchMemOp_v2) < 0 ||
        rpc_write(0, &stream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(0, paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncGetAttribute) < 0 ||
        rpc_write(0, pi, sizeof(int)) < 0 ||
        rpc_write(0, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pi, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncSetAttribute) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(0, &value, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncSetCacheConfig) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &config, sizeof(CUfunc_cache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncSetSharedMemConfig) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &config, sizeof(CUsharedconfig)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncGetModule) < 0 ||
        rpc_write(0, hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, hmod, sizeof(CUmodule)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLaunchKernel) < 0 ||
        rpc_write(0, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &gridDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &gridDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &gridDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &blockDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &blockDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &blockDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &sharedMemBytes, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, &kernelParams, sizeof(void**)) < 0 ||
        rpc_write(0, &extra, sizeof(void**)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLaunchCooperativeKernel) < 0 ||
        rpc_write(0, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &gridDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &gridDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &gridDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &blockDimX, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &blockDimY, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &blockDimZ, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &sharedMemBytes, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_write(0, kernelParams, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, kernelParams, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLaunchCooperativeKernelMultiDevice) < 0 ||
        rpc_write(0, launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_write(0, &numDevices, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncSetBlockShape) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &x, sizeof(int)) < 0 ||
        rpc_write(0, &y, sizeof(int)) < 0 ||
        rpc_write(0, &z, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuFuncSetSharedSize) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &bytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuParamSetSize) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &numbytes, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuParamSeti) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &offset, sizeof(int)) < 0 ||
        rpc_write(0, &value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuParamSetf) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &offset, sizeof(int)) < 0 ||
        rpc_write(0, &value, sizeof(float)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunch(CUfunction f)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLaunch) < 0 ||
        rpc_write(0, &f, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLaunchGrid) < 0 ||
        rpc_write(0, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &grid_width, sizeof(int)) < 0 ||
        rpc_write(0, &grid_height, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuLaunchGridAsync) < 0 ||
        rpc_write(0, &f, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &grid_width, sizeof(int)) < 0 ||
        rpc_write(0, &grid_height, sizeof(int)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuParamSetTexRef) < 0 ||
        rpc_write(0, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &texunit, sizeof(int)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphCreate) < 0 ||
        rpc_write(0, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddKernelNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddKernelNode_v2) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_KERNEL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphKernelNodeGetParams_v2) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphKernelNodeSetParams_v2) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_KERNEL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddMemcpyNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &copyParams, sizeof(const CUDA_MEMCPY3D*)) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphMemcpyNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphMemcpyNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_MEMCPY3D*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddMemsetNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS*)) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphMemsetNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphMemsetNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_MEMSET_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddHostNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphHostNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphHostNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddChildGraphNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &childGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphChildGraphNodeGetGraph) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddEmptyNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddEventRecordNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphEventRecordNodeGetEvent) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, event_out, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event_out, sizeof(CUevent)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphEventRecordNodeSetEvent) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddEventWaitNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphEventWaitNodeGetEvent) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, event_out, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event_out, sizeof(CUevent)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphEventWaitNodeSetEvent) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddExternalSemaphoresSignalNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExternalSemaphoresSignalNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExternalSemaphoresSignalNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddExternalSemaphoresWaitNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_EXT_SEM_WAIT_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExternalSemaphoresWaitNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExternalSemaphoresWaitNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_EXT_SEM_WAIT_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddBatchMemOpNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_BATCH_MEM_OP_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphBatchMemOpNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphBatchMemOpNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_BATCH_MEM_OP_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecBatchMemOpNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_BATCH_MEM_OP_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddMemAllocNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &dependencies, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphMemAllocNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddMemFreeNode) < 0 ||
        rpc_write(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, dependencies, numDependencies * sizeof(const CUgraphNode)) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphMemFreeNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGraphMemTrim(CUdevice device)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGraphMemTrim) < 0 ||
        rpc_write(0, &device, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphClone) < 0 ||
        rpc_write(0, phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &originalGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphNodeFindInClone) < 0 ||
        rpc_write(0, phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hOriginalNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &hClonedGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphNodeGetType) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphGetNodes) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, nodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, numNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, numNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphGetRootNodes) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, numRootNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, numRootNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphGetEdges) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, from, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, to, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, numEdges, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, from, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, to, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, numEdges, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphNodeGetDependencies) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, numDependencies, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphNodeGetDependentNodes) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, dependentNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, numDependentNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dependentNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_read(0, numDependentNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphAddDependencies) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &from, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &to, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphRemoveDependencies) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &from, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &to, sizeof(const CUgraphNode*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDestroyNode(CUgraphNode hNode)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphDestroyNode) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphInstantiateWithFlags) < 0 ||
        rpc_write(0, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphInstantiateWithParams) < 0 ||
        rpc_write(0, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_read(0, instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecGetFlags) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, flags, sizeof(cuuint64_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(cuuint64_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecKernelNodeSetParams_v2) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_KERNEL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecMemcpyNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &copyParams, sizeof(const CUDA_MEMCPY3D*)) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecMemsetNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS*)) < 0 ||
        rpc_write(0, &ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecHostNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecChildGraphNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &childGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecEventRecordNodeSetEvent) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecEventWaitNodeSetEvent) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &event, sizeof(CUevent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const CUDA_EXT_SEM_WAIT_NODE_PARAMS*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphNodeSetEnabled) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphNodeGetEnabled) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphUpload) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphLaunch) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecDestroy) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDestroy(CUgraph hGraph)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphDestroy) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphExecUpdate_v2) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphKernelNodeCopyAttributes) < 0 ||
        rpc_write(0, &dst, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &src, sizeof(CUgraphNode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphKernelNodeGetAttribute) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
        rpc_write(0, value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphKernelNodeSetAttribute) < 0 ||
        rpc_write(0, &hNode, sizeof(CUgraphNode)) < 0 ||
        rpc_write(0, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
        rpc_write(0, &value, sizeof(const CUkernelNodeAttrValue*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphDebugDotPrint) < 0 ||
        rpc_write(0, &hGraph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &path, sizeof(const char*)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectRetain(CUuserObject object, unsigned int count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuUserObjectRetain) < 0 ||
        rpc_write(0, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuUserObjectRelease(CUuserObject object, unsigned int count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuUserObjectRelease) < 0 ||
        rpc_write(0, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphRetainUserObject) < 0 ||
        rpc_write(0, &graph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphReleaseUserObject) < 0 ||
        rpc_write(0, &graph, sizeof(CUgraph)) < 0 ||
        rpc_write(0, &object, sizeof(CUuserObject)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor) < 0 ||
        rpc_write(0, numBlocks, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &blockSize, sizeof(int)) < 0 ||
        rpc_write(0, &dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numBlocks, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) < 0 ||
        rpc_write(0, numBlocks, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &blockSize, sizeof(int)) < 0 ||
        rpc_write(0, &dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numBlocks, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuOccupancyAvailableDynamicSMemPerBlock) < 0 ||
        rpc_write(0, dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_write(0, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &numBlocks, sizeof(int)) < 0 ||
        rpc_write(0, &blockSize, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuOccupancyMaxPotentialClusterSize) < 0 ||
        rpc_write(0, clusterSize, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &config, sizeof(const CUlaunchConfig*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clusterSize, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuOccupancyMaxActiveClusters) < 0 ||
        rpc_write(0, numClusters, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(CUfunction)) < 0 ||
        rpc_write(0, &config, sizeof(const CUlaunchConfig*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numClusters, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetArray) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &hArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetMipmappedArray) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetAddress_v2) < 0 ||
        rpc_write(0, ByteOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &bytes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ByteOffset, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetAddress2D_v3) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &desc, sizeof(const CUDA_ARRAY_DESCRIPTOR*)) < 0 ||
        rpc_write(0, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &Pitch, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetFormat) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &fmt, sizeof(CUarray_format)) < 0 ||
        rpc_write(0, &NumPackedComponents, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetAddressMode) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &dim, sizeof(int)) < 0 ||
        rpc_write(0, &am, sizeof(CUaddress_mode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetFilterMode) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &fm, sizeof(CUfilter_mode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetMipmapFilterMode) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &fm, sizeof(CUfilter_mode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetMipmapLevelBias) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &bias, sizeof(float)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetMipmapLevelClamp) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &minMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(0, &maxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetMaxAnisotropy) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &maxAniso, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetBorderColor) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, pBorderColor, sizeof(float)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pBorderColor, sizeof(float)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefSetFlags) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetAddress_v2) < 0 ||
        rpc_write(0, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetArray) < 0 ||
        rpc_write(0, phArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetMipmappedArray) < 0 ||
        rpc_write(0, phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetAddressMode) < 0 ||
        rpc_write(0, pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_write(0, &dim, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetFilterMode) < 0 ||
        rpc_write(0, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetFormat) < 0 ||
        rpc_write(0, pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_write(0, pNumChannels, sizeof(int)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_read(0, pNumChannels, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetMipmapFilterMode) < 0 ||
        rpc_write(0, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetMipmapLevelBias) < 0 ||
        rpc_write(0, pbias, sizeof(float)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pbias, sizeof(float)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetMipmapLevelClamp) < 0 ||
        rpc_write(0, pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(0, pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_read(0, pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetMaxAnisotropy) < 0 ||
        rpc_write(0, pmaxAniso, sizeof(int)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pmaxAniso, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetBorderColor) < 0 ||
        rpc_write(0, pBorderColor, sizeof(float)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pBorderColor, sizeof(float)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefGetFlags) < 0 ||
        rpc_write(0, pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pFlags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefCreate(CUtexref* pTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefCreate) < 0 ||
        rpc_write(0, pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexRefDestroy(CUtexref hTexRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexRefDestroy) < 0 ||
        rpc_write(0, &hTexRef, sizeof(CUtexref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuSurfRefSetArray) < 0 ||
        rpc_write(0, &hSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_write(0, &hArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuSurfRefGetArray) < 0 ||
        rpc_write(0, phArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &hSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, phArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexObjectCreate) < 0 ||
        rpc_write(0, pTexObject, sizeof(CUtexObject)) < 0 ||
        rpc_write(0, &pResDesc, sizeof(const CUDA_RESOURCE_DESC*)) < 0 ||
        rpc_write(0, &pTexDesc, sizeof(const CUDA_TEXTURE_DESC*)) < 0 ||
        rpc_write(0, &pResViewDesc, sizeof(const CUDA_RESOURCE_VIEW_DESC*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pTexObject, sizeof(CUtexObject)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectDestroy(CUtexObject texObject)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexObjectDestroy) < 0 ||
        rpc_write(0, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexObjectGetResourceDesc) < 0 ||
        rpc_write(0, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(0, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexObjectGetTextureDesc) < 0 ||
        rpc_write(0, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_write(0, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuTexObjectGetResourceViewDesc) < 0 ||
        rpc_write(0, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_write(0, &texObject, sizeof(CUtexObject)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuSurfObjectCreate) < 0 ||
        rpc_write(0, pSurfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_write(0, &pResDesc, sizeof(const CUDA_RESOURCE_DESC*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pSurfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuSurfObjectDestroy) < 0 ||
        rpc_write(0, &surfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuSurfObjectGetResourceDesc) < 0 ||
        rpc_write(0, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_write(0, &surfObject, sizeof(CUsurfObject)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceCanAccessPeer) < 0 ||
        rpc_write(0, canAccessPeer, sizeof(int)) < 0 ||
        rpc_write(0, &dev, sizeof(CUdevice)) < 0 ||
        rpc_write(0, &peerDev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, canAccessPeer, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxEnablePeerAccess) < 0 ||
        rpc_write(0, &peerContext, sizeof(CUcontext)) < 0 ||
        rpc_write(0, &Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuCtxDisablePeerAccess) < 0 ||
        rpc_write(0, &peerContext, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuDeviceGetP2PAttribute) < 0 ||
        rpc_write(0, value, sizeof(int)) < 0 ||
        rpc_write(0, &attrib, sizeof(CUdevice_P2PAttribute)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(CUdevice)) < 0 ||
        rpc_write(0, &dstDevice, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsUnregisterResource) < 0 ||
        rpc_write(0, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsSubResourceGetMappedArray) < 0 ||
        rpc_write(0, pArray, sizeof(CUarray)) < 0 ||
        rpc_write(0, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(0, &arrayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &mipLevel, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsResourceGetMappedMipmappedArray) < 0 ||
        rpc_write(0, pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_write(0, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsResourceGetMappedPointer_v2) < 0 ||
        rpc_write(0, pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(0, pSize, sizeof(size_t)) < 0 ||
        rpc_write(0, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(0, pSize, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsResourceSetMapFlags_v2) < 0 ||
        rpc_write(0, &resource, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsMapResources) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(0, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream)
{
    CUresult return_value;
    if (rpc_start_request(0, RPC_cuGraphicsUnmapResources) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(0, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_write(0, &hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
}

cudaError_t cudaDeviceReset()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceReset) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSynchronize()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceSynchronize) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceSetLimit) < 0 ||
        rpc_write(0, &limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_write(0, &value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetLimit) < 0 ||
        rpc_write(0, pValue, sizeof(size_t)) < 0 ||
        rpc_write(0, &limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pValue, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const struct cudaChannelFormatDesc* fmtDesc, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetTexture1DLinearMaxWidth) < 0 ||
        rpc_write(0, maxWidthInElements, sizeof(size_t)) < 0 ||
        rpc_write(0, &fmtDesc, sizeof(const struct cudaChannelFormatDesc*)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, maxWidthInElements, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetCacheConfig) < 0 ||
        rpc_write(0, pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetStreamPriorityRange) < 0 ||
        rpc_write(0, leastPriority, sizeof(int)) < 0 ||
        rpc_write(0, greatestPriority, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, leastPriority, sizeof(int)) < 0 ||
        rpc_read(0, greatestPriority, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceSetCacheConfig) < 0 ||
        rpc_write(0, &cacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetSharedMemConfig) < 0 ||
        rpc_write(0, pConfig, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pConfig, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceSetSharedMemConfig) < 0 ||
        rpc_write(0, &config, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetByPCIBusId) < 0 ||
        rpc_write(0, device, sizeof(int)) < 0 ||
        rpc_write(0, &pciBusId, sizeof(const char*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetPCIBusId) < 0 ||
        rpc_write(0, pciBusId, sizeof(char)) < 0 ||
        rpc_write(0, &len, sizeof(int)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pciBusId, sizeof(char)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaIpcGetEventHandle) < 0 ||
        rpc_write(0, handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaIpcOpenEventHandle) < 0 ||
        rpc_write(0, event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(0, &handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaIpcOpenMemHandle) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, &handle, sizeof(cudaIpcMemHandle_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceFlushGPUDirectRDMAWrites) < 0 ||
        rpc_write(0, &target, sizeof(enum cudaFlushGPUDirectRDMAWritesTarget)) < 0 ||
        rpc_write(0, &scope, sizeof(enum cudaFlushGPUDirectRDMAWritesScope)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadExit()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadExit) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadSynchronize()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadSynchronize) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadSetLimit) < 0 ||
        rpc_write(0, &limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_write(0, &value, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadGetLimit(size_t* pValue, enum cudaLimit limit)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadGetLimit) < 0 ||
        rpc_write(0, pValue, sizeof(size_t)) < 0 ||
        rpc_write(0, &limit, sizeof(enum cudaLimit)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pValue, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache* pCacheConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadGetCacheConfig) < 0 ||
        rpc_write(0, pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadSetCacheConfig) < 0 ||
        rpc_write(0, &cacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetLastError()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetLastError) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaPeekAtLastError()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaPeekAtLastError) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDeviceCount(int* count)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetDeviceCount) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, count, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp* prop, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetDeviceProperties_v2) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, prop, sizeof(struct cudaDeviceProp)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetAttribute) < 0 ||
        rpc_write(0, value, sizeof(int)) < 0 ||
        rpc_write(0, &attr, sizeof(enum cudaDeviceAttr)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetDefaultMemPool) < 0 ||
        rpc_write(0, memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceSetMemPool) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetMemPool) < 0 ||
        rpc_write(0, memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGetP2PAttribute) < 0 ||
        rpc_write(0, value, sizeof(int)) < 0 ||
        rpc_write(0, &attr, sizeof(enum cudaDeviceP2PAttr)) < 0 ||
        rpc_write(0, &srcDevice, sizeof(int)) < 0 ||
        rpc_write(0, &dstDevice, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaChooseDevice) < 0 ||
        rpc_write(0, device, sizeof(int)) < 0 ||
        rpc_write(0, &prop, sizeof(const struct cudaDeviceProp*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaInitDevice) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_write(0, &deviceFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDevice(int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaSetDevice) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDevice(int* device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetDevice) < 0 ||
        rpc_write(0, device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetValidDevices(int* device_arr, int len)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaSetValidDevices) < 0 ||
        rpc_write(0, device_arr, sizeof(int)) < 0 ||
        rpc_write(0, &len, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, device_arr, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaSetDeviceFlags) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetDeviceFlags) < 0 ||
        rpc_write(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamCreate) < 0 ||
        rpc_write(0, pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamCreateWithFlags) < 0 ||
        rpc_write(0, pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamCreateWithPriority) < 0 ||
        rpc_write(0, pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamGetPriority) < 0 ||
        rpc_write(0, &hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, priority, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, priority, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamGetFlags) < 0 ||
        rpc_write(0, &hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamGetId) < 0 ||
        rpc_write(0, &hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, streamId, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, streamId, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaCtxResetPersistingL2Cache()
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaCtxResetPersistingL2Cache) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamCopyAttributes) < 0 ||
        rpc_write(0, &dst, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &src, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamGetAttribute) < 0 ||
        rpc_write(0, &hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &attr, sizeof(cudaLaunchAttributeID)) < 0 ||
        rpc_write(0, value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamSetAttribute) < 0 ||
        rpc_write(0, &hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &attr, sizeof(cudaLaunchAttributeID)) < 0 ||
        rpc_write(0, &value, sizeof(const cudaLaunchAttributeValue*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamDestroy) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamWaitEvent) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamSynchronize) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamQuery) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamBeginCapture) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaThreadExchangeStreamCaptureMode) < 0 ||
        rpc_write(0, mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamEndCapture) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamIsCapturing) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamGetCaptureInfo_v2) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, captureStatus_out, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_read(0, id_out, sizeof(unsigned long long)) < 0 ||
        rpc_read(0, graph_out, sizeof(cudaGraph_t)) < 0 ||
        rpc_read(0, numDependencies_out, sizeof(size_t)) < 0 ||
        rpc_read(0, dependencies_out, *numDependencies_out * sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaStreamUpdateCaptureDependencies) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, dependencies, numDependencies * sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventCreate) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventCreateWithFlags) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventRecord) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventRecordWithFlags) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventQuery) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventSynchronize) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventDestroy) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaEventElapsedTime) < 0 ||
        rpc_write(0, &start, sizeof(cudaEvent_t)) < 0 ||
        rpc_write(0, &end, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ms, sizeof(float)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc* bufferDesc)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaExternalMemoryGetMappedBuffer) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, &extMem, sizeof(cudaExternalMemory_t)) < 0 ||
        rpc_write(0, &bufferDesc, sizeof(const struct cudaExternalMemoryBufferDesc*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaExternalMemoryGetMappedMipmappedArray) < 0 ||
        rpc_write(0, mipmap, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_write(0, &extMem, sizeof(cudaExternalMemory_t)) < 0 ||
        rpc_write(0, &mipmapDesc, sizeof(const struct cudaExternalMemoryMipmappedArrayDesc*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mipmap, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDestroyExternalMemory) < 0 ||
        rpc_write(0, &extMem, sizeof(cudaExternalMemory_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const struct cudaExternalSemaphoreHandleDesc* semHandleDesc)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaImportExternalSemaphore) < 0 ||
        rpc_write(0, extSem_out, sizeof(cudaExternalSemaphore_t)) < 0 ||
        rpc_write(0, &semHandleDesc, sizeof(const struct cudaExternalSemaphoreHandleDesc*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, extSem_out, sizeof(cudaExternalSemaphore_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaSignalExternalSemaphoresAsync_v2) < 0 ||
        rpc_write(0, &extSemArray, sizeof(const cudaExternalSemaphore_t*)) < 0 ||
        rpc_write(0, &paramsArray, sizeof(const struct cudaExternalSemaphoreSignalParams*)) < 0 ||
        rpc_write(0, &numExtSems, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaWaitExternalSemaphoresAsync_v2) < 0 ||
        rpc_write(0, &extSemArray, sizeof(const cudaExternalSemaphore_t*)) < 0 ||
        rpc_write(0, &paramsArray, sizeof(const struct cudaExternalSemaphoreWaitParams*)) < 0 ||
        rpc_write(0, &numExtSems, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDestroyExternalSemaphore) < 0 ||
        rpc_write(0, &extSem, sizeof(cudaExternalSemaphore_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaLaunchKernelExC) < 0 ||
        rpc_write(0, &config, sizeof(const cudaLaunchConfig_t*)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, args, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, args, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaLaunchCooperativeKernel) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &gridDim, sizeof(dim3)) < 0 ||
        rpc_write(0, &blockDim, sizeof(dim3)) < 0 ||
        rpc_write(0, args, sizeof(void*)) < 0 ||
        rpc_write(0, &sharedMem, sizeof(size_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, args, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaLaunchCooperativeKernelMultiDevice) < 0 ||
        rpc_write(0, launchParamsList, sizeof(struct cudaLaunchParams)) < 0 ||
        rpc_write(0, &numDevices, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, launchParamsList, sizeof(struct cudaLaunchParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFuncSetCacheConfig) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &cacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFuncSetSharedMemConfig) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &config, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFuncGetAttributes) < 0 ||
        rpc_write(0, attr, sizeof(struct cudaFuncAttributes)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, attr, sizeof(struct cudaFuncAttributes)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFuncSetAttribute) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &attr, sizeof(enum cudaFuncAttribute)) < 0 ||
        rpc_write(0, &value, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDoubleForDevice(double* d)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaSetDoubleForDevice) < 0 ||
        rpc_write(0, d, sizeof(double)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, d, sizeof(double)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaSetDoubleForHost(double* d)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaSetDoubleForHost) < 0 ||
        rpc_write(0, d, sizeof(double)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, d, sizeof(double)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessor) < 0 ||
        rpc_write(0, numBlocks, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &blockSize, sizeof(int)) < 0 ||
        rpc_write(0, &dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numBlocks, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaOccupancyAvailableDynamicSMemPerBlock) < 0 ||
        rpc_write(0, dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &numBlocks, sizeof(int)) < 0 ||
        rpc_write(0, &blockSize, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) < 0 ||
        rpc_write(0, numBlocks, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &blockSize, sizeof(int)) < 0 ||
        rpc_write(0, &dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numBlocks, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaOccupancyMaxPotentialClusterSize(int* clusterSize, const void* func, const cudaLaunchConfig_t* launchConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaOccupancyMaxPotentialClusterSize) < 0 ||
        rpc_write(0, clusterSize, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &launchConfig, sizeof(const cudaLaunchConfig_t*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, clusterSize, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, const void* func, const cudaLaunchConfig_t* launchConfig)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaOccupancyMaxActiveClusters) < 0 ||
        rpc_write(0, numClusters, sizeof(int)) < 0 ||
        rpc_write(0, &func, sizeof(const void*)) < 0 ||
        rpc_write(0, &launchConfig, sizeof(const cudaLaunchConfig_t*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, numClusters, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocManaged) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMalloc) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocHost(void** ptr, size_t size)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocHost) < 0 ||
        rpc_write(0, ptr, sizeof(void*)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ptr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocPitch) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, pitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &width, sizeof(size_t)) < 0 ||
        rpc_write(0, &height, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_read(0, pitch, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocArray) < 0 ||
        rpc_write(0, array, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &desc, sizeof(const struct cudaChannelFormatDesc*)) < 0 ||
        rpc_write(0, &width, sizeof(size_t)) < 0 ||
        rpc_write(0, &height, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, array, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFree(void* devPtr)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFree) < 0 ||
        rpc_write(0, &devPtr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeHost(void* ptr)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFreeHost) < 0 ||
        rpc_write(0, &ptr, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeArray(cudaArray_t array)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFreeArray) < 0 ||
        rpc_write(0, &array, sizeof(cudaArray_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaFreeMipmappedArray) < 0 ||
        rpc_write(0, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaHostAlloc) < 0 ||
        rpc_write(0, pHost, sizeof(void*)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pHost, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMalloc3D) < 0 ||
        rpc_write(0, pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_write(0, &extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMalloc3DArray) < 0 ||
        rpc_write(0, array, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &desc, sizeof(const struct cudaChannelFormatDesc*)) < 0 ||
        rpc_write(0, &extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, array, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocMipmappedArray) < 0 ||
        rpc_write(0, mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_write(0, &desc, sizeof(const struct cudaChannelFormatDesc*)) < 0 ||
        rpc_write(0, &extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(0, &numLevels, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetMipmappedArrayLevel) < 0 ||
        rpc_write(0, levelArray, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &mipmappedArray, sizeof(cudaMipmappedArray_const_t)) < 0 ||
        rpc_write(0, &level, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, levelArray, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy3D) < 0 ||
        rpc_write(0, &p, sizeof(const struct cudaMemcpy3DParms*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy3DPeer) < 0 ||
        rpc_write(0, &p, sizeof(const struct cudaMemcpy3DPeerParms*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy3DAsync) < 0 ||
        rpc_write(0, &p, sizeof(const struct cudaMemcpy3DParms*)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy3DPeerAsync) < 0 ||
        rpc_write(0, &p, sizeof(const struct cudaMemcpy3DPeerParms*)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemGetInfo) < 0 ||
        rpc_write(0, free, sizeof(size_t)) < 0 ||
        rpc_write(0, total, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, free, sizeof(size_t)) < 0 ||
        rpc_read(0, total, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaArrayGetInfo) < 0 ||
        rpc_write(0, desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_write(0, extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &array, sizeof(cudaArray_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_read(0, extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_read(0, flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaArrayGetPlane) < 0 ||
        rpc_write(0, pPlaneArray, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &hArray, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &planeIdx, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pPlaneArray, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaArrayGetMemoryRequirements) < 0 ||
        rpc_write(0, memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_write(0, &array, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMipmappedArrayGetMemoryRequirements) < 0 ||
        rpc_write(0, memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_write(0, &mipmap, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaArray_t array)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaArrayGetSparseProperties) < 0 ||
        rpc_write(0, sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_write(0, &array, sizeof(cudaArray_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMipmappedArrayGetSparseProperties) < 0 ||
        rpc_write(0, sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_write(0, &mipmap, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy2DToArray) < 0 ||
        rpc_write(0, &dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &wOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &spitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &width, sizeof(size_t)) < 0 ||
        rpc_write(0, &height, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy2DArrayToArray) < 0 ||
        rpc_write(0, &dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &wOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(0, &src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(0, &wOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(0, &width, sizeof(size_t)) < 0 ||
        rpc_write(0, &height, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpyToSymbol) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &offset, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpy2DToArrayAsync) < 0 ||
        rpc_write(0, &dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &wOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &spitch, sizeof(size_t)) < 0 ||
        rpc_write(0, &width, sizeof(size_t)) < 0 ||
        rpc_write(0, &height, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpyToSymbolAsync) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &offset, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemset3D) < 0 ||
        rpc_write(0, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_write(0, &value, sizeof(int)) < 0 ||
        rpc_write(0, &extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemset3DAsync) < 0 ||
        rpc_write(0, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_write(0, &value, sizeof(int)) < 0 ||
        rpc_write(0, &extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetSymbolAddress) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetSymbolSize) < 0 ||
        rpc_write(0, size, sizeof(size_t)) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, size, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPrefetchAsync) < 0 ||
        rpc_write(0, &devPtr, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &dstDevice, sizeof(int)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemAdvise) < 0 ||
        rpc_write(0, &devPtr, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &advice, sizeof(enum cudaMemoryAdvise)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, enum cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemRangeGetAttributes) < 0 ||
        rpc_write(0, data, sizeof(void*)) < 0 ||
        rpc_write(0, dataSizes, sizeof(size_t)) < 0 ||
        rpc_write(0, attributes, sizeof(enum cudaMemRangeAttribute)) < 0 ||
        rpc_write(0, &numAttributes, sizeof(size_t)) < 0 ||
        rpc_write(0, &devPtr, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, data, sizeof(void*)) < 0 ||
        rpc_read(0, dataSizes, sizeof(size_t)) < 0 ||
        rpc_read(0, attributes, sizeof(enum cudaMemRangeAttribute)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpyToArray) < 0 ||
        rpc_write(0, &dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &wOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpyArrayToArray) < 0 ||
        rpc_write(0, &dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &wOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffsetDst, sizeof(size_t)) < 0 ||
        rpc_write(0, &src, sizeof(cudaArray_const_t)) < 0 ||
        rpc_write(0, &wOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffsetSrc, sizeof(size_t)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemcpyToArrayAsync) < 0 ||
        rpc_write(0, &dst, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &wOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &hOffset, sizeof(size_t)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocAsync) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &hStream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPoolTrimTo) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, &minBytesToKeep, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc* descList, size_t count)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPoolSetAccess) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, &descList, sizeof(const struct cudaMemAccessDesc*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags* flags, cudaMemPool_t memPool, struct cudaMemLocation* location)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPoolGetAccess) < 0 ||
        rpc_write(0, flags, sizeof(enum cudaMemAccessFlags)) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, location, sizeof(struct cudaMemLocation)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(enum cudaMemAccessFlags)) < 0 ||
        rpc_read(0, location, sizeof(struct cudaMemLocation)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const struct cudaMemPoolProps* poolProps)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPoolCreate) < 0 ||
        rpc_write(0, memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, &poolProps, sizeof(const struct cudaMemPoolProps*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPoolDestroy) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMallocFromPoolAsync) < 0 ||
        rpc_write(0, ptr, sizeof(void*)) < 0 ||
        rpc_write(0, &size, sizeof(size_t)) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ptr, sizeof(void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData* exportData)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaMemPoolImportPointer) < 0 ||
        rpc_write(0, ptr, sizeof(void*)) < 0 ||
        rpc_write(0, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(0, exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ptr, sizeof(void*)) < 0 ||
        rpc_read(0, exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaPointerGetAttributes) < 0 ||
        rpc_write(0, attributes, sizeof(struct cudaPointerAttributes)) < 0 ||
        rpc_write(0, &ptr, sizeof(const void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, attributes, sizeof(struct cudaPointerAttributes)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceCanAccessPeer) < 0 ||
        rpc_write(0, canAccessPeer, sizeof(int)) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_write(0, &peerDevice, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, canAccessPeer, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceEnablePeerAccess) < 0 ||
        rpc_write(0, &peerDevice, sizeof(int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceDisablePeerAccess) < 0 ||
        rpc_write(0, &peerDevice, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsUnregisterResource) < 0 ||
        rpc_write(0, &resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsResourceSetMapFlags) < 0 ||
        rpc_write(0, &resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsMapResources) < 0 ||
        rpc_write(0, &count, sizeof(int)) < 0 ||
        rpc_write(0, resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsUnmapResources) < 0 ||
        rpc_write(0, &count, sizeof(int)) < 0 ||
        rpc_write(0, resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsResourceGetMappedPointer) < 0 ||
        rpc_write(0, devPtr, sizeof(void*)) < 0 ||
        rpc_write(0, size, sizeof(size_t)) < 0 ||
        rpc_write(0, &resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, devPtr, sizeof(void*)) < 0 ||
        rpc_read(0, size, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsSubResourceGetMappedArray) < 0 ||
        rpc_write(0, array, sizeof(cudaArray_t)) < 0 ||
        rpc_write(0, &resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_write(0, &arrayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &mipLevel, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, array, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphicsResourceGetMappedMipmappedArray) < 0 ||
        rpc_write(0, mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_write(0, &resource, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc* desc, cudaArray_const_t array)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetChannelDesc) < 0 ||
        rpc_write(0, desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_write(0, &array, sizeof(cudaArray_const_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const struct cudaResourceDesc* pResDesc, const struct cudaTextureDesc* pTexDesc, const struct cudaResourceViewDesc* pResViewDesc)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaCreateTextureObject) < 0 ||
        rpc_write(0, pTexObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_write(0, &pResDesc, sizeof(const struct cudaResourceDesc*)) < 0 ||
        rpc_write(0, &pTexDesc, sizeof(const struct cudaTextureDesc*)) < 0 ||
        rpc_write(0, &pResViewDesc, sizeof(const struct cudaResourceViewDesc*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pTexObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDestroyTextureObject) < 0 ||
        rpc_write(0, &texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaTextureObject_t texObject)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetTextureObjectResourceDesc) < 0 ||
        rpc_write(0, pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_write(0, &texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetTextureObjectTextureDesc) < 0 ||
        rpc_write(0, pTexDesc, sizeof(struct cudaTextureDesc)) < 0 ||
        rpc_write(0, &texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pTexDesc, sizeof(struct cudaTextureDesc)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetTextureObjectResourceViewDesc) < 0 ||
        rpc_write(0, pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0 ||
        rpc_write(0, &texObject, sizeof(cudaTextureObject_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const struct cudaResourceDesc* pResDesc)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaCreateSurfaceObject) < 0 ||
        rpc_write(0, pSurfObject, sizeof(cudaSurfaceObject_t)) < 0 ||
        rpc_write(0, &pResDesc, sizeof(const struct cudaResourceDesc*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pSurfObject, sizeof(cudaSurfaceObject_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDestroySurfaceObject) < 0 ||
        rpc_write(0, &surfObject, sizeof(cudaSurfaceObject_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetSurfaceObjectResourceDesc) < 0 ||
        rpc_write(0, pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_write(0, &surfObject, sizeof(cudaSurfaceObject_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDriverGetVersion(int* driverVersion)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDriverGetVersion) < 0 ||
        rpc_write(0, driverVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, driverVersion, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaRuntimeGetVersion) < 0 ||
        rpc_write(0, runtimeVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, runtimeVersion, sizeof(int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphCreate) < 0 ||
        rpc_write(0, pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaKernelNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddKernelNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaKernelNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphKernelNodeGetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphKernelNodeSetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaKernelNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphKernelNodeCopyAttributes) < 0 ||
        rpc_write(0, &hSrc, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &hDst, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphKernelNodeGetAttribute) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &attr, sizeof(cudaLaunchAttributeID)) < 0 ||
        rpc_write(0, value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphKernelNodeSetAttribute) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &attr, sizeof(cudaLaunchAttributeID)) < 0 ||
        rpc_write(0, &value, sizeof(const cudaLaunchAttributeValue*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms* pCopyParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddMemcpyNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &pCopyParams, sizeof(const struct cudaMemcpy3DParms*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddMemcpyNodeToSymbol) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &offset, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphMemcpyNodeGetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphMemcpyNodeSetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaMemcpy3DParms*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphMemcpyNodeSetParamsToSymbol) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &offset, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemsetParams* pMemsetParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddMemsetNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &pMemsetParams, sizeof(const struct cudaMemsetParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphMemsetNodeGetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNodeParams, sizeof(struct cudaMemsetParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pNodeParams, sizeof(struct cudaMemsetParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphMemsetNodeSetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaMemsetParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaHostNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddHostNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaHostNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphHostNodeGetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNodeParams, sizeof(struct cudaHostNodeParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pNodeParams, sizeof(struct cudaHostNodeParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphHostNodeSetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaHostNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddChildGraphNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &childGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphChildGraphNodeGetGraph) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddEmptyNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddEventRecordNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphEventRecordNodeGetEvent) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphEventRecordNodeSetEvent) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddEventWaitNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphEventWaitNodeGetEvent) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphEventWaitNodeSetEvent) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddExternalSemaphoresSignalNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const struct cudaExternalSemaphoreSignalNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams* params_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExternalSemaphoresSignalNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExternalSemaphoresSignalNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const struct cudaExternalSemaphoreSignalNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddExternalSemaphoresWaitNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const struct cudaExternalSemaphoreWaitNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams* params_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExternalSemaphoresWaitNodeGetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExternalSemaphoresWaitNodeSetParams) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const struct cudaExternalSemaphoreWaitNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddMemAllocNode) < 0 ||
        rpc_write(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &pDependencies, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_write(0, nodeParams, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, nodeParams, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams* params_out)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphMemAllocNodeGetParams) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, params_out, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, params_out, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaDeviceGraphMemTrim(int device)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaDeviceGraphMemTrim) < 0 ||
        rpc_write(0, &device, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphClone) < 0 ||
        rpc_write(0, pGraphClone, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &originalGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphClone, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphNodeFindInClone) < 0 ||
        rpc_write(0, pNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &originalNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &clonedGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* pType)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphNodeGetType) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pType, sizeof(enum cudaGraphNodeType)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pType, sizeof(enum cudaGraphNodeType)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphGetNodes) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, nodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, numNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, nodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, numNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphGetRootNodes) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, pRootNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNumRootNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pRootNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, pNumRootNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphGetEdges) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, from, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, to, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, numEdges, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, from, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, to, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, numEdges, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphNodeGetDependencies) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pDependencies, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNumDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pDependencies, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, pNumDependencies, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphNodeGetDependentNodes) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pDependentNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, pNumDependentNodes, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pDependentNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_read(0, pNumDependentNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphAddDependencies) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &from, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &to, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphRemoveDependencies) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &from, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &to, sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_write(0, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphDestroyNode) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphInstantiate) < 0 ||
        rpc_write(0, pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphInstantiateWithFlags) < 0 ||
        rpc_write(0, pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphInstantiateWithParams) < 0 ||
        rpc_write(0, pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_read(0, instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecGetFlags) < 0 ||
        rpc_write(0, &graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, flags, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, flags, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecKernelNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaKernelNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecMemcpyNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaMemcpy3DParms*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecMemcpyNodeSetParamsToSymbol) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &symbol, sizeof(const void*)) < 0 ||
        rpc_write(0, &src, sizeof(const void*)) < 0 ||
        rpc_write(0, &count, sizeof(size_t)) < 0 ||
        rpc_write(0, &offset, sizeof(size_t)) < 0 ||
        rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecMemsetNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaMemsetParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecHostNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &pNodeParams, sizeof(const struct cudaHostNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecChildGraphNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &node, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &childGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecEventRecordNodeSetEvent) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecEventWaitNodeSetEvent) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecExternalSemaphoresSignalNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const struct cudaExternalSemaphoreSignalNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams* nodeParams)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecExternalSemaphoresWaitNodeSetParams) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &nodeParams, sizeof(const struct cudaExternalSemaphoreWaitNodeParams*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphNodeSetEnabled) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphNodeGetEnabled) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(0, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecUpdate) < 0 ||
        rpc_write(0, &hGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &hGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphUpload) < 0 ||
        rpc_write(0, &graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphLaunch) < 0 ||
        rpc_write(0, &graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphExecDestroy) < 0 ||
        rpc_write(0, &graphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphDestroy) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphDebugDotPrint) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &path, sizeof(const char*)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaUserObjectRetain) < 0 ||
        rpc_write(0, &object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaUserObjectRelease) < 0 ||
        rpc_write(0, &object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphRetainUserObject) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGraphReleaseUserObject) < 0 ||
        rpc_write(0, &graph, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(0, &object, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(0, &count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult* driverStatus)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetDriverEntryPoint) < 0 ||
        rpc_write(0, &symbol, sizeof(const char*)) < 0 ||
        rpc_write(0, funcPtr, sizeof(void*)) < 0 ||
        rpc_write(0, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_write(0, driverStatus, sizeof(enum cudaDriverEntryPointQueryResult)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, funcPtr, sizeof(void*)) < 0 ||
        rpc_read(0, driverStatus, sizeof(enum cudaDriverEntryPointQueryResult)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetExportTable) < 0 ||
        rpc_write(0, ppExportTable, sizeof(const void*)) < 0 ||
        rpc_write(0, &pExportTableId, sizeof(const cudaUUID_t*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, ppExportTable, sizeof(const void*)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr)
{
    cudaError_t return_value;
    if (rpc_start_request(0, RPC_cudaGetFuncBySymbol) < 0 ||
        rpc_write(0, functionPtr, sizeof(cudaFunction_t)) < 0 ||
        rpc_write(0, &symbolPtr, sizeof(const void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, functionPtr, sizeof(cudaFunction_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return cudaErrorDevicesUnavailable;
    return return_value;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle)
{
    cublasStatus_t return_value;
    if (rpc_start_request(0, RPC_cublasCreate_v2) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, handle, sizeof(cublasHandle_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUBLAS_STATUS_NOT_INITIALIZED;
    return return_value;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
{
    cublasStatus_t return_value;
    if (rpc_start_request(0, RPC_cublasDestroy_v2) < 0 ||
        rpc_write(0, &handle, sizeof(cublasHandle_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUBLAS_STATUS_NOT_INITIALIZED;
    return return_value;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    cublasStatus_t return_value;
    if (rpc_start_request(0, RPC_cublasSgemm_v2) < 0 ||
        rpc_write(0, &handle, sizeof(cublasHandle_t)) < 0 ||
        rpc_write(0, &transa, sizeof(cublasOperation_t)) < 0 ||
        rpc_write(0, &transb, sizeof(cublasOperation_t)) < 0 ||
        rpc_write(0, &m, sizeof(int)) < 0 ||
        rpc_write(0, &n, sizeof(int)) < 0 ||
        rpc_write(0, &k, sizeof(int)) < 0 ||
        rpc_write(0, &alpha, sizeof(const float*)) < 0 ||
        (alpha != nullptr && rpc_write(0, alpha, sizeof(const float)) < 0) ||
        rpc_write(0, &A, sizeof(const float*)) < 0 ||
        rpc_write(0, &lda, sizeof(int)) < 0 ||
        rpc_write(0, &B, sizeof(const float*)) < 0 ||
        rpc_write(0, &ldb, sizeof(int)) < 0 ||
        rpc_write(0, &beta, sizeof(const float*)) < 0 ||
        (beta != nullptr && rpc_write(0, beta, sizeof(const float)) < 0) ||
        rpc_write(0, &C, sizeof(float*)) < 0 ||
        rpc_write(0, &ldc, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUBLAS_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnCreate) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, handle, sizeof(cudnnHandle_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnDestroy) < 0 ||
        rpc_write(0, &handle, sizeof(cudnnHandle_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnCreateTensorDescriptor) < 0 ||
        rpc_write(0, tensorDesc, sizeof(cudnnTensorDescriptor_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, tensorDesc, sizeof(cudnnTensorDescriptor_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnSetTensor4dDescriptor) < 0 ||
        rpc_write(0, &tensorDesc, sizeof(cudnnTensorDescriptor_t)) < 0 ||
        rpc_write(0, &format, sizeof(cudnnTensorFormat_t)) < 0 ||
        rpc_write(0, &dataType, sizeof(cudnnDataType_t)) < 0 ||
        rpc_write(0, &n, sizeof(int)) < 0 ||
        rpc_write(0, &c, sizeof(int)) < 0 ||
        rpc_write(0, &h, sizeof(int)) < 0 ||
        rpc_write(0, &w, sizeof(int)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnCreateActivationDescriptor) < 0 ||
        rpc_write(0, activationDesc, sizeof(cudnnActivationDescriptor_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, activationDesc, sizeof(cudnnActivationDescriptor_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnSetActivationDescriptor) < 0 ||
        rpc_write(0, &activationDesc, sizeof(cudnnActivationDescriptor_t)) < 0 ||
        rpc_write(0, &mode, sizeof(cudnnActivationMode_t)) < 0 ||
        rpc_write(0, &reluNanOpt, sizeof(cudnnNanPropagation_t)) < 0 ||
        rpc_write(0, &coef, sizeof(double)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, &activationDesc, sizeof(cudnnActivationDescriptor_t)) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y)
{
    cudnnStatus_t return_value;
    if (rpc_start_request(0, RPC_cudnnActivationForward) < 0 ||
        rpc_write(0, &handle, sizeof(cudnnHandle_t)) < 0 ||
        rpc_write(0, &activationDesc, sizeof(cudnnActivationDescriptor_t)) < 0 ||
        rpc_write(0, &alpha, sizeof(const void*)) < 0 ||
        rpc_write(0, &xDesc, sizeof(cudnnTensorDescriptor_t)) < 0 ||
        rpc_write(0, &x, sizeof(const void*)) < 0 ||
        rpc_write(0, &beta, sizeof(const void*)) < 0 ||
        rpc_write(0, &yDesc, sizeof(cudnnTensorDescriptor_t)) < 0 ||
        rpc_write(0, &y, sizeof(void*)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_response(0, &return_value) < 0)
        return CUDNN_STATUS_NOT_INITIALIZED;
    return return_value;
}

std::unordered_map<std::string, void *> functionMap = {
    {"__cudaRegisterVar", (void *)__cudaRegisterVar},
    {"__cudaRegisterFunction", (void *)__cudaRegisterFunction},
    {"__cudaRegisterFatBinary", (void *)__cudaRegisterFatBinary},
    {"__cudaRegisterFatBinaryEnd", (void *)__cudaRegisterFatBinaryEnd},
    {"__cudaPushCallConfiguration", (void *)__cudaPushCallConfiguration},
    {"__cudaPopCallConfiguration", (void *)__cudaPopCallConfiguration},
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
    {"cuMemcpy", (void *)cuMemcpy},
    {"cuMemcpyPeer", (void *)cuMemcpyPeer},
    {"cuMemcpyHtoD_v2", (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyDtoD_v2", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyAsync", (void *)cuMemcpyAsync},
    {"cuMemcpyPeerAsync", (void *)cuMemcpyPeerAsync},
    {"cuMemcpyHtoDAsync_v2", (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyDtoDAsync_v2", (void *)cuMemcpyDtoDAsync_v2},
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
    {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
    {"cuArrayGetDescriptor_v2", (void *)cuArrayGetDescriptor_v2},
    {"cuArrayGetSparseProperties", (void *)cuArrayGetSparseProperties},
    {"cuMipmappedArrayGetSparseProperties", (void *)cuMipmappedArrayGetSparseProperties},
    {"cuArrayGetMemoryRequirements", (void *)cuArrayGetMemoryRequirements},
    {"cuMipmappedArrayGetMemoryRequirements", (void *)cuMipmappedArrayGetMemoryRequirements},
    {"cuArrayGetPlane", (void *)cuArrayGetPlane},
    {"cuArrayDestroy", (void *)cuArrayDestroy},
    {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
    {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
    {"cuMipmappedArrayCreate", (void *)cuMipmappedArrayCreate},
    {"cuMipmappedArrayGetLevel", (void *)cuMipmappedArrayGetLevel},
    {"cuMipmappedArrayDestroy", (void *)cuMipmappedArrayDestroy},
    {"cuMemAddressReserve", (void *)cuMemAddressReserve},
    {"cuMemAddressFree", (void *)cuMemAddressFree},
    {"cuMemCreate", (void *)cuMemCreate},
    {"cuMemRelease", (void *)cuMemRelease},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemMapArrayAsync", (void *)cuMemMapArrayAsync},
    {"cuMemUnmap", (void *)cuMemUnmap},
    {"cuMemSetAccess", (void *)cuMemSetAccess},
    {"cuMemGetAccess", (void *)cuMemGetAccess},
    {"cuMemGetAllocationGranularity", (void *)cuMemGetAllocationGranularity},
    {"cuMemGetAllocationPropertiesFromHandle", (void *)cuMemGetAllocationPropertiesFromHandle},
    {"cuMemFreeAsync", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync", (void *)cuMemAllocAsync},
    {"cuMemPoolTrimTo", (void *)cuMemPoolTrimTo},
    {"cuMemPoolSetAccess", (void *)cuMemPoolSetAccess},
    {"cuMemPoolGetAccess", (void *)cuMemPoolGetAccess},
    {"cuMemPoolCreate", (void *)cuMemPoolCreate},
    {"cuMemPoolDestroy", (void *)cuMemPoolDestroy},
    {"cuMemAllocFromPoolAsync", (void *)cuMemAllocFromPoolAsync},
    {"cuMemPoolExportPointer", (void *)cuMemPoolExportPointer},
    {"cuMemPoolImportPointer", (void *)cuMemPoolImportPointer},
    {"cuMemPrefetchAsync", (void *)cuMemPrefetchAsync},
    {"cuMemAdvise", (void *)cuMemAdvise},
    {"cuMemRangeGetAttributes", (void *)cuMemRangeGetAttributes},
    {"cuPointerSetAttribute", (void *)cuPointerSetAttribute},
    {"cuPointerGetAttributes", (void *)cuPointerGetAttributes},
    {"cuStreamCreate", (void *)cuStreamCreate},
    {"cuStreamCreateWithPriority", (void *)cuStreamCreateWithPriority},
    {"cuStreamGetPriority", (void *)cuStreamGetPriority},
    {"cuStreamGetFlags", (void *)cuStreamGetFlags},
    {"cuStreamGetId", (void *)cuStreamGetId},
    {"cuStreamGetCtx", (void *)cuStreamGetCtx},
    {"cuStreamWaitEvent", (void *)cuStreamWaitEvent},
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
    {"cuStreamSetAttribute", (void *)cuStreamSetAttribute},
    {"cuEventCreate", (void *)cuEventCreate},
    {"cuEventRecord", (void *)cuEventRecord},
    {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
    {"cuEventQuery", (void *)cuEventQuery},
    {"cuEventSynchronize", (void *)cuEventSynchronize},
    {"cuEventDestroy_v2", (void *)cuEventDestroy_v2},
    {"cuEventElapsedTime", (void *)cuEventElapsedTime},
    {"cuImportExternalMemory", (void *)cuImportExternalMemory},
    {"cuExternalMemoryGetMappedBuffer", (void *)cuExternalMemoryGetMappedBuffer},
    {"cuExternalMemoryGetMappedMipmappedArray", (void *)cuExternalMemoryGetMappedMipmappedArray},
    {"cuDestroyExternalMemory", (void *)cuDestroyExternalMemory},
    {"cuImportExternalSemaphore", (void *)cuImportExternalSemaphore},
    {"cuSignalExternalSemaphoresAsync", (void *)cuSignalExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync", (void *)cuWaitExternalSemaphoresAsync},
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
    {"cuFuncSetBlockShape", (void *)cuFuncSetBlockShape},
    {"cuFuncSetSharedSize", (void *)cuFuncSetSharedSize},
    {"cuParamSetSize", (void *)cuParamSetSize},
    {"cuParamSeti", (void *)cuParamSeti},
    {"cuParamSetf", (void *)cuParamSetf},
    {"cuLaunch", (void *)cuLaunch},
    {"cuLaunchGrid", (void *)cuLaunchGrid},
    {"cuLaunchGridAsync", (void *)cuLaunchGridAsync},
    {"cuParamSetTexRef", (void *)cuParamSetTexRef},
    {"cuGraphCreate", (void *)cuGraphCreate},
    {"cuGraphAddKernelNode_v2", (void *)cuGraphAddKernelNode_v2},
    {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
    {"cuGraphKernelNodeSetParams_v2", (void *)cuGraphKernelNodeSetParams_v2},
    {"cuGraphAddMemcpyNode", (void *)cuGraphAddMemcpyNode},
    {"cuGraphMemcpyNodeGetParams", (void *)cuGraphMemcpyNodeGetParams},
    {"cuGraphMemcpyNodeSetParams", (void *)cuGraphMemcpyNodeSetParams},
    {"cuGraphAddMemsetNode", (void *)cuGraphAddMemsetNode},
    {"cuGraphMemsetNodeGetParams", (void *)cuGraphMemsetNodeGetParams},
    {"cuGraphMemsetNodeSetParams", (void *)cuGraphMemsetNodeSetParams},
    {"cuGraphAddHostNode", (void *)cuGraphAddHostNode},
    {"cuGraphHostNodeGetParams", (void *)cuGraphHostNodeGetParams},
    {"cuGraphHostNodeSetParams", (void *)cuGraphHostNodeSetParams},
    {"cuGraphAddChildGraphNode", (void *)cuGraphAddChildGraphNode},
    {"cuGraphChildGraphNodeGetGraph", (void *)cuGraphChildGraphNodeGetGraph},
    {"cuGraphAddEmptyNode", (void *)cuGraphAddEmptyNode},
    {"cuGraphAddEventRecordNode", (void *)cuGraphAddEventRecordNode},
    {"cuGraphEventRecordNodeGetEvent", (void *)cuGraphEventRecordNodeGetEvent},
    {"cuGraphEventRecordNodeSetEvent", (void *)cuGraphEventRecordNodeSetEvent},
    {"cuGraphAddEventWaitNode", (void *)cuGraphAddEventWaitNode},
    {"cuGraphEventWaitNodeGetEvent", (void *)cuGraphEventWaitNodeGetEvent},
    {"cuGraphEventWaitNodeSetEvent", (void *)cuGraphEventWaitNodeSetEvent},
    {"cuGraphAddExternalSemaphoresSignalNode", (void *)cuGraphAddExternalSemaphoresSignalNode},
    {"cuGraphExternalSemaphoresSignalNodeGetParams", (void *)cuGraphExternalSemaphoresSignalNodeGetParams},
    {"cuGraphExternalSemaphoresSignalNodeSetParams", (void *)cuGraphExternalSemaphoresSignalNodeSetParams},
    {"cuGraphAddExternalSemaphoresWaitNode", (void *)cuGraphAddExternalSemaphoresWaitNode},
    {"cuGraphExternalSemaphoresWaitNodeGetParams", (void *)cuGraphExternalSemaphoresWaitNodeGetParams},
    {"cuGraphExternalSemaphoresWaitNodeSetParams", (void *)cuGraphExternalSemaphoresWaitNodeSetParams},
    {"cuGraphAddBatchMemOpNode", (void *)cuGraphAddBatchMemOpNode},
    {"cuGraphBatchMemOpNodeGetParams", (void *)cuGraphBatchMemOpNodeGetParams},
    {"cuGraphBatchMemOpNodeSetParams", (void *)cuGraphBatchMemOpNodeSetParams},
    {"cuGraphExecBatchMemOpNodeSetParams", (void *)cuGraphExecBatchMemOpNodeSetParams},
    {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
    {"cuGraphMemAllocNodeGetParams", (void *)cuGraphMemAllocNodeGetParams},
    {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
    {"cuGraphMemFreeNodeGetParams", (void *)cuGraphMemFreeNodeGetParams},
    {"cuDeviceGraphMemTrim", (void *)cuDeviceGraphMemTrim},
    {"cuGraphClone", (void *)cuGraphClone},
    {"cuGraphNodeFindInClone", (void *)cuGraphNodeFindInClone},
    {"cuGraphNodeGetType", (void *)cuGraphNodeGetType},
    {"cuGraphGetNodes", (void *)cuGraphGetNodes},
    {"cuGraphGetRootNodes", (void *)cuGraphGetRootNodes},
    {"cuGraphGetEdges", (void *)cuGraphGetEdges},
    {"cuGraphNodeGetDependencies", (void *)cuGraphNodeGetDependencies},
    {"cuGraphNodeGetDependentNodes", (void *)cuGraphNodeGetDependentNodes},
    {"cuGraphAddDependencies", (void *)cuGraphAddDependencies},
    {"cuGraphRemoveDependencies", (void *)cuGraphRemoveDependencies},
    {"cuGraphDestroyNode", (void *)cuGraphDestroyNode},
    {"cuGraphInstantiateWithFlags", (void *)cuGraphInstantiateWithFlags},
    {"cuGraphInstantiateWithParams", (void *)cuGraphInstantiateWithParams},
    {"cuGraphExecGetFlags", (void *)cuGraphExecGetFlags},
    {"cuGraphExecKernelNodeSetParams_v2", (void *)cuGraphExecKernelNodeSetParams_v2},
    {"cuGraphExecMemcpyNodeSetParams", (void *)cuGraphExecMemcpyNodeSetParams},
    {"cuGraphExecMemsetNodeSetParams", (void *)cuGraphExecMemsetNodeSetParams},
    {"cuGraphExecHostNodeSetParams", (void *)cuGraphExecHostNodeSetParams},
    {"cuGraphExecChildGraphNodeSetParams", (void *)cuGraphExecChildGraphNodeSetParams},
    {"cuGraphExecEventRecordNodeSetEvent", (void *)cuGraphExecEventRecordNodeSetEvent},
    {"cuGraphExecEventWaitNodeSetEvent", (void *)cuGraphExecEventWaitNodeSetEvent},
    {"cuGraphExecExternalSemaphoresSignalNodeSetParams", (void *)cuGraphExecExternalSemaphoresSignalNodeSetParams},
    {"cuGraphExecExternalSemaphoresWaitNodeSetParams", (void *)cuGraphExecExternalSemaphoresWaitNodeSetParams},
    {"cuGraphNodeSetEnabled", (void *)cuGraphNodeSetEnabled},
    {"cuGraphNodeGetEnabled", (void *)cuGraphNodeGetEnabled},
    {"cuGraphUpload", (void *)cuGraphUpload},
    {"cuGraphLaunch", (void *)cuGraphLaunch},
    {"cuGraphExecDestroy", (void *)cuGraphExecDestroy},
    {"cuGraphDestroy", (void *)cuGraphDestroy},
    {"cuGraphExecUpdate_v2", (void *)cuGraphExecUpdate_v2},
    {"cuGraphKernelNodeCopyAttributes", (void *)cuGraphKernelNodeCopyAttributes},
    {"cuGraphKernelNodeGetAttribute", (void *)cuGraphKernelNodeGetAttribute},
    {"cuGraphKernelNodeSetAttribute", (void *)cuGraphKernelNodeSetAttribute},
    {"cuGraphDebugDotPrint", (void *)cuGraphDebugDotPrint},
    {"cuUserObjectRetain", (void *)cuUserObjectRetain},
    {"cuUserObjectRelease", (void *)cuUserObjectRelease},
    {"cuGraphRetainUserObject", (void *)cuGraphRetainUserObject},
    {"cuGraphReleaseUserObject", (void *)cuGraphReleaseUserObject},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessor", (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", (void *)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {"cuOccupancyAvailableDynamicSMemPerBlock", (void *)cuOccupancyAvailableDynamicSMemPerBlock},
    {"cuOccupancyMaxPotentialClusterSize", (void *)cuOccupancyMaxPotentialClusterSize},
    {"cuOccupancyMaxActiveClusters", (void *)cuOccupancyMaxActiveClusters},
    {"cuTexRefSetArray", (void *)cuTexRefSetArray},
    {"cuTexRefSetMipmappedArray", (void *)cuTexRefSetMipmappedArray},
    {"cuTexRefSetAddress_v2", (void *)cuTexRefSetAddress_v2},
    {"cuTexRefSetAddress2D_v3", (void *)cuTexRefSetAddress2D_v3},
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
    {"cuTexObjectCreate", (void *)cuTexObjectCreate},
    {"cuTexObjectDestroy", (void *)cuTexObjectDestroy},
    {"cuTexObjectGetResourceDesc", (void *)cuTexObjectGetResourceDesc},
    {"cuTexObjectGetTextureDesc", (void *)cuTexObjectGetTextureDesc},
    {"cuTexObjectGetResourceViewDesc", (void *)cuTexObjectGetResourceViewDesc},
    {"cuSurfObjectCreate", (void *)cuSurfObjectCreate},
    {"cuSurfObjectDestroy", (void *)cuSurfObjectDestroy},
    {"cuSurfObjectGetResourceDesc", (void *)cuSurfObjectGetResourceDesc},
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
    {"cudaDeviceGetTexture1DLinearMaxWidth", (void *)cudaDeviceGetTexture1DLinearMaxWidth},
    {"cudaDeviceGetCacheConfig", (void *)cudaDeviceGetCacheConfig},
    {"cudaDeviceGetStreamPriorityRange", (void *)cudaDeviceGetStreamPriorityRange},
    {"cudaDeviceSetCacheConfig", (void *)cudaDeviceSetCacheConfig},
    {"cudaDeviceGetSharedMemConfig", (void *)cudaDeviceGetSharedMemConfig},
    {"cudaDeviceSetSharedMemConfig", (void *)cudaDeviceSetSharedMemConfig},
    {"cudaDeviceGetByPCIBusId", (void *)cudaDeviceGetByPCIBusId},
    {"cudaDeviceGetPCIBusId", (void *)cudaDeviceGetPCIBusId},
    {"cudaIpcGetEventHandle", (void *)cudaIpcGetEventHandle},
    {"cudaIpcOpenEventHandle", (void *)cudaIpcOpenEventHandle},
    {"cudaIpcOpenMemHandle", (void *)cudaIpcOpenMemHandle},
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
    {"cudaDeviceGetP2PAttribute", (void *)cudaDeviceGetP2PAttribute},
    {"cudaChooseDevice", (void *)cudaChooseDevice},
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
    {"cudaStreamSetAttribute", (void *)cudaStreamSetAttribute},
    {"cudaStreamDestroy", (void *)cudaStreamDestroy},
    {"cudaStreamWaitEvent", (void *)cudaStreamWaitEvent},
    {"cudaStreamSynchronize", (void *)cudaStreamSynchronize},
    {"cudaStreamQuery", (void *)cudaStreamQuery},
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
    {"cudaExternalMemoryGetMappedBuffer", (void *)cudaExternalMemoryGetMappedBuffer},
    {"cudaExternalMemoryGetMappedMipmappedArray", (void *)cudaExternalMemoryGetMappedMipmappedArray},
    {"cudaDestroyExternalMemory", (void *)cudaDestroyExternalMemory},
    {"cudaImportExternalSemaphore", (void *)cudaImportExternalSemaphore},
    {"cudaSignalExternalSemaphoresAsync_v2", (void *)cudaSignalExternalSemaphoresAsync_v2},
    {"cudaWaitExternalSemaphoresAsync_v2", (void *)cudaWaitExternalSemaphoresAsync_v2},
    {"cudaDestroyExternalSemaphore", (void *)cudaDestroyExternalSemaphore},
    {"cudaLaunchKernelExC", (void *)cudaLaunchKernelExC},
    {"cudaLaunchCooperativeKernel", (void *)cudaLaunchCooperativeKernel},
    {"cudaLaunchCooperativeKernelMultiDevice", (void *)cudaLaunchCooperativeKernelMultiDevice},
    {"cudaFuncSetCacheConfig", (void *)cudaFuncSetCacheConfig},
    {"cudaFuncSetSharedMemConfig", (void *)cudaFuncSetSharedMemConfig},
    {"cudaFuncGetAttributes", (void *)cudaFuncGetAttributes},
    {"cudaFuncSetAttribute", (void *)cudaFuncSetAttribute},
    {"cudaSetDoubleForDevice", (void *)cudaSetDoubleForDevice},
    {"cudaSetDoubleForHost", (void *)cudaSetDoubleForHost},
    {"cudaOccupancyMaxActiveBlocksPerMultiprocessor", (void *)cudaOccupancyMaxActiveBlocksPerMultiprocessor},
    {"cudaOccupancyAvailableDynamicSMemPerBlock", (void *)cudaOccupancyAvailableDynamicSMemPerBlock},
    {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", (void *)cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {"cudaOccupancyMaxPotentialClusterSize", (void *)cudaOccupancyMaxPotentialClusterSize},
    {"cudaOccupancyMaxActiveClusters", (void *)cudaOccupancyMaxActiveClusters},
    {"cudaMallocManaged", (void *)cudaMallocManaged},
    {"cudaMalloc", (void *)cudaMalloc},
    {"cudaMallocHost", (void *)cudaMallocHost},
    {"cudaMallocPitch", (void *)cudaMallocPitch},
    {"cudaMallocArray", (void *)cudaMallocArray},
    {"cudaFree", (void *)cudaFree},
    {"cudaFreeHost", (void *)cudaFreeHost},
    {"cudaFreeArray", (void *)cudaFreeArray},
    {"cudaFreeMipmappedArray", (void *)cudaFreeMipmappedArray},
    {"cudaHostAlloc", (void *)cudaHostAlloc},
    {"cudaMalloc3D", (void *)cudaMalloc3D},
    {"cudaMalloc3DArray", (void *)cudaMalloc3DArray},
    {"cudaMallocMipmappedArray", (void *)cudaMallocMipmappedArray},
    {"cudaGetMipmappedArrayLevel", (void *)cudaGetMipmappedArrayLevel},
    {"cudaMemcpy3D", (void *)cudaMemcpy3D},
    {"cudaMemcpy3DPeer", (void *)cudaMemcpy3DPeer},
    {"cudaMemcpy3DAsync", (void *)cudaMemcpy3DAsync},
    {"cudaMemcpy3DPeerAsync", (void *)cudaMemcpy3DPeerAsync},
    {"cudaMemGetInfo", (void *)cudaMemGetInfo},
    {"cudaArrayGetInfo", (void *)cudaArrayGetInfo},
    {"cudaArrayGetPlane", (void *)cudaArrayGetPlane},
    {"cudaArrayGetMemoryRequirements", (void *)cudaArrayGetMemoryRequirements},
    {"cudaMipmappedArrayGetMemoryRequirements", (void *)cudaMipmappedArrayGetMemoryRequirements},
    {"cudaArrayGetSparseProperties", (void *)cudaArrayGetSparseProperties},
    {"cudaMipmappedArrayGetSparseProperties", (void *)cudaMipmappedArrayGetSparseProperties},
    {"cudaMemcpy2DToArray", (void *)cudaMemcpy2DToArray},
    {"cudaMemcpy2DArrayToArray", (void *)cudaMemcpy2DArrayToArray},
    {"cudaMemcpyToSymbol", (void *)cudaMemcpyToSymbol},
    {"cudaMemcpy2DToArrayAsync", (void *)cudaMemcpy2DToArrayAsync},
    {"cudaMemcpyToSymbolAsync", (void *)cudaMemcpyToSymbolAsync},
    {"cudaMemset3D", (void *)cudaMemset3D},
    {"cudaMemset3DAsync", (void *)cudaMemset3DAsync},
    {"cudaGetSymbolAddress", (void *)cudaGetSymbolAddress},
    {"cudaGetSymbolSize", (void *)cudaGetSymbolSize},
    {"cudaMemPrefetchAsync", (void *)cudaMemPrefetchAsync},
    {"cudaMemAdvise", (void *)cudaMemAdvise},
    {"cudaMemRangeGetAttributes", (void *)cudaMemRangeGetAttributes},
    {"cudaMemcpyToArray", (void *)cudaMemcpyToArray},
    {"cudaMemcpyArrayToArray", (void *)cudaMemcpyArrayToArray},
    {"cudaMemcpyToArrayAsync", (void *)cudaMemcpyToArrayAsync},
    {"cudaMallocAsync", (void *)cudaMallocAsync},
    {"cudaMemPoolTrimTo", (void *)cudaMemPoolTrimTo},
    {"cudaMemPoolSetAccess", (void *)cudaMemPoolSetAccess},
    {"cudaMemPoolGetAccess", (void *)cudaMemPoolGetAccess},
    {"cudaMemPoolCreate", (void *)cudaMemPoolCreate},
    {"cudaMemPoolDestroy", (void *)cudaMemPoolDestroy},
    {"cudaMallocFromPoolAsync", (void *)cudaMallocFromPoolAsync},
    {"cudaMemPoolImportPointer", (void *)cudaMemPoolImportPointer},
    {"cudaPointerGetAttributes", (void *)cudaPointerGetAttributes},
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
    {"cudaCreateTextureObject", (void *)cudaCreateTextureObject},
    {"cudaDestroyTextureObject", (void *)cudaDestroyTextureObject},
    {"cudaGetTextureObjectResourceDesc", (void *)cudaGetTextureObjectResourceDesc},
    {"cudaGetTextureObjectTextureDesc", (void *)cudaGetTextureObjectTextureDesc},
    {"cudaGetTextureObjectResourceViewDesc", (void *)cudaGetTextureObjectResourceViewDesc},
    {"cudaCreateSurfaceObject", (void *)cudaCreateSurfaceObject},
    {"cudaDestroySurfaceObject", (void *)cudaDestroySurfaceObject},
    {"cudaGetSurfaceObjectResourceDesc", (void *)cudaGetSurfaceObjectResourceDesc},
    {"cudaDriverGetVersion", (void *)cudaDriverGetVersion},
    {"cudaRuntimeGetVersion", (void *)cudaRuntimeGetVersion},
    {"cudaGraphCreate", (void *)cudaGraphCreate},
    {"cudaGraphAddKernelNode", (void *)cudaGraphAddKernelNode},
    {"cudaGraphKernelNodeGetParams", (void *)cudaGraphKernelNodeGetParams},
    {"cudaGraphKernelNodeSetParams", (void *)cudaGraphKernelNodeSetParams},
    {"cudaGraphKernelNodeCopyAttributes", (void *)cudaGraphKernelNodeCopyAttributes},
    {"cudaGraphKernelNodeGetAttribute", (void *)cudaGraphKernelNodeGetAttribute},
    {"cudaGraphKernelNodeSetAttribute", (void *)cudaGraphKernelNodeSetAttribute},
    {"cudaGraphAddMemcpyNode", (void *)cudaGraphAddMemcpyNode},
    {"cudaGraphAddMemcpyNodeToSymbol", (void *)cudaGraphAddMemcpyNodeToSymbol},
    {"cudaGraphMemcpyNodeGetParams", (void *)cudaGraphMemcpyNodeGetParams},
    {"cudaGraphMemcpyNodeSetParams", (void *)cudaGraphMemcpyNodeSetParams},
    {"cudaGraphMemcpyNodeSetParamsToSymbol", (void *)cudaGraphMemcpyNodeSetParamsToSymbol},
    {"cudaGraphAddMemsetNode", (void *)cudaGraphAddMemsetNode},
    {"cudaGraphMemsetNodeGetParams", (void *)cudaGraphMemsetNodeGetParams},
    {"cudaGraphMemsetNodeSetParams", (void *)cudaGraphMemsetNodeSetParams},
    {"cudaGraphAddHostNode", (void *)cudaGraphAddHostNode},
    {"cudaGraphHostNodeGetParams", (void *)cudaGraphHostNodeGetParams},
    {"cudaGraphHostNodeSetParams", (void *)cudaGraphHostNodeSetParams},
    {"cudaGraphAddChildGraphNode", (void *)cudaGraphAddChildGraphNode},
    {"cudaGraphChildGraphNodeGetGraph", (void *)cudaGraphChildGraphNodeGetGraph},
    {"cudaGraphAddEmptyNode", (void *)cudaGraphAddEmptyNode},
    {"cudaGraphAddEventRecordNode", (void *)cudaGraphAddEventRecordNode},
    {"cudaGraphEventRecordNodeGetEvent", (void *)cudaGraphEventRecordNodeGetEvent},
    {"cudaGraphEventRecordNodeSetEvent", (void *)cudaGraphEventRecordNodeSetEvent},
    {"cudaGraphAddEventWaitNode", (void *)cudaGraphAddEventWaitNode},
    {"cudaGraphEventWaitNodeGetEvent", (void *)cudaGraphEventWaitNodeGetEvent},
    {"cudaGraphEventWaitNodeSetEvent", (void *)cudaGraphEventWaitNodeSetEvent},
    {"cudaGraphAddExternalSemaphoresSignalNode", (void *)cudaGraphAddExternalSemaphoresSignalNode},
    {"cudaGraphExternalSemaphoresSignalNodeGetParams", (void *)cudaGraphExternalSemaphoresSignalNodeGetParams},
    {"cudaGraphExternalSemaphoresSignalNodeSetParams", (void *)cudaGraphExternalSemaphoresSignalNodeSetParams},
    {"cudaGraphAddExternalSemaphoresWaitNode", (void *)cudaGraphAddExternalSemaphoresWaitNode},
    {"cudaGraphExternalSemaphoresWaitNodeGetParams", (void *)cudaGraphExternalSemaphoresWaitNodeGetParams},
    {"cudaGraphExternalSemaphoresWaitNodeSetParams", (void *)cudaGraphExternalSemaphoresWaitNodeSetParams},
    {"cudaGraphAddMemAllocNode", (void *)cudaGraphAddMemAllocNode},
    {"cudaGraphMemAllocNodeGetParams", (void *)cudaGraphMemAllocNodeGetParams},
    {"cudaDeviceGraphMemTrim", (void *)cudaDeviceGraphMemTrim},
    {"cudaGraphClone", (void *)cudaGraphClone},
    {"cudaGraphNodeFindInClone", (void *)cudaGraphNodeFindInClone},
    {"cudaGraphNodeGetType", (void *)cudaGraphNodeGetType},
    {"cudaGraphGetNodes", (void *)cudaGraphGetNodes},
    {"cudaGraphGetRootNodes", (void *)cudaGraphGetRootNodes},
    {"cudaGraphGetEdges", (void *)cudaGraphGetEdges},
    {"cudaGraphNodeGetDependencies", (void *)cudaGraphNodeGetDependencies},
    {"cudaGraphNodeGetDependentNodes", (void *)cudaGraphNodeGetDependentNodes},
    {"cudaGraphAddDependencies", (void *)cudaGraphAddDependencies},
    {"cudaGraphRemoveDependencies", (void *)cudaGraphRemoveDependencies},
    {"cudaGraphDestroyNode", (void *)cudaGraphDestroyNode},
    {"cudaGraphInstantiate", (void *)cudaGraphInstantiate},
    {"cudaGraphInstantiateWithFlags", (void *)cudaGraphInstantiateWithFlags},
    {"cudaGraphInstantiateWithParams", (void *)cudaGraphInstantiateWithParams},
    {"cudaGraphExecGetFlags", (void *)cudaGraphExecGetFlags},
    {"cudaGraphExecKernelNodeSetParams", (void *)cudaGraphExecKernelNodeSetParams},
    {"cudaGraphExecMemcpyNodeSetParams", (void *)cudaGraphExecMemcpyNodeSetParams},
    {"cudaGraphExecMemcpyNodeSetParamsToSymbol", (void *)cudaGraphExecMemcpyNodeSetParamsToSymbol},
    {"cudaGraphExecMemsetNodeSetParams", (void *)cudaGraphExecMemsetNodeSetParams},
    {"cudaGraphExecHostNodeSetParams", (void *)cudaGraphExecHostNodeSetParams},
    {"cudaGraphExecChildGraphNodeSetParams", (void *)cudaGraphExecChildGraphNodeSetParams},
    {"cudaGraphExecEventRecordNodeSetEvent", (void *)cudaGraphExecEventRecordNodeSetEvent},
    {"cudaGraphExecEventWaitNodeSetEvent", (void *)cudaGraphExecEventWaitNodeSetEvent},
    {"cudaGraphExecExternalSemaphoresSignalNodeSetParams", (void *)cudaGraphExecExternalSemaphoresSignalNodeSetParams},
    {"cudaGraphExecExternalSemaphoresWaitNodeSetParams", (void *)cudaGraphExecExternalSemaphoresWaitNodeSetParams},
    {"cudaGraphNodeSetEnabled", (void *)cudaGraphNodeSetEnabled},
    {"cudaGraphNodeGetEnabled", (void *)cudaGraphNodeGetEnabled},
    {"cudaGraphExecUpdate", (void *)cudaGraphExecUpdate},
    {"cudaGraphUpload", (void *)cudaGraphUpload},
    {"cudaGraphLaunch", (void *)cudaGraphLaunch},
    {"cudaGraphExecDestroy", (void *)cudaGraphExecDestroy},
    {"cudaGraphDestroy", (void *)cudaGraphDestroy},
    {"cudaGraphDebugDotPrint", (void *)cudaGraphDebugDotPrint},
    {"cudaUserObjectRetain", (void *)cudaUserObjectRetain},
    {"cudaUserObjectRelease", (void *)cudaUserObjectRelease},
    {"cudaGraphRetainUserObject", (void *)cudaGraphRetainUserObject},
    {"cudaGraphReleaseUserObject", (void *)cudaGraphReleaseUserObject},
    {"cudaGetDriverEntryPoint", (void *)cudaGetDriverEntryPoint},
    {"cudaGetExportTable", (void *)cudaGetExportTable},
    {"cudaGetFuncBySymbol", (void *)cudaGetFuncBySymbol},
    {"cublasCreate_v2", (void *)cublasCreate_v2},
    {"cublasDestroy_v2", (void *)cublasDestroy_v2},
    {"cublasSgemm_v2", (void *)cublasSgemm_v2},
    {"cudnnCreate", (void *)cudnnCreate},
    {"cudnnDestroy", (void *)cudnnDestroy},
    {"cudnnCreateTensorDescriptor", (void *)cudnnCreateTensorDescriptor},
    {"cudnnSetTensor4dDescriptor", (void *)cudnnSetTensor4dDescriptor},
    {"cudnnCreateActivationDescriptor", (void *)cudnnCreateActivationDescriptor},
    {"cudnnSetActivationDescriptor", (void *)cudnnSetActivationDescriptor},
    {"cudnnActivationForward", (void *)cudnnActivationForward},
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
    {"cuStreamAttachMemAsync_ptsz", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery_ptsz", (void *)cuStreamQuery},
    {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize},
    {"cuEventRecord_ptsz", (void *)cuEventRecord},
    {"cuEventRecordWithFlags_ptsz", (void *)cuEventRecordWithFlags},
    {"cuLaunchKernel_ptsz", (void *)cuLaunchKernel},
    {"cuGraphicsMapResources_ptsz", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources_ptsz", (void *)cuGraphicsUnmapResources},
    {"cuLaunchCooperativeKernel_ptsz", (void *)cuLaunchCooperativeKernel},
    {"cuSignalExternalSemaphoresAsync_ptsz", (void *)cuSignalExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync_ptsz", (void *)cuWaitExternalSemaphoresAsync},
    {"cuGraphInstantiateWithParams_ptsz", (void *)cuGraphInstantiateWithParams},
    {"cuGraphUpload_ptsz", (void *)cuGraphUpload},
    {"cuGraphLaunch_ptsz", (void *)cuGraphLaunch},
    {"cuStreamCopyAttributes_ptsz", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute_ptsz", (void *)cuStreamGetAttribute},
    {"cuStreamSetAttribute_ptsz", (void *)cuStreamSetAttribute},
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
