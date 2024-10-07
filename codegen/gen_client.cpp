#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const std::size_t size);
extern int rpc_read(void *data, const std::size_t size);
extern int rpc_wait_for_response(const unsigned int request_id);
extern int rpc_end_request(void *return_value, const unsigned int request_id);
nvmlReturn_t nvmlInit_v2() {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlInit_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlInitWithFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlShutdown() {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlShutdown);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetNVMLVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&cudaDriverVersion, sizeof(cudaDriverVersion)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&cudaDriverVersion, sizeof(cudaDriverVersion)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetProcessName);
    if (request_id < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&unitCount, sizeof(unitCount)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&unit, sizeof(unit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetUnitInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetLedState);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&state, sizeof(state)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetPsuInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&psu, sizeof(psu)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetTemperature);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&type, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&temp, sizeof(temp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetFanSpeedInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&fanSpeeds, sizeof(fanSpeeds)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlUnitGetDevices);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&deviceCount, sizeof(deviceCount)) < 0 ||
        rpc_read(devices, *deviceCount * sizeof(devices)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    if (request_id < 0 ||
        rpc_write(&hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&hwbcCount, sizeof(hwbcCount)) < 0 ||
        rpc_read(hwbcEntries, *hwbcCount * sizeof(hwbcEntries)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCount_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&deviceCount, sizeof(deviceCount)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAttributes_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&attributes, sizeof(attributes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleBySerial);
    std::size_t serial_len = std::strlen(serial) + 1;
    if (request_id < 0 ||
        rpc_write(&serial_len, sizeof(std::size_t)) < 0 ||
        rpc_write(serial, serial_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByUUID);
    std::size_t uuid_len = std::strlen(uuid) + 1;
    if (request_id < 0 ||
        rpc_write(&uuid_len, sizeof(std::size_t)) < 0 ||
        rpc_write(uuid, uuid_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByPciBusId_v2);
    std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
    if (request_id < 0 ||
        rpc_write(&pciBusId_len, sizeof(std::size_t)) < 0 ||
        rpc_write(pciBusId, pciBusId_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetName);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBrand);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&type, sizeof(type)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&index, sizeof(index)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSerial);
    std::size_t serial_len = std::strlen(serial) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&serial_len, sizeof(serial_len)) < 0 ||
        rpc_read(serial, serial_len) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&nodeSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeSet, nodeSetSize * sizeof(nodeSet)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinityWithinScope);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cpuSet, cpuSetSize * sizeof(cpuSet)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cpuSet, cpuSetSize * sizeof(cpuSet)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pathInfo, sizeof(pathInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyNearestGpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(deviceArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetTopologyGpuSet);
    if (request_id < 0 ||
        rpc_write(&cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(deviceArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetP2PStatus);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p2pStatus, sizeof(p2pStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetUUID);
    std::size_t uuid_len = std::strlen(uuid) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&uuid_len, sizeof(uuid_len)) < 0 ||
        rpc_read(uuid, uuid_len) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMdevUUID);
    std::size_t mdevUuid_len = std::strlen(mdevUuid) + 1;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mdevUuid_len, sizeof(mdevUuid_len)) < 0 ||
        rpc_read(mdevUuid, mdevUuid_len) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinorNumber);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&minorNumber, sizeof(minorNumber)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardPartNumber);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(partNumber, length * sizeof(partNumber)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&object, sizeof(nvmlInforomObject_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomImageVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomConfigurationChecksum);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&checksum, sizeof(checksum)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceValidateInforom);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&display, sizeof(display)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayActive);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isActive, sizeof(isActive)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPersistenceMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPciInfo_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pci, sizeof(pci)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&maxLinkGen, sizeof(maxLinkGen)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&maxLinkGenDevice, sizeof(maxLinkGenDevice)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&maxLinkWidth, sizeof(maxLinkWidth)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&currLinkGen, sizeof(currLinkGen)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&currLinkWidth, sizeof(currLinkWidth)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieThroughput);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieReplayCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clock, sizeof(clock)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clock, sizeof(clock)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetApplicationsClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clockMHz, sizeof(clockMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultApplicationsClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clockMHz, sizeof(clockMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetApplicationsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clockMHz, sizeof(clockMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxCustomerBoostClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clockMHz, sizeof(clockMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedMemoryClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(clocksMHz, *count * sizeof(clocksMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedGraphicsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memoryClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(clocksMHz, *count * sizeof(clocksMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_read(&defaultIsEnabled, sizeof(defaultIsEnabled)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) {
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

nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) {
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

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&speed, sizeof(speed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&speed, sizeof(speed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTargetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&targetSpeed, sizeof(targetSpeed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) {
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

nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&minSpeed, sizeof(minSpeed)) < 0 ||
        rpc_read(&maxSpeed, sizeof(maxSpeed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanControlPolicy_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&policy, sizeof(policy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) {
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

nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumFans);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numFans, sizeof(numFans)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperature);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&temp, sizeof(temp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&temp, sizeof(temp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(&temp, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&temp, sizeof(temp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetThermalSettings);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pThermalSettings, sizeof(pThermalSettings)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPerformanceState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pState, sizeof(pState)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long* clocksThrottleReasons) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clocksThrottleReasons, sizeof(clocksThrottleReasons)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long* supportedClocksThrottleReasons) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&supportedClocksThrottleReasons, sizeof(supportedClocksThrottleReasons)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pState, sizeof(pState)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&limit, sizeof(limit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&minLimit, sizeof(minLimit)) < 0 ||
        rpc_read(&maxLimit, sizeof(maxLimit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&defaultLimit, sizeof(defaultLimit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerUsage);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&power, sizeof(power)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEnergyConsumption);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&energy, sizeof(energy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEnforcedPowerLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&limit, sizeof(limit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuOperationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&current, sizeof(current)) < 0 ||
        rpc_read(&pending, sizeof(pending)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memory, sizeof(memory)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memory, sizeof(memory)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCudaComputeCapability);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&major, sizeof(major)) < 0 ||
        rpc_read(&minor, sizeof(minor)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&current, sizeof(current)) < 0 ||
        rpc_read(&pending, sizeof(pending)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&defaultMode, sizeof(defaultMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&boardId, sizeof(boardId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMultiGpuBoard);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&multiGpuBool, sizeof(multiGpuBool)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEccErrors);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&eccCounts, sizeof(eccCounts)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDetailedEccErrors);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&eccCounts, sizeof(eccCounts)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryErrorCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(&locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetUtilizationRates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&utilization, sizeof(utilization)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&utilization, sizeof(utilization)) < 0 ||
        rpc_read(&samplingPeriodUs, sizeof(samplingPeriodUs)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&encoderCapacity, sizeof(encoderCapacity)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sessionCount, sizeof(sessionCount)) < 0 ||
        rpc_read(&averageFps, sizeof(averageFps)) < 0 ||
        rpc_read(&averageLatency, sizeof(averageLatency)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sessionCount, sizeof(sessionCount)) < 0 ||
        rpc_read(sessionInfos, *sessionCount * sizeof(sessionInfos)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDecoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&utilization, sizeof(utilization)) < 0 ||
        rpc_read(&samplingPeriodUs, sizeof(samplingPeriodUs)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&fbcStats, sizeof(fbcStats)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCSessions);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sessionCount, sizeof(sessionCount)) < 0 ||
        rpc_read(sessionInfo, *sessionCount * sizeof(sessionInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDriverModel);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&current, sizeof(current)) < 0 ||
        rpc_read(&pending, sizeof(pending)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVbiosVersion);
    std::size_t version_len = std::strlen(version) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&version_len, sizeof(version_len)) < 0 ||
        rpc_read(version, version_len) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBridgeChipInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bridgeHierarchy, sizeof(bridgeHierarchy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&infoCount, sizeof(infoCount)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(infos)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGraphicsRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&infoCount, sizeof(infoCount)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(infos)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&infoCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&infoCount, sizeof(infoCount)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(infos)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceOnSameBoard);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&onSameBoard, sizeof(onSameBoard)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAPIRestriction);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isRestricted, sizeof(isRestricted)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSamples);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlSamplingType_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(&sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sampleValType, sizeof(sampleValType)) < 0 ||
        rpc_read(&sampleCount, sizeof(sampleCount)) < 0 ||
        rpc_read(samples, *sampleCount * sizeof(samples)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBAR1MemoryInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bar1Memory, sizeof(bar1Memory)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t* violTime) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetViolationStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&violTime, sizeof(violTime)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetIrqNum);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&irqNum, sizeof(irqNum)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumGpuCores);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numCores, sizeof(numCores)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerSource);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&powerSource, sizeof(powerSource)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryBusWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&busWidth, sizeof(busWidth)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieLinkMaxSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&maxSpeed, sizeof(maxSpeed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pcieSpeed, sizeof(pcieSpeed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAdaptiveClockInfoStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&adaptiveClockStatus, sizeof(adaptiveClockStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&stats, sizeof(stats)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingPids);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(pids, *count * sizeof(pids)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingBufferSize);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bufferSize, sizeof(bufferSize)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(&pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pageCount, sizeof(pageCount)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(addresses)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(&pageCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pageCount, sizeof(pageCount)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(addresses)) < 0 ||
        rpc_read(timestamps, *pageCount * sizeof(timestamps)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isPending, sizeof(isPending)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRemappedRows);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&corrRows, sizeof(corrRows)) < 0 ||
        rpc_read(&uncRows, sizeof(uncRows)) < 0 ||
        rpc_read(&isPending, sizeof(isPending)) < 0 ||
        rpc_read(&failureOccurred, sizeof(failureOccurred)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetRowRemapperHistogram);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&values, sizeof(values)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetArchitecture);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&arch, sizeof(arch)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) {
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

nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
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

nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
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

nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
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

nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
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

nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) {
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

nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
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

nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetGpuLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) {
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

nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetMemoryLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz) {
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

nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClkMonStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&status, sizeof(status)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
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

nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
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

nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) {
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

nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
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

nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isActive, sizeof(isActive)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&version, sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkCapability);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&capResult, sizeof(capResult)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pci, sizeof(pci)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkErrorCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&counterValue, sizeof(counterValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
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

nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control, unsigned int reset) {
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

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&control, sizeof(control)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long* rxcounter, unsigned long long* txcounter) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&rxcounter, sizeof(rxcounter)) < 0 ||
        rpc_read(&txcounter, sizeof(txcounter)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze) {
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

nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter) {
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

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNvLinkDeviceType, sizeof(pNvLinkDeviceType)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlEventSetCreate);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&set, sizeof(set)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) {
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

nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedEventTypes);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&eventTypes, sizeof(eventTypes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlEventSetWait_v2);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_write(&timeoutms, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlEventSetFree);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) {
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

nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceQueryDrainState);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&currentState, sizeof(currentState)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) {
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

nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceDiscoverGpus);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFieldValues);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(values)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearFieldValues);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(values)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVirtualizationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pVirtualMode, sizeof(pVirtualMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHostVgpuMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHostVgpuMode, sizeof(pHostVgpuMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) {
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

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGridLicensableFeatures_v4);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGridLicensableFeatures, sizeof(pGridLicensableFeatures)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetProcessUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&processSamplesCount, sizeof(processSamplesCount)) < 0 ||
        rpc_read(utilization, *processSamplesCount * sizeof(utilization)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&version, sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_read(&defaultMode, sizeof(defaultMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetVgpuDriverCapabilities);
    if (request_id < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&capResult, sizeof(capResult)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuCapabilities);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&capResult, sizeof(capResult)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedVgpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuCount, sizeof(vgpuCount)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(vgpuTypeIds)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetCreatableVgpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuCount, sizeof(vgpuCount)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(vgpuTypeIds)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetClass);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&size, sizeof(size)) < 0 ||
        rpc_read(vgpuTypeClass, *size * sizeof(vgpuTypeClass)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetName);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&size, sizeof(size)) < 0 ||
        rpc_read(vgpuTypeName, *size * sizeof(vgpuTypeName)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpuInstanceProfileId, sizeof(gpuInstanceProfileId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetDeviceID);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&deviceID, sizeof(deviceID)) < 0 ||
        rpc_read(&subsystemID, sizeof(subsystemID)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFramebufferSize);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&fbSize, sizeof(fbSize)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetNumDisplayHeads);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numDisplayHeads, sizeof(numDisplayHeads)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetResolution);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&displayIndex, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&xdim, sizeof(xdim)) < 0 ||
        rpc_read(&ydim, sizeof(ydim)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetLicense);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeLicenseString, size * sizeof(vgpuTypeLicenseString)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFrameRateLimit);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&frameRateLimit, sizeof(frameRateLimit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstances);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuInstanceCount, sizeof(vgpuInstanceCount)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuInstanceCountPerVm, sizeof(vgpuInstanceCountPerVm)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetActiveVgpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuCount, sizeof(vgpuCount)) < 0 ||
        rpc_read(vgpuInstances, *vgpuCount * sizeof(vgpuInstances)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmID);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vmId, size * sizeof(vmId)) < 0 ||
        rpc_read(&vmIdType, sizeof(vmIdType)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetUUID);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, size * sizeof(uuid)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmDriverVersion);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFbUsage);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&fbUsage, sizeof(fbUsage)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseStatus);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&licensed, sizeof(licensed)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetType);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFrameRateLimit);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&frameRateLimit, sizeof(frameRateLimit)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&eccMode, sizeof(eccMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&encoderCapacity, sizeof(encoderCapacity)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) {
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

nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sessionCount, sizeof(sessionCount)) < 0 ||
        rpc_read(&averageFps, sizeof(averageFps)) < 0 ||
        rpc_read(&averageLatency, sizeof(averageLatency)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sessionCount, sizeof(sessionCount)) < 0 ||
        rpc_read(sessionInfo, *sessionCount * sizeof(sessionInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&fbcStats, sizeof(fbcStats)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCSessions);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sessionCount, sizeof(sessionCount)) < 0 ||
        rpc_read(sessionInfo, *sessionCount * sizeof(sessionInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuInstanceId);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpuInstanceId, sizeof(gpuInstanceId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuPciId);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&length, sizeof(length)) < 0 ||
        rpc_read(vgpuPciId, *length * sizeof(vgpuPciId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetCapabilities);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&capResult, sizeof(capResult)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMetadata);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bufferSize, sizeof(bufferSize)) < 0 ||
        rpc_read(vgpuMetadata, *bufferSize * sizeof(vgpuMetadata)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuMetadata);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bufferSize, sizeof(bufferSize)) < 0 ||
        rpc_read(pgpuMetadata, *bufferSize * sizeof(pgpuMetadata)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetVgpuCompatibility);
    if (request_id < 0 ||
        rpc_write(&vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuMetadata, sizeof(vgpuMetadata)) < 0 ||
        rpc_read(&pgpuMetadata, sizeof(pgpuMetadata)) < 0 ||
        rpc_read(&compatibilityInfo, sizeof(compatibilityInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetPgpuMetadataString);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bufferSize, sizeof(bufferSize)) < 0 ||
        rpc_read(pgpuMetadata, *bufferSize * sizeof(pgpuMetadata)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerLog);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pSchedulerLog, sizeof(pSchedulerLog)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pSchedulerState, sizeof(pSchedulerState)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerCapabilities);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCapabilities, sizeof(pCapabilities)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetVgpuVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&supported, sizeof(supported)) < 0 ||
        rpc_read(&current, sizeof(current)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSetVgpuVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuVersion, sizeof(vgpuVersion)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(&sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(&vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sampleValType, sizeof(sampleValType)) < 0 ||
        rpc_read(&vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount)) < 0 ||
        rpc_read(utilizationSamples, *vgpuInstanceSamplesCount * sizeof(utilizationSamples)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuProcessUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(&vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount)) < 0 ||
        rpc_read(utilizationSamples, *vgpuProcessSamplesCount * sizeof(utilizationSamples)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingPids);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(pids, *count * sizeof(pids)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&stats, sizeof(stats)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&licenseInfo, sizeof(licenseInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&deviceCount, sizeof(deviceCount)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceInfoByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetMigMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&activationStatus, sizeof(activationStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&currentMode, sizeof(currentMode)) < 0 ||
        rpc_read(&pendingMode, sizeof(pendingMode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfoV);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(placements, *count * sizeof(placements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstance);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&placement, sizeof(nvmlGpuInstancePlacement_t*)) < 0 ||
        (placement != nullptr && rpc_write(placement, sizeof(nvmlGpuInstancePlacement_t)) < 0) ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstances);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(gpuInstances, *count * sizeof(gpuInstances)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceById);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetInfo);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(placements, *count * sizeof(placements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstance);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&computeInstance, sizeof(computeInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t* placement, nvmlComputeInstance_t* computeInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstanceWithPlacement);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&placement, sizeof(nvmlComputeInstancePlacement_t*)) < 0 ||
        (placement != nullptr && rpc_write(placement, sizeof(nvmlComputeInstancePlacement_t)) < 0) ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&computeInstance, sizeof(computeInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlComputeInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstances);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_read(computeInstances, *count * sizeof(computeInstances)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceById);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&computeInstance, sizeof(computeInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlComputeInstanceGetInfo_v2);
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceIsMigDeviceHandle);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isMigDevice, sizeof(isMigDevice)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&id, sizeof(id)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeInstanceId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&id, sizeof(id)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxMigDeviceCount);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&migDevice, sizeof(migDevice)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    if (request_id < 0 ||
        rpc_write(&migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBusType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&type, sizeof(type)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDynamicPstatesInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pDynamicPstatesInfo, sizeof(pDynamicPstatesInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) {
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

nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&offset, sizeof(offset)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset) {
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

nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&offset, sizeof(offset)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset) {
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

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxClockOfPState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&pstate, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&minClockMHz, sizeof(minClockMHz)) < 0 ||
        rpc_read(&maxClockMHz, sizeof(maxClockMHz)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedPerformanceStates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pstates, size * sizeof(pstates)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkMinMaxVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&minOffset, sizeof(minOffset)) < 0 ||
        rpc_read(&maxOffset, sizeof(maxOffset)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkMinMaxVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&minOffset, sizeof(minOffset)) < 0 ||
        rpc_read(&maxOffset, sizeof(maxOffset)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* gpuFabricInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuFabricInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpuFabricInfo, sizeof(gpuFabricInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t* metricsGet) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmMetricsGet);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&metricsGet, sizeof(metricsGet)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmSampleFree);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpmSample, sizeof(gpmSample)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t* gpmSample) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmSampleAlloc);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpmSample, sizeof(gpmSample)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample) {
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

nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample) {
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

nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t* gpmSupport) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpmQueryDeviceSupport);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpmSupport, sizeof(gpmSupport)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

std::unordered_map<std::string, void *> functionMap = {
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
};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it == functionMap.end())
        return nullptr;
    return it->second;
}
