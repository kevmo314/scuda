#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const size_t size);
extern int rpc_read(void *data, const size_t size);
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
        rpc_write(&flags, sizeof(flags)) < 0 ||
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

const char* nvmlErrorString(nvmlReturn_t result) {
    const char* return_value;

    int request_id = rpc_start_request(RPC_nvmlErrorString);
    if (request_id < 0 ||
        rpc_write(&result, sizeof(result)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&pid, sizeof(pid)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&index, sizeof(index)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
        rpc_write(&deviceCount, sizeof(deviceCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&deviceCount, sizeof(deviceCount)) < 0 ||
        rpc_read(&devices, sizeof(devices)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    if (request_id < 0 ||
        rpc_write(&hwbcCount, sizeof(hwbcCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&index, sizeof(index)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleBySerial);
    size_t serial_len = strlen(serial) + 1;
    if (request_id < 0 ||
        rpc_write(&serial_len, sizeof(serial_len)) < 0 ||
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
    size_t uuid_len = strlen(uuid) + 1;
    if (request_id < 0 ||
        rpc_write(&uuid_len, sizeof(uuid_len)) < 0 ||
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
    size_t pciBusId_len = strlen(pciBusId) + 1;
    if (request_id < 0 ||
        rpc_write(&pciBusId_len, sizeof(pciBusId_len)) < 0 ||
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
    size_t name_len = strlen(name) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&name_len, sizeof(name_len)) < 0 ||
        rpc_read(name, name_len) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetBrand);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&index, sizeof(index)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSerial);
    size_t serial_len = strlen(serial) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&nodeSetSize, sizeof(nodeSetSize)) < 0 ||
        rpc_write(&scope, sizeof(scope)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&cpuSetSize, sizeof(cpuSetSize)) < 0 ||
        rpc_write(&scope, sizeof(scope)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&cpuSetSize, sizeof(cpuSetSize)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(device1)) < 0 ||
        rpc_write(&device2, sizeof(device2)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&level, sizeof(level)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&cpuNumber, sizeof(cpuNumber)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&device1, sizeof(device1)) < 0 ||
        rpc_write(&device2, sizeof(device2)) < 0 ||
        rpc_write(&p2pIndex, sizeof(p2pIndex)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p2pStatus, sizeof(p2pStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetUUID);
    size_t uuid_len = strlen(uuid) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
    size_t mdevUuid_len = strlen(mdevUuid) + 1;
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&clockType, sizeof(clockType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&clockType, sizeof(clockType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&clockType, sizeof(clockType)) < 0 ||
        rpc_write(&clockId, sizeof(clockId)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&clockType, sizeof(clockType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&memoryClockMHz, sizeof(memoryClockMHz)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&enabled, sizeof(enabled)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&enabled, sizeof(enabled)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&fan, sizeof(fan)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&fan, sizeof(fan)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&fan, sizeof(fan)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&fan, sizeof(fan)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&fan, sizeof(fan)) < 0 ||
        rpc_write(&policy, sizeof(policy)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumFans);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&sensorType, sizeof(sensorType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&thresholdType, sizeof(thresholdType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&thresholdType, sizeof(thresholdType)) < 0 ||
        rpc_write(&temp, sizeof(temp)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&sensorIndex, sizeof(sensorIndex)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&errorType, sizeof(errorType)) < 0 ||
        rpc_write(&counterType, sizeof(counterType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&errorType, sizeof(errorType)) < 0 ||
        rpc_write(&counterType, sizeof(counterType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&errorType, sizeof(errorType)) < 0 ||
        rpc_write(&counterType, sizeof(counterType)) < 0 ||
        rpc_write(&locationType, sizeof(locationType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&encoderQueryType, sizeof(encoderQueryType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&sessionCount, sizeof(sessionCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&sessionCount, sizeof(sessionCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
    size_t version_len = strlen(version) + 1;
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&infoCount, sizeof(infoCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&infoCount, sizeof(infoCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&infoCount, sizeof(infoCount)) < 0 ||
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
        rpc_write(&device1, sizeof(device1)) < 0 ||
        rpc_write(&device2, sizeof(device2)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&apiType, sizeof(apiType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0 ||
        rpc_write(&sampleCount, sizeof(sampleCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&perfPolicyType, sizeof(perfPolicyType)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&pid, sizeof(pid)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&cause, sizeof(cause)) < 0 ||
        rpc_write(&pageCount, sizeof(pageCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&cause, sizeof(cause)) < 0 ||
        rpc_write(&pageCount, sizeof(pageCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&unit, sizeof(unit)) < 0 ||
        rpc_write(&color, sizeof(color)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetPersistenceMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetComputeMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&ecc, sizeof(ecc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearEccErrorCounts);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&counterType, sizeof(counterType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetDriverModel);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&driverModel, sizeof(driverModel)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&minGpuClockMHz, sizeof(minGpuClockMHz)) < 0 ||
        rpc_write(&maxGpuClockMHz, sizeof(maxGpuClockMHz)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetGpuLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemoryLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&minMemClockMHz, sizeof(minMemClockMHz)) < 0 ||
        rpc_write(&maxMemClockMHz, sizeof(maxMemClockMHz)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetMemoryLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetApplicationsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&memClockMHz, sizeof(memClockMHz)) < 0 ||
        rpc_write(&graphicsClockMHz, sizeof(graphicsClockMHz)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetClkMonStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuOperationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetAPIRestriction);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&apiType, sizeof(apiType)) < 0 ||
        rpc_write(&isRestricted, sizeof(isRestricted)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&capability, sizeof(capability)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control, unsigned int reset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
        rpc_write(&control, sizeof(control)) < 0 ||
        rpc_write(&reset, sizeof(reset)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
        rpc_write(&freeze, sizeof(freeze)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
        rpc_write(&counter, sizeof(counter)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&link, sizeof(link)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&eventTypes, sizeof(eventTypes)) < 0 ||
        rpc_write(&set, sizeof(set)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedEventTypes);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&set, sizeof(set)) < 0 ||
        rpc_write(&timeoutms, sizeof(timeoutms)) < 0 ||
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
        rpc_write(&set, sizeof(set)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceModifyDrainState);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(pciInfo)) < 0 ||
        rpc_write(&newState, sizeof(newState)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceQueryDrainState);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(pciInfo)) < 0 ||
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
        rpc_write(&pciInfo, sizeof(pciInfo)) < 0 ||
        rpc_write(&gpuState, sizeof(gpuState)) < 0 ||
        rpc_write(&linkState, sizeof(linkState)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceDiscoverGpus);
    if (request_id < 0 ||
        rpc_write(&pciInfo, sizeof(pciInfo)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetFieldValues);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&valuesCount, sizeof(valuesCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&valuesCount, sizeof(valuesCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&virtualMode, sizeof(virtualMode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGridLicensableFeatures_v4);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&processSamplesCount, sizeof(processSamplesCount)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&capability, sizeof(capability)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&capability, sizeof(capability)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&vgpuCount, sizeof(vgpuCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&vgpuCount, sizeof(vgpuCount)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
        rpc_write(&displayIndex, sizeof(displayIndex)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&vgpuCount, sizeof(vgpuCount)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&encoderCapacity, sizeof(encoderCapacity)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&sessionCount, sizeof(sessionCount)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&sessionCount, sizeof(sessionCount)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
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
        rpc_write(&vgpuTypeId, sizeof(vgpuTypeId)) < 0 ||
        rpc_write(&capability, sizeof(capability)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&bufferSize, sizeof(bufferSize)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&bufferSize, sizeof(bufferSize)) < 0 ||
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
        rpc_write(&vgpuMetadata, sizeof(vgpuMetadata)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&bufferSize, sizeof(bufferSize)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0 ||
        rpc_write(&sampleValType, sizeof(sampleValType)) < 0 ||
        rpc_write(&vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0 ||
        rpc_write(&vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_write(&pid, sizeof(pid)) < 0 ||
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
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(vgpuInstance)) < 0 ||
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
        rpc_write(&index, sizeof(index)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profile, sizeof(profile)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profile, sizeof(profile)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&placement, sizeof(placement)) < 0 ||
        rpc_read(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstances);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&id, sizeof(id)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profile, sizeof(profile)) < 0 ||
        rpc_write(&engProfile, sizeof(engProfile)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profile, sizeof(profile)) < 0 ||
        rpc_write(&engProfile, sizeof(engProfile)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
        rpc_write(&placement, sizeof(placement)) < 0 ||
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
        rpc_write(&computeInstance, sizeof(computeInstance)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstances);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&profileId, sizeof(profileId)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
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
        rpc_write(&gpuInstance, sizeof(gpuInstance)) < 0 ||
        rpc_write(&id, sizeof(id)) < 0 ||
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
        rpc_write(&computeInstance, sizeof(computeInstance)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&index, sizeof(index)) < 0 ||
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
        rpc_write(&migDevice, sizeof(migDevice)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&fan, sizeof(fan)) < 0 ||
        rpc_write(&speed, sizeof(speed)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&offset, sizeof(offset)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxClockOfPState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_write(&pstate, sizeof(pstate)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&gpuInstanceId, sizeof(gpuInstanceId)) < 0 ||
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
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&gpmSupport, sizeof(gpmSupport)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCcuGetStreamState(nvmlDevice_t device, unsigned int* state) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceCcuGetStreamState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&state, sizeof(state)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceCcuSetStreamState(nvmlDevice_t device, unsigned int state) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceCcuSetStreamState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&state, sizeof(state)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) {
    nvmlReturn_t return_value;

    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&info, sizeof(info)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuInit(unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuInit);
    if (request_id < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDriverGetVersion(int* driverVersion) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDriverGetVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&driverVersion, sizeof(driverVersion)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGet);
    if (request_id < 0 ||
        rpc_write(&ordinal, sizeof(ordinal)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetCount(int* count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetName);
    if (request_id < 0 ||
        rpc_write(&len, sizeof(len)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, len * sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetUuid);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, 16) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetUuid_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, 16) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetLuid);
    size_t luid_len = strlen(luid) + 1;
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&luid_len, sizeof(luid_len)) < 0 ||
        rpc_read(luid, luid_len) < 0 ||
        rpc_read(&deviceNodeMask, sizeof(deviceNodeMask)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceTotalMem_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&bytes, sizeof(bytes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetTexture1DLinearMaxWidth);
    if (request_id < 0 ||
        rpc_write(&format, sizeof(format)) < 0 ||
        rpc_write(&numChannels, sizeof(numChannels)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&maxWidthInElements, sizeof(maxWidthInElements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetAttribute);
    if (request_id < 0 ||
        rpc_write(&attrib, sizeof(attrib)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pi, sizeof(pi)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceSetMemPool);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetMemPool);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pool, sizeof(pool)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetDefaultMemPool);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pool_out, sizeof(pool_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetExecAffinitySupport);
    if (request_id < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pi, sizeof(pi)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFlushGPUDirectRDMAWrites);
    if (request_id < 0 ||
        rpc_write(&target, sizeof(target)) < 0 ||
        rpc_write(&scope, sizeof(scope)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetProperties);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&prop, sizeof(prop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceComputeCapability);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&major, sizeof(major)) < 0 ||
        rpc_read(&minor, sizeof(minor)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxRetain);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxRelease_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxSetFlags_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxGetState);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_read(&active, sizeof(active)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxReset_v2);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxCreate_v2);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxCreate_v3);
    if (request_id < 0 ||
        rpc_write(&numParams, sizeof(numParams)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_read(paramsArray, numParams * sizeof(paramsArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxDestroy_v2);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxPushCurrent_v2);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxPopCurrent_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetCurrent);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetCurrent);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetDevice);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetFlags(unsigned int* flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetFlags);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long* ctxId) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetId);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ctxId, sizeof(ctxId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxSynchronize() {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSynchronize);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pvalue, sizeof(pvalue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetCacheConfig);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pconfig, sizeof(pconfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetSharedMemConfig);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pConfig, sizeof(pConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetApiVersion);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&version, sizeof(version)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetStreamPriorityRange);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&leastPriority, sizeof(leastPriority)) < 0 ||
        rpc_read(&greatestPriority, sizeof(greatestPriority)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxResetPersistingL2Cache() {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxResetPersistingL2Cache);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxGetExecAffinity);
    if (request_id < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pExecAffinity, sizeof(pExecAffinity)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxAttach);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxDetach(CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxDetach);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleLoad);
    size_t fname_len = strlen(fname) + 1;
    if (request_id < 0 ||
        rpc_write(&fname_len, sizeof(fname_len)) < 0 ||
        rpc_write(fname, fname_len) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&module, sizeof(module)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleLoadDataEx);
    size_t image_len = strlen(image) + 1;
    if (request_id < 0 ||
        rpc_write(&image_len, sizeof(image_len)) < 0 ||
        rpc_write(image, image_len) < 0 ||
        rpc_write(&numOptions, sizeof(numOptions)) < 0 ||
        rpc_write(&options, sizeof(options)) < 0 ||
        rpc_write(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&module, sizeof(module)) < 0 ||
        rpc_read(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleUnload(CUmodule hmod) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleUnload);
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(hmod)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetLoadingMode);
    if (request_id < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetFunction);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&hmod, sizeof(hmod)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetGlobal_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytes, sizeof(bytes)) < 0 ||
        rpc_write(&hmod, sizeof(hmod)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_read(&bytes, sizeof(bytes)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkCreate_v2);
    if (request_id < 0 ||
        rpc_write(&numOptions, sizeof(numOptions)) < 0 ||
        rpc_write(&options, sizeof(options)) < 0 ||
        rpc_write(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_write(&stateOut, sizeof(stateOut)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&options, sizeof(options)) < 0 ||
        rpc_read(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_read(&stateOut, sizeof(stateOut)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkAddData_v2);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(state)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_write(&numOptions, sizeof(numOptions)) < 0 ||
        rpc_write(&options, sizeof(options)) < 0 ||
        rpc_write(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_read(&options, sizeof(options)) < 0 ||
        rpc_read(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkAddFile_v2);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(state)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_write(&path, sizeof(path)) < 0 ||
        rpc_write(&numOptions, sizeof(numOptions)) < 0 ||
        rpc_write(&options, sizeof(options)) < 0 ||
        rpc_write(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&path, sizeof(path)) < 0 ||
        rpc_read(&options, sizeof(options)) < 0 ||
        rpc_read(&optionValues, sizeof(optionValues)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkComplete);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(state)) < 0 ||
        rpc_write(&cubinOut, sizeof(cubinOut)) < 0 ||
        rpc_write(&sizeOut, sizeof(sizeOut)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&cubinOut, sizeof(cubinOut)) < 0 ||
        rpc_read(&sizeOut, sizeof(sizeOut)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLinkDestroy(CUlinkState state) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLinkDestroy);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(state)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetTexRef);
    if (request_id < 0 ||
        rpc_write(&pTexRef, sizeof(pTexRef)) < 0 ||
        rpc_write(&hmod, sizeof(hmod)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pTexRef, sizeof(pTexRef)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuModuleGetSurfRef);
    if (request_id < 0 ||
        rpc_write(&pSurfRef, sizeof(pSurfRef)) < 0 ||
        rpc_write(&hmod, sizeof(hmod)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pSurfRef, sizeof(pSurfRef)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryLoadData);
    if (request_id < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_write(&code, sizeof(code)) < 0 ||
        rpc_write(&jitOptions, sizeof(jitOptions)) < 0 ||
        rpc_write(&jitOptionsValues, sizeof(jitOptionsValues)) < 0 ||
        rpc_write(&numJitOptions, sizeof(numJitOptions)) < 0 ||
        rpc_write(&libraryOptions, sizeof(libraryOptions)) < 0 ||
        rpc_write(&libraryOptionValues, sizeof(libraryOptionValues)) < 0 ||
        rpc_write(&numLibraryOptions, sizeof(numLibraryOptions)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&library, sizeof(library)) < 0 ||
        rpc_read(&code, sizeof(code)) < 0 ||
        rpc_read(&jitOptions, sizeof(jitOptions)) < 0 ||
        rpc_read(&jitOptionsValues, sizeof(jitOptionsValues)) < 0 ||
        rpc_read(&libraryOptions, sizeof(libraryOptions)) < 0 ||
        rpc_read(&libraryOptionValues, sizeof(libraryOptionValues)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryLoadFromFile);
    if (request_id < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_write(&fileName, sizeof(fileName)) < 0 ||
        rpc_write(&jitOptions, sizeof(jitOptions)) < 0 ||
        rpc_write(&jitOptionsValues, sizeof(jitOptionsValues)) < 0 ||
        rpc_write(&numJitOptions, sizeof(numJitOptions)) < 0 ||
        rpc_write(&libraryOptions, sizeof(libraryOptions)) < 0 ||
        rpc_write(&libraryOptionValues, sizeof(libraryOptionValues)) < 0 ||
        rpc_write(&numLibraryOptions, sizeof(numLibraryOptions)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&library, sizeof(library)) < 0 ||
        rpc_read(&fileName, sizeof(fileName)) < 0 ||
        rpc_read(&jitOptions, sizeof(jitOptions)) < 0 ||
        rpc_read(&jitOptionsValues, sizeof(jitOptionsValues)) < 0 ||
        rpc_read(&libraryOptions, sizeof(libraryOptions)) < 0 ||
        rpc_read(&libraryOptionValues, sizeof(libraryOptionValues)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryUnload(CUlibrary library) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryUnload);
    if (request_id < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetKernel);
    if (request_id < 0 ||
        rpc_write(&pKernel, sizeof(pKernel)) < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pKernel, sizeof(pKernel)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryGetModule(CUmodule* pMod, CUlibrary library) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetModule);
    if (request_id < 0 ||
        rpc_write(&pMod, sizeof(pMod)) < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pMod, sizeof(pMod)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelGetFunction);
    if (request_id < 0 ||
        rpc_write(&pFunc, sizeof(pFunc)) < 0 ||
        rpc_write(&kernel, sizeof(kernel)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pFunc, sizeof(pFunc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetGlobal);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytes, sizeof(bytes)) < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_read(&bytes, sizeof(bytes)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetManaged);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytes, sizeof(bytes)) < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_write(&name, sizeof(name)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_read(&bytes, sizeof(bytes)) < 0 ||
        rpc_read(&name, sizeof(name)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLibraryGetUnifiedFunction);
    if (request_id < 0 ||
        rpc_write(&fptr, sizeof(fptr)) < 0 ||
        rpc_write(&library, sizeof(library)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&fptr, sizeof(fptr)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelGetAttribute);
    if (request_id < 0 ||
        rpc_write(&pi, sizeof(pi)) < 0 ||
        rpc_write(&attrib, sizeof(attrib)) < 0 ||
        rpc_write(&kernel, sizeof(kernel)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pi, sizeof(pi)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelSetAttribute);
    if (request_id < 0 ||
        rpc_write(&attrib, sizeof(attrib)) < 0 ||
        rpc_write(&val, sizeof(val)) < 0 ||
        rpc_write(&kernel, sizeof(kernel)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuKernelSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&kernel, sizeof(kernel)) < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetInfo_v2);
    if (request_id < 0 ||
        rpc_write(&free, sizeof(free)) < 0 ||
        rpc_write(&total, sizeof(total)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&free, sizeof(free)) < 0 ||
        rpc_read(&total, sizeof(total)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAlloc_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocPitch_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&pPitch, sizeof(pPitch)) < 0 ||
        rpc_write(&WidthInBytes, sizeof(WidthInBytes)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_write(&ElementSizeBytes, sizeof(ElementSizeBytes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_read(&pPitch, sizeof(pPitch)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemFree_v2);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetAddressRange_v2);
    if (request_id < 0 ||
        rpc_write(&pbase, sizeof(pbase)) < 0 ||
        rpc_write(&psize, sizeof(psize)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pbase, sizeof(pbase)) < 0 ||
        rpc_read(&psize, sizeof(psize)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocHost_v2);
    if (request_id < 0 ||
        rpc_write(&pp, sizeof(pp)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pp, sizeof(pp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemFreeHost(void* p) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemFreeHost);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostAlloc);
    if (request_id < 0 ||
        rpc_write(&pp, sizeof(pp)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pp, sizeof(pp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostGetDevicePointer_v2);
    if (request_id < 0 ||
        rpc_write(&pdptr, sizeof(pdptr)) < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pdptr, sizeof(pdptr)) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostGetFlags);
    if (request_id < 0 ||
        rpc_write(&pFlags, sizeof(pFlags)) < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pFlags, sizeof(pFlags)) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocManaged);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetByPCIBusId);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_write(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dev, sizeof(dev)) < 0 ||
        rpc_read(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetPCIBusId);
    if (request_id < 0 ||
        rpc_write(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_write(&len, sizeof(len)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcGetEventHandle);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcOpenEventHandle);
    if (request_id < 0 ||
        rpc_write(&phEvent, sizeof(phEvent)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phEvent, sizeof(phEvent)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcGetMemHandle);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcOpenMemHandle_v2);
    if (request_id < 0 ||
        rpc_write(&pdptr, sizeof(pdptr)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pdptr, sizeof(pdptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuIpcCloseMemHandle);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostRegister_v2);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemHostUnregister(void* p) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemHostUnregister);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyPeer);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstContext, sizeof(dstContext)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&srcContext, sizeof(srcContext)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyHtoD_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoH_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoD_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoA_v2);
    if (request_id < 0 ||
        rpc_write(&dstArray, sizeof(dstArray)) < 0 ||
        rpc_write(&dstOffset, sizeof(dstOffset)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoD_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&srcArray, sizeof(srcArray)) < 0 ||
        rpc_write(&srcOffset, sizeof(srcOffset)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyHtoA_v2);
    if (request_id < 0 ||
        rpc_write(&dstArray, sizeof(dstArray)) < 0 ||
        rpc_write(&dstOffset, sizeof(dstOffset)) < 0 ||
        rpc_write(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoH_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_write(&srcArray, sizeof(srcArray)) < 0 ||
        rpc_write(&srcOffset, sizeof(srcOffset)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoA_v2);
    if (request_id < 0 ||
        rpc_write(&dstArray, sizeof(dstArray)) < 0 ||
        rpc_write(&dstOffset, sizeof(dstOffset)) < 0 ||
        rpc_write(&srcArray, sizeof(srcArray)) < 0 ||
        rpc_write(&srcOffset, sizeof(srcOffset)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy2D_v2);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy2DUnaligned_v2);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy3D_v2);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy3DPeer);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyPeerAsync);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstContext, sizeof(dstContext)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&srcContext, sizeof(srcContext)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyHtoDAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoHAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyDtoDAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyHtoAAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstArray, sizeof(dstArray)) < 0 ||
        rpc_write(&dstOffset, sizeof(dstOffset)) < 0 ||
        rpc_write(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&srcHost, sizeof(srcHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpyAtoHAsync_v2);
    if (request_id < 0 ||
        rpc_write(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_write(&srcArray, sizeof(srcArray)) < 0 ||
        rpc_write(&srcOffset, sizeof(srcOffset)) < 0 ||
        rpc_write(&ByteCount, sizeof(ByteCount)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dstHost, sizeof(dstHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy2DAsync_v2);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy3DAsync_v2);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemcpy3DPeerAsync);
    if (request_id < 0 ||
        rpc_write(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCopy, sizeof(pCopy)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD8_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&uc, sizeof(uc)) < 0 ||
        rpc_write(&N, sizeof(N)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD16_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&us, sizeof(us)) < 0 ||
        rpc_write(&N, sizeof(N)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD32_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&ui, sizeof(ui)) < 0 ||
        rpc_write(&N, sizeof(N)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D8_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstPitch, sizeof(dstPitch)) < 0 ||
        rpc_write(&uc, sizeof(uc)) < 0 ||
        rpc_write(&Width, sizeof(Width)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D16_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstPitch, sizeof(dstPitch)) < 0 ||
        rpc_write(&us, sizeof(us)) < 0 ||
        rpc_write(&Width, sizeof(Width)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D32_v2);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstPitch, sizeof(dstPitch)) < 0 ||
        rpc_write(&ui, sizeof(ui)) < 0 ||
        rpc_write(&Width, sizeof(Width)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD8Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&uc, sizeof(uc)) < 0 ||
        rpc_write(&N, sizeof(N)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD16Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&us, sizeof(us)) < 0 ||
        rpc_write(&N, sizeof(N)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD32Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&ui, sizeof(ui)) < 0 ||
        rpc_write(&N, sizeof(N)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D8Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstPitch, sizeof(dstPitch)) < 0 ||
        rpc_write(&uc, sizeof(uc)) < 0 ||
        rpc_write(&Width, sizeof(Width)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D16Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstPitch, sizeof(dstPitch)) < 0 ||
        rpc_write(&us, sizeof(us)) < 0 ||
        rpc_write(&Width, sizeof(Width)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemsetD2D32Async);
    if (request_id < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&dstPitch, sizeof(dstPitch)) < 0 ||
        rpc_write(&ui, sizeof(ui)) < 0 ||
        rpc_write(&Width, sizeof(Width)) < 0 ||
        rpc_write(&Height, sizeof(Height)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayCreate_v2);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_write(&pAllocateArray, sizeof(pAllocateArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_read(&pAllocateArray, sizeof(pAllocateArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetDescriptor_v2);
    if (request_id < 0 ||
        rpc_write(&pArrayDescriptor, sizeof(pArrayDescriptor)) < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pArrayDescriptor, sizeof(pArrayDescriptor)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_write(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_write(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayGetPlane);
    if (request_id < 0 ||
        rpc_write(&pPlaneArray, sizeof(pPlaneArray)) < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_write(&planeIdx, sizeof(planeIdx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pPlaneArray, sizeof(pPlaneArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArrayDestroy(CUarray hArray) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArrayDestroy);
    if (request_id < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArray3DCreate_v2);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_write(&pAllocateArray, sizeof(pAllocateArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_read(&pAllocateArray, sizeof(pAllocateArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuArray3DGetDescriptor_v2);
    if (request_id < 0 ||
        rpc_write(&pArrayDescriptor, sizeof(pArrayDescriptor)) < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pArrayDescriptor, sizeof(pArrayDescriptor)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayCreate);
    if (request_id < 0 ||
        rpc_write(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_write(&pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc)) < 0 ||
        rpc_write(&numMipmapLevels, sizeof(numMipmapLevels)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHandle, sizeof(pHandle)) < 0 ||
        rpc_read(&pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayGetLevel);
    if (request_id < 0 ||
        rpc_write(&pLevelArray, sizeof(pLevelArray)) < 0 ||
        rpc_write(&hMipmappedArray, sizeof(hMipmappedArray)) < 0 ||
        rpc_write(&level, sizeof(level)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pLevelArray, sizeof(pLevelArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMipmappedArrayDestroy);
    if (request_id < 0 ||
        rpc_write(&hMipmappedArray, sizeof(hMipmappedArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetHandleForAddressRange);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&handleType, sizeof(handleType)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAddressReserve);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&alignment, sizeof(alignment)) < 0 ||
        rpc_write(&addr, sizeof(addr)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAddressFree);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemCreate);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&prop, sizeof(prop)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_read(&prop, sizeof(prop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRelease);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemMap);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemMapArrayAsync);
    if (request_id < 0 ||
        rpc_write(&mapInfoList, sizeof(mapInfoList)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mapInfoList, sizeof(mapInfoList)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemUnmap);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemSetAccess);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetAccess);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&location, sizeof(location)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_read(&location, sizeof(location)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemExportToShareableHandle);
    if (request_id < 0 ||
        rpc_write(&shareableHandle, sizeof(shareableHandle)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&handleType, sizeof(handleType)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&shareableHandle, sizeof(shareableHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemImportFromShareableHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&osHandle, sizeof(osHandle)) < 0 ||
        rpc_write(&shHandleType, sizeof(shHandleType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_read(&osHandle, sizeof(osHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetAllocationGranularity);
    if (request_id < 0 ||
        rpc_write(&granularity, sizeof(granularity)) < 0 ||
        rpc_write(&prop, sizeof(prop)) < 0 ||
        rpc_write(&option, sizeof(option)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&granularity, sizeof(granularity)) < 0 ||
        rpc_read(&prop, sizeof(prop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemGetAllocationPropertiesFromHandle);
    if (request_id < 0 ||
        rpc_write(&prop, sizeof(prop)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&prop, sizeof(prop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRetainAllocationHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&addr, sizeof(addr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_read(&addr, sizeof(addr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemFreeAsync);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocAsync);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolTrimTo);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&minBytesToKeep, sizeof(minBytesToKeep)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolSetAttribute);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolGetAttribute);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolSetAccess);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&map, sizeof(map)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&map, sizeof(map)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolGetAccess);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&location, sizeof(location)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_read(&location, sizeof(location)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolCreate);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&poolProps, sizeof(poolProps)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pool, sizeof(pool)) < 0 ||
        rpc_read(&poolProps, sizeof(poolProps)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolDestroy);
    if (request_id < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAllocFromPoolAsync);
    if (request_id < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytesize, sizeof(bytesize)) < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolExportToShareableHandle);
    if (request_id < 0 ||
        rpc_write(&handle_out, sizeof(handle_out)) < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&handleType, sizeof(handleType)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle_out, sizeof(handle_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolImportFromShareableHandle);
    if (request_id < 0 ||
        rpc_write(&pool_out, sizeof(pool_out)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&handleType, sizeof(handleType)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pool_out, sizeof(pool_out)) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolExportPointer);
    if (request_id < 0 ||
        rpc_write(&shareData_out, sizeof(shareData_out)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&shareData_out, sizeof(shareData_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPoolImportPointer);
    if (request_id < 0 ||
        rpc_write(&ptr_out, sizeof(ptr_out)) < 0 ||
        rpc_write(&pool, sizeof(pool)) < 0 ||
        rpc_write(&shareData, sizeof(shareData)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr_out, sizeof(ptr_out)) < 0 ||
        rpc_read(&shareData, sizeof(shareData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuPointerGetAttribute);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&attribute, sizeof(attribute)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemPrefetchAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemAdvise);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&advice, sizeof(advice)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRangeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&dataSize, sizeof(dataSize)) < 0 ||
        rpc_write(&attribute, sizeof(attribute)) < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuMemRangeGetAttributes);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&dataSizes, sizeof(dataSizes)) < 0 ||
        rpc_write(&attributes, sizeof(attributes)) < 0 ||
        rpc_write(&numAttributes, sizeof(numAttributes)) < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_read(&dataSizes, sizeof(dataSizes)) < 0 ||
        rpc_read(&attributes, sizeof(attributes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuPointerSetAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&attribute, sizeof(attribute)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuPointerGetAttributes);
    if (request_id < 0 ||
        rpc_write(&numAttributes, sizeof(numAttributes)) < 0 ||
        rpc_write(&attributes, sizeof(attributes)) < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&attributes, sizeof(attributes)) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamCreate);
    if (request_id < 0 ||
        rpc_write(&phStream, sizeof(phStream)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phStream, sizeof(phStream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamCreateWithPriority);
    if (request_id < 0 ||
        rpc_write(&phStream, sizeof(phStream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&priority, sizeof(priority)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phStream, sizeof(phStream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamGetPriority(CUstream hStream, int* priority) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetPriority);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&priority, sizeof(priority)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&priority, sizeof(priority)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetFlags);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long* streamId) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetId);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&streamId, sizeof(streamId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&streamId, sizeof(streamId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetCtx);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&pctx, sizeof(pctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pctx, sizeof(pctx)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWaitEvent);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&hEvent, sizeof(hEvent)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamAddCallback);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&callback, sizeof(callback)) < 0 ||
        rpc_write(&userData, sizeof(userData)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&userData, sizeof(userData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamBeginCapture_v2);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuThreadExchangeStreamCaptureMode);
    if (request_id < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamEndCapture);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&phGraph, sizeof(phGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraph, sizeof(phGraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamIsCapturing);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&captureStatus, sizeof(captureStatus)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&captureStatus, sizeof(captureStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetCaptureInfo_v2);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&captureStatus_out, sizeof(captureStatus_out)) < 0 ||
        rpc_write(&id_out, sizeof(id_out)) < 0 ||
        rpc_write(&graph_out, sizeof(graph_out)) < 0 ||
        rpc_write(&dependencies_out, sizeof(dependencies_out)) < 0 ||
        rpc_write(&numDependencies_out, sizeof(numDependencies_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&captureStatus_out, sizeof(captureStatus_out)) < 0 ||
        rpc_read(&id_out, sizeof(id_out)) < 0 ||
        rpc_read(&graph_out, sizeof(graph_out)) < 0 ||
        rpc_read(&dependencies_out, sizeof(dependencies_out)) < 0 ||
        rpc_read(&numDependencies_out, sizeof(numDependencies_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamUpdateCaptureDependencies);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamAttachMemAsync);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamQuery(CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamQuery);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamSynchronize(CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamSynchronize);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamDestroy_v2);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value_out, sizeof(value_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value_out, sizeof(value_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventCreate);
    if (request_id < 0 ||
        rpc_write(&phEvent, sizeof(phEvent)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phEvent, sizeof(phEvent)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventRecord);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(hEvent)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventRecordWithFlags);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(hEvent)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventQuery(CUevent hEvent) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventQuery);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(hEvent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventSynchronize(CUevent hEvent) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventSynchronize);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(hEvent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventDestroy_v2);
    if (request_id < 0 ||
        rpc_write(&hEvent, sizeof(hEvent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuEventElapsedTime);
    if (request_id < 0 ||
        rpc_write(&pMilliseconds, sizeof(pMilliseconds)) < 0 ||
        rpc_write(&hStart, sizeof(hStart)) < 0 ||
        rpc_write(&hEnd, sizeof(hEnd)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pMilliseconds, sizeof(pMilliseconds)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuImportExternalMemory);
    if (request_id < 0 ||
        rpc_write(&extMem_out, sizeof(extMem_out)) < 0 ||
        rpc_write(&memHandleDesc, sizeof(memHandleDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extMem_out, sizeof(extMem_out)) < 0 ||
        rpc_read(&memHandleDesc, sizeof(memHandleDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuExternalMemoryGetMappedBuffer);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&extMem, sizeof(extMem)) < 0 ||
        rpc_write(&bufferDesc, sizeof(bufferDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_read(&bufferDesc, sizeof(bufferDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuExternalMemoryGetMappedMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_write(&extMem, sizeof(extMem)) < 0 ||
        rpc_write(&mipmapDesc, sizeof(mipmapDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_read(&mipmapDesc, sizeof(mipmapDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDestroyExternalMemory);
    if (request_id < 0 ||
        rpc_write(&extMem, sizeof(extMem)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuImportExternalSemaphore);
    if (request_id < 0 ||
        rpc_write(&extSem_out, sizeof(extSem_out)) < 0 ||
        rpc_write(&semHandleDesc, sizeof(semHandleDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extSem_out, sizeof(extSem_out)) < 0 ||
        rpc_read(&semHandleDesc, sizeof(semHandleDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSignalExternalSemaphoresAsync);
    if (request_id < 0 ||
        rpc_write(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_write(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_write(&numExtSems, sizeof(numExtSems)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_read(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuWaitExternalSemaphoresAsync);
    if (request_id < 0 ||
        rpc_write(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_write(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_write(&numExtSems, sizeof(numExtSems)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_read(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDestroyExternalSemaphore);
    if (request_id < 0 ||
        rpc_write(&extSem, sizeof(extSem)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWaitValue32_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&addr, sizeof(addr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWaitValue64_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&addr, sizeof(addr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWriteValue32_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&addr, sizeof(addr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamWriteValue64_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&addr, sizeof(addr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuStreamBatchMemOp_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&paramArray, sizeof(paramArray)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&paramArray, sizeof(paramArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncGetAttribute);
    if (request_id < 0 ||
        rpc_write(&pi, sizeof(pi)) < 0 ||
        rpc_write(&attrib, sizeof(attrib)) < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pi, sizeof(pi)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&attrib, sizeof(attrib)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncGetModule);
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(hmod)) < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&hmod, sizeof(hmod)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchKernel);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_write(&gridDimX, sizeof(gridDimX)) < 0 ||
        rpc_write(&gridDimY, sizeof(gridDimY)) < 0 ||
        rpc_write(&gridDimZ, sizeof(gridDimZ)) < 0 ||
        rpc_write(&blockDimX, sizeof(blockDimX)) < 0 ||
        rpc_write(&blockDimY, sizeof(blockDimY)) < 0 ||
        rpc_write(&blockDimZ, sizeof(blockDimZ)) < 0 ||
        rpc_write(&sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&kernelParams, sizeof(kernelParams)) < 0 ||
        rpc_write(&extra, sizeof(extra)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchKernelEx);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_write(&kernelParams, sizeof(kernelParams)) < 0 ||
        rpc_write(&extra, sizeof(extra)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchCooperativeKernel);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_write(&gridDimX, sizeof(gridDimX)) < 0 ||
        rpc_write(&gridDimY, sizeof(gridDimY)) < 0 ||
        rpc_write(&gridDimZ, sizeof(gridDimZ)) < 0 ||
        rpc_write(&blockDimX, sizeof(blockDimX)) < 0 ||
        rpc_write(&blockDimY, sizeof(blockDimY)) < 0 ||
        rpc_write(&blockDimZ, sizeof(blockDimZ)) < 0 ||
        rpc_write(&sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&kernelParams, sizeof(kernelParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&kernelParams, sizeof(kernelParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchCooperativeKernelMultiDevice);
    if (request_id < 0 ||
        rpc_write(&launchParamsList, sizeof(launchParamsList)) < 0 ||
        rpc_write(&numDevices, sizeof(numDevices)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&launchParamsList, sizeof(launchParamsList)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchHostFunc);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&fn, sizeof(fn)) < 0 ||
        rpc_write(&userData, sizeof(userData)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&userData, sizeof(userData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetBlockShape);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&x, sizeof(x)) < 0 ||
        rpc_write(&y, sizeof(y)) < 0 ||
        rpc_write(&z, sizeof(z)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuFuncSetSharedSize);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&bytes, sizeof(bytes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetSize);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&numbytes, sizeof(numbytes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSeti);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetf);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetv);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&numbytes, sizeof(numbytes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunch(CUfunction f) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunch);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchGrid);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_write(&grid_width, sizeof(grid_width)) < 0 ||
        rpc_write(&grid_height, sizeof(grid_height)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuLaunchGridAsync);
    if (request_id < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_write(&grid_width, sizeof(grid_width)) < 0 ||
        rpc_write(&grid_height, sizeof(grid_height)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuParamSetTexRef);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(hfunc)) < 0 ||
        rpc_write(&texunit, sizeof(texunit)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphCreate);
    if (request_id < 0 ||
        rpc_write(&phGraph, sizeof(phGraph)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraph, sizeof(phGraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddKernelNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddKernelNode_v2);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeGetParams_v2);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeSetParams_v2);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddMemcpyNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&copyParams, sizeof(copyParams)) < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&copyParams, sizeof(copyParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemcpyNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemcpyNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddMemsetNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&memsetParams, sizeof(memsetParams)) < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&memsetParams, sizeof(memsetParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemsetNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemsetNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddHostNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphHostNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphHostNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddChildGraphNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&childGraph, sizeof(childGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphChildGraphNodeGetGraph);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&phGraph, sizeof(phGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraph, sizeof(phGraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddEmptyNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddEventRecordNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventRecordNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event_out, sizeof(event_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event_out, sizeof(event_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddEventWaitNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventWaitNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event_out, sizeof(event_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event_out, sizeof(event_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddExternalSemaphoresSignalNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExternalSemaphoresSignalNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&params_out, sizeof(params_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&params_out, sizeof(params_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExternalSemaphoresSignalNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddExternalSemaphoresWaitNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExternalSemaphoresWaitNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&params_out, sizeof(params_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&params_out, sizeof(params_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExternalSemaphoresWaitNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddBatchMemOpNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphBatchMemOpNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams_out, sizeof(nodeParams_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams_out, sizeof(nodeParams_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphBatchMemOpNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecBatchMemOpNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddMemAllocNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemAllocNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&params_out, sizeof(params_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&params_out, sizeof(params_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddMemFreeNode);
    if (request_id < 0 ||
        rpc_write(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphNode, sizeof(phGraphNode)) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphMemFreeNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&dptr_out, sizeof(dptr_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr_out, sizeof(dptr_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGraphMemTrim(CUdevice device) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGraphMemTrim);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceSetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphClone);
    if (request_id < 0 ||
        rpc_write(&phGraphClone, sizeof(phGraphClone)) < 0 ||
        rpc_write(&originalGraph, sizeof(originalGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphClone, sizeof(phGraphClone)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeFindInClone);
    if (request_id < 0 ||
        rpc_write(&phNode, sizeof(phNode)) < 0 ||
        rpc_write(&hOriginalNode, sizeof(hOriginalNode)) < 0 ||
        rpc_write(&hClonedGraph, sizeof(hClonedGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phNode, sizeof(phNode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetType);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&type, sizeof(type)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&type, sizeof(type)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphGetNodes);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&nodes, sizeof(nodes)) < 0 ||
        rpc_write(&numNodes, sizeof(numNodes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodes, sizeof(nodes)) < 0 ||
        rpc_read(&numNodes, sizeof(numNodes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphGetRootNodes);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&rootNodes, sizeof(rootNodes)) < 0 ||
        rpc_write(&numRootNodes, sizeof(numRootNodes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&rootNodes, sizeof(rootNodes)) < 0 ||
        rpc_read(&numRootNodes, sizeof(numRootNodes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphGetEdges);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&from, sizeof(from)) < 0 ||
        rpc_write(&to, sizeof(to)) < 0 ||
        rpc_write(&numEdges, sizeof(numEdges)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&from, sizeof(from)) < 0 ||
        rpc_read(&to, sizeof(to)) < 0 ||
        rpc_read(&numEdges, sizeof(numEdges)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetDependencies);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_read(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetDependentNodes);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&dependentNodes, sizeof(dependentNodes)) < 0 ||
        rpc_write(&numDependentNodes, sizeof(numDependentNodes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dependentNodes, sizeof(dependentNodes)) < 0 ||
        rpc_read(&numDependentNodes, sizeof(numDependentNodes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphAddDependencies);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&from, sizeof(from)) < 0 ||
        rpc_write(&to, sizeof(to)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&from, sizeof(from)) < 0 ||
        rpc_read(&to, sizeof(to)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphRemoveDependencies);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&from, sizeof(from)) < 0 ||
        rpc_write(&to, sizeof(to)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&from, sizeof(from)) < 0 ||
        rpc_read(&to, sizeof(to)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphDestroyNode(CUgraphNode hNode) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphDestroyNode);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphInstantiateWithFlags);
    if (request_id < 0 ||
        rpc_write(&phGraphExec, sizeof(phGraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphExec, sizeof(phGraphExec)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphInstantiateWithParams);
    if (request_id < 0 ||
        rpc_write(&phGraphExec, sizeof(phGraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&instantiateParams, sizeof(instantiateParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phGraphExec, sizeof(phGraphExec)) < 0 ||
        rpc_read(&instantiateParams, sizeof(instantiateParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecGetFlags);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecKernelNodeSetParams_v2);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecMemcpyNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&copyParams, sizeof(copyParams)) < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&copyParams, sizeof(copyParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecMemsetNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&memsetParams, sizeof(memsetParams)) < 0 ||
        rpc_write(&ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memsetParams, sizeof(memsetParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecHostNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecChildGraphNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&childGraph, sizeof(childGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeSetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphNodeGetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphUpload);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphLaunch);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecDestroy);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphDestroy);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphExecUpdate_v2);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&resultInfo, sizeof(resultInfo)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&resultInfo, sizeof(resultInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value_out, sizeof(value_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value_out, sizeof(value_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphKernelNodeSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphDebugDotPrint);
    if (request_id < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&path, sizeof(path)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&path, sizeof(path)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuUserObjectCreate);
    if (request_id < 0 ||
        rpc_write(&object_out, sizeof(object_out)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&destroy, sizeof(destroy)) < 0 ||
        rpc_write(&initialRefcount, sizeof(initialRefcount)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&object_out, sizeof(object_out)) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuUserObjectRetain);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuUserObjectRelease);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphRetainUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphReleaseUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor);
    if (request_id < 0 ||
        rpc_write(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&blockSize, sizeof(blockSize)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    if (request_id < 0 ||
        rpc_write(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&blockSize, sizeof(blockSize)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyAvailableDynamicSMemPerBlock);
    if (request_id < 0 ||
        rpc_write(&dynamicSmemSize, sizeof(dynamicSmemSize)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_write(&blockSize, sizeof(blockSize)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dynamicSmemSize, sizeof(dynamicSmemSize)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyMaxPotentialClusterSize);
    if (request_id < 0 ||
        rpc_write(&clusterSize, sizeof(clusterSize)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clusterSize, sizeof(clusterSize)) < 0 ||
        rpc_read(&config, sizeof(config)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuOccupancyMaxActiveClusters);
    if (request_id < 0 ||
        rpc_write(&numClusters, sizeof(numClusters)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numClusters, sizeof(numClusters)) < 0 ||
        rpc_read(&config, sizeof(config)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetArray);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&hMipmappedArray, sizeof(hMipmappedArray)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetAddress_v2);
    if (request_id < 0 ||
        rpc_write(&ByteOffset, sizeof(ByteOffset)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&bytes, sizeof(bytes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ByteOffset, sizeof(ByteOffset)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetAddress2D_v3);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_write(&Pitch, sizeof(Pitch)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetFormat);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&fmt, sizeof(fmt)) < 0 ||
        rpc_write(&NumPackedComponents, sizeof(NumPackedComponents)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetAddressMode);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&dim, sizeof(dim)) < 0 ||
        rpc_write(&am, sizeof(am)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetFilterMode);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&fm, sizeof(fm)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmapFilterMode);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&fm, sizeof(fm)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmapLevelBias);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&bias, sizeof(bias)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMipmapLevelClamp);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&minMipmapLevelClamp, sizeof(minMipmapLevelClamp)) < 0 ||
        rpc_write(&maxMipmapLevelClamp, sizeof(maxMipmapLevelClamp)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetMaxAnisotropy);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&maxAniso, sizeof(maxAniso)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetBorderColor);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&pBorderColor, sizeof(pBorderColor)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pBorderColor, sizeof(pBorderColor)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefSetFlags);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetAddress_v2);
    if (request_id < 0 ||
        rpc_write(&pdptr, sizeof(pdptr)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pdptr, sizeof(pdptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetArray);
    if (request_id < 0 ||
        rpc_write(&phArray, sizeof(phArray)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phArray, sizeof(phArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&phMipmappedArray, sizeof(phMipmappedArray)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phMipmappedArray, sizeof(phMipmappedArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetAddressMode);
    if (request_id < 0 ||
        rpc_write(&pam, sizeof(pam)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_write(&dim, sizeof(dim)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pam, sizeof(pam)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetFilterMode);
    if (request_id < 0 ||
        rpc_write(&pfm, sizeof(pfm)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pfm, sizeof(pfm)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetFormat);
    if (request_id < 0 ||
        rpc_write(&pFormat, sizeof(pFormat)) < 0 ||
        rpc_write(&pNumChannels, sizeof(pNumChannels)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pFormat, sizeof(pFormat)) < 0 ||
        rpc_read(&pNumChannels, sizeof(pNumChannels)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmapFilterMode);
    if (request_id < 0 ||
        rpc_write(&pfm, sizeof(pfm)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pfm, sizeof(pfm)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmapLevelBias);
    if (request_id < 0 ||
        rpc_write(&pbias, sizeof(pbias)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pbias, sizeof(pbias)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMipmapLevelClamp);
    if (request_id < 0 ||
        rpc_write(&pminMipmapLevelClamp, sizeof(pminMipmapLevelClamp)) < 0 ||
        rpc_write(&pmaxMipmapLevelClamp, sizeof(pmaxMipmapLevelClamp)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pminMipmapLevelClamp, sizeof(pminMipmapLevelClamp)) < 0 ||
        rpc_read(&pmaxMipmapLevelClamp, sizeof(pmaxMipmapLevelClamp)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetMaxAnisotropy);
    if (request_id < 0 ||
        rpc_write(&pmaxAniso, sizeof(pmaxAniso)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pmaxAniso, sizeof(pmaxAniso)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetBorderColor);
    if (request_id < 0 ||
        rpc_write(&pBorderColor, sizeof(pBorderColor)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pBorderColor, sizeof(pBorderColor)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefGetFlags);
    if (request_id < 0 ||
        rpc_write(&pFlags, sizeof(pFlags)) < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pFlags, sizeof(pFlags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefCreate(CUtexref* pTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefCreate);
    if (request_id < 0 ||
        rpc_write(&pTexRef, sizeof(pTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pTexRef, sizeof(pTexRef)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexRefDestroy(CUtexref hTexRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexRefDestroy);
    if (request_id < 0 ||
        rpc_write(&hTexRef, sizeof(hTexRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfRefSetArray);
    if (request_id < 0 ||
        rpc_write(&hSurfRef, sizeof(hSurfRef)) < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfRefGetArray);
    if (request_id < 0 ||
        rpc_write(&phArray, sizeof(phArray)) < 0 ||
        rpc_write(&hSurfRef, sizeof(hSurfRef)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&phArray, sizeof(phArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectCreate);
    if (request_id < 0 ||
        rpc_write(&pTexObject, sizeof(pTexObject)) < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_write(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_write(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pTexObject, sizeof(pTexObject)) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_read(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_read(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexObjectDestroy(CUtexObject texObject) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectDestroy);
    if (request_id < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectGetResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectGetTextureDesc);
    if (request_id < 0 ||
        rpc_write(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTexObjectGetResourceViewDesc);
    if (request_id < 0 ||
        rpc_write(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfObjectCreate);
    if (request_id < 0 ||
        rpc_write(&pSurfObject, sizeof(pSurfObject)) < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pSurfObject, sizeof(pSurfObject)) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfObjectDestroy);
    if (request_id < 0 ||
        rpc_write(&surfObject, sizeof(surfObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuSurfObjectGetResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_write(&surfObject, sizeof(surfObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTensorMapEncodeTiled);
    if (request_id < 0 ||
        rpc_write(&tensorMap, sizeof(tensorMap)) < 0 ||
        rpc_write(&tensorDataType, sizeof(tensorDataType)) < 0 ||
        rpc_write(&tensorRank, sizeof(tensorRank)) < 0 ||
        rpc_write(&globalAddress, sizeof(globalAddress)) < 0 ||
        rpc_write(&globalDim, sizeof(globalDim)) < 0 ||
        rpc_write(&globalStrides, sizeof(globalStrides)) < 0 ||
        rpc_write(&boxDim, sizeof(boxDim)) < 0 ||
        rpc_write(&elementStrides, sizeof(elementStrides)) < 0 ||
        rpc_write(&interleave, sizeof(interleave)) < 0 ||
        rpc_write(&swizzle, sizeof(swizzle)) < 0 ||
        rpc_write(&l2Promotion, sizeof(l2Promotion)) < 0 ||
        rpc_write(&oobFill, sizeof(oobFill)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&tensorMap, sizeof(tensorMap)) < 0 ||
        rpc_read(&globalAddress, sizeof(globalAddress)) < 0 ||
        rpc_read(&globalDim, sizeof(globalDim)) < 0 ||
        rpc_read(&globalStrides, sizeof(globalStrides)) < 0 ||
        rpc_read(&boxDim, sizeof(boxDim)) < 0 ||
        rpc_read(&elementStrides, sizeof(elementStrides)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTensorMapEncodeIm2col);
    if (request_id < 0 ||
        rpc_write(&tensorMap, sizeof(tensorMap)) < 0 ||
        rpc_write(&tensorDataType, sizeof(tensorDataType)) < 0 ||
        rpc_write(&tensorRank, sizeof(tensorRank)) < 0 ||
        rpc_write(&globalAddress, sizeof(globalAddress)) < 0 ||
        rpc_write(&globalDim, sizeof(globalDim)) < 0 ||
        rpc_write(&globalStrides, sizeof(globalStrides)) < 0 ||
        rpc_write(&pixelBoxLowerCorner, sizeof(pixelBoxLowerCorner)) < 0 ||
        rpc_write(&pixelBoxUpperCorner, sizeof(pixelBoxUpperCorner)) < 0 ||
        rpc_write(&channelsPerPixel, sizeof(channelsPerPixel)) < 0 ||
        rpc_write(&pixelsPerColumn, sizeof(pixelsPerColumn)) < 0 ||
        rpc_write(&elementStrides, sizeof(elementStrides)) < 0 ||
        rpc_write(&interleave, sizeof(interleave)) < 0 ||
        rpc_write(&swizzle, sizeof(swizzle)) < 0 ||
        rpc_write(&l2Promotion, sizeof(l2Promotion)) < 0 ||
        rpc_write(&oobFill, sizeof(oobFill)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&tensorMap, sizeof(tensorMap)) < 0 ||
        rpc_read(&globalAddress, sizeof(globalAddress)) < 0 ||
        rpc_read(&globalDim, sizeof(globalDim)) < 0 ||
        rpc_read(&globalStrides, sizeof(globalStrides)) < 0 ||
        rpc_read(&pixelBoxLowerCorner, sizeof(pixelBoxLowerCorner)) < 0 ||
        rpc_read(&pixelBoxUpperCorner, sizeof(pixelBoxUpperCorner)) < 0 ||
        rpc_read(&elementStrides, sizeof(elementStrides)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuTensorMapReplaceAddress);
    if (request_id < 0 ||
        rpc_write(&tensorMap, sizeof(tensorMap)) < 0 ||
        rpc_write(&globalAddress, sizeof(globalAddress)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&tensorMap, sizeof(tensorMap)) < 0 ||
        rpc_read(&globalAddress, sizeof(globalAddress)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceCanAccessPeer);
    if (request_id < 0 ||
        rpc_write(&canAccessPeer, sizeof(canAccessPeer)) < 0 ||
        rpc_write(&dev, sizeof(dev)) < 0 ||
        rpc_write(&peerDev, sizeof(peerDev)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&canAccessPeer, sizeof(canAccessPeer)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxEnablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerContext, sizeof(peerContext)) < 0 ||
        rpc_write(&Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuCtxDisablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerContext, sizeof(peerContext)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuDeviceGetP2PAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&attrib, sizeof(attrib)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsUnregisterResource);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsSubResourceGetMappedArray);
    if (request_id < 0 ||
        rpc_write(&pArray, sizeof(pArray)) < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_write(&arrayIndex, sizeof(arrayIndex)) < 0 ||
        rpc_write(&mipLevel, sizeof(mipLevel)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pArray, sizeof(pArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsResourceGetMappedMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&pMipmappedArray, sizeof(pMipmappedArray)) < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pMipmappedArray, sizeof(pMipmappedArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsResourceGetMappedPointer_v2);
    if (request_id < 0 ||
        rpc_write(&pDevPtr, sizeof(pDevPtr)) < 0 ||
        rpc_write(&pSize, sizeof(pSize)) < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pDevPtr, sizeof(pDevPtr)) < 0 ||
        rpc_read(&pSize, sizeof(pSize)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsResourceSetMapFlags_v2);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsMapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&resources, sizeof(resources)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&resources, sizeof(resources)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGraphicsUnmapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&resources, sizeof(resources)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&resources, sizeof(resources)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGetProcAddress_v2);
    if (request_id < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&pfn, sizeof(pfn)) < 0 ||
        rpc_write(&cudaVersion, sizeof(cudaVersion)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&symbolStatus, sizeof(symbolStatus)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&pfn, sizeof(pfn)) < 0 ||
        rpc_read(&symbolStatus, sizeof(symbolStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) {
    CUresult return_value;

    int request_id = rpc_start_request(RPC_cuGetExportTable);
    if (request_id < 0 ||
        rpc_write(&ppExportTable, sizeof(ppExportTable)) < 0 ||
        rpc_write(&pExportTableId, sizeof(pExportTableId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ppExportTable, sizeof(ppExportTable)) < 0 ||
        rpc_read(&pExportTableId, sizeof(pExportTableId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceReset() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceReset);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceSynchronize() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSynchronize);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetLimit);
    if (request_id < 0 ||
        rpc_write(&pValue, sizeof(pValue)) < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pValue, sizeof(pValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const struct cudaChannelFormatDesc* fmtDesc, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetTexture1DLinearMaxWidth);
    if (request_id < 0 ||
        rpc_write(&maxWidthInElements, sizeof(maxWidthInElements)) < 0 ||
        rpc_write(&fmtDesc, sizeof(fmtDesc)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&maxWidthInElements, sizeof(maxWidthInElements)) < 0 ||
        rpc_read(&fmtDesc, sizeof(fmtDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&pCacheConfig, sizeof(pCacheConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCacheConfig, sizeof(pCacheConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetStreamPriorityRange);
    if (request_id < 0 ||
        rpc_write(&leastPriority, sizeof(leastPriority)) < 0 ||
        rpc_write(&greatestPriority, sizeof(greatestPriority)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&leastPriority, sizeof(leastPriority)) < 0 ||
        rpc_read(&greatestPriority, sizeof(greatestPriority)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&cacheConfig, sizeof(cacheConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig* pConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&pConfig, sizeof(pConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pConfig, sizeof(pConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetByPCIBusId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_read(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetPCIBusId);
    if (request_id < 0 ||
        rpc_write(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_write(&len, sizeof(len)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pciBusId, sizeof(pciBusId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcGetEventHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcOpenEventHandle);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event, sizeof(event)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcGetMemHandle);
    if (request_id < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&handle, sizeof(handle)) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcOpenMemHandle);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&handle, sizeof(handle)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaIpcCloseMemHandle);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceFlushGPUDirectRDMAWrites);
    if (request_id < 0 ||
        rpc_write(&target, sizeof(target)) < 0 ||
        rpc_write(&scope, sizeof(scope)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadExit() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadExit);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadSynchronize() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadSynchronize);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadSetLimit);
    if (request_id < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadGetLimit(size_t* pValue, enum cudaLimit limit) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadGetLimit);
    if (request_id < 0 ||
        rpc_write(&pValue, sizeof(pValue)) < 0 ||
        rpc_write(&limit, sizeof(limit)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pValue, sizeof(pValue)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache* pCacheConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadGetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&pCacheConfig, sizeof(pCacheConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCacheConfig, sizeof(pCacheConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&cacheConfig, sizeof(cacheConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetLastError() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetLastError);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaPeekAtLastError() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaPeekAtLastError);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

const char* cudaGetErrorName(cudaError_t error) {
    const char* return_value;

    int request_id = rpc_start_request(RPC_cudaGetErrorName);
    if (request_id < 0 ||
        rpc_write(&error, sizeof(error)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

const char* cudaGetErrorString(cudaError_t error) {
    const char* return_value;

    int request_id = rpc_start_request(RPC_cudaGetErrorString);
    if (request_id < 0 ||
        rpc_write(&error, sizeof(error)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetDeviceCount(int* count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDeviceCount);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&count, sizeof(count)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp* prop, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDeviceProperties_v2);
    if (request_id < 0 ||
        rpc_write(&prop, sizeof(prop)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&prop, sizeof(prop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetDefaultMemPool);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memPool, sizeof(memPool)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetMemPool);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetMemPool);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memPool, sizeof(memPool)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetNvSciSyncAttributes);
    if (request_id < 0 ||
        rpc_write(&nvSciSyncAttrList, sizeof(nvSciSyncAttrList)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nvSciSyncAttrList, sizeof(nvSciSyncAttrList)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetP2PAttribute);
    if (request_id < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaChooseDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&prop, sizeof(prop)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_read(&prop, sizeof(prop)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaInitDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&deviceFlags, sizeof(deviceFlags)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaSetDevice(int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetDevice(int* device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDevice);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device, sizeof(device)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaSetValidDevices(int* device_arr, int len) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetValidDevices);
    if (request_id < 0 ||
        rpc_write(&device_arr, sizeof(device_arr)) < 0 ||
        rpc_write(&len, sizeof(len)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&device_arr, sizeof(device_arr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDeviceFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDeviceFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCreate);
    if (request_id < 0 ||
        rpc_write(&pStream, sizeof(pStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pStream, sizeof(pStream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCreateWithFlags);
    if (request_id < 0 ||
        rpc_write(&pStream, sizeof(pStream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pStream, sizeof(pStream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCreateWithPriority);
    if (request_id < 0 ||
        rpc_write(&pStream, sizeof(pStream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&priority, sizeof(priority)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pStream, sizeof(pStream)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetPriority);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&priority, sizeof(priority)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&priority, sizeof(priority)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetFlags);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetId);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&streamId, sizeof(streamId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&streamId, sizeof(streamId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaCtxResetPersistingL2Cache() {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaCtxResetPersistingL2Cache);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value_out, sizeof(value_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value_out, sizeof(value_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamDestroy);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamWaitEvent);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamAddCallback);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&callback, sizeof(callback)) < 0 ||
        rpc_write(&userData, sizeof(userData)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&userData, sizeof(userData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamSynchronize);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamQuery);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamAttachMemAsync);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&length, sizeof(length)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamBeginCapture);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaThreadExchangeStreamCaptureMode);
    if (request_id < 0 ||
        rpc_write(&mode, sizeof(mode)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mode, sizeof(mode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamEndCapture);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&pGraph, sizeof(pGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraph, sizeof(pGraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamIsCapturing);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&pCaptureStatus, sizeof(pCaptureStatus)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pCaptureStatus, sizeof(pCaptureStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamGetCaptureInfo_v2);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&captureStatus_out, sizeof(captureStatus_out)) < 0 ||
        rpc_write(&id_out, sizeof(id_out)) < 0 ||
        rpc_write(&graph_out, sizeof(graph_out)) < 0 ||
        rpc_write(&dependencies_out, sizeof(dependencies_out)) < 0 ||
        rpc_write(&numDependencies_out, sizeof(numDependencies_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&captureStatus_out, sizeof(captureStatus_out)) < 0 ||
        rpc_read(&id_out, sizeof(id_out)) < 0 ||
        rpc_read(&graph_out, sizeof(graph_out)) < 0 ||
        rpc_read(&dependencies_out, sizeof(dependencies_out)) < 0 ||
        rpc_read(&numDependencies_out, sizeof(numDependencies_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaStreamUpdateCaptureDependencies);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dependencies, sizeof(dependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventCreate);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event, sizeof(event)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventCreateWithFlags);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event, sizeof(event)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventRecord);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventRecordWithFlags);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventQuery);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventSynchronize);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventDestroy);
    if (request_id < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaEventElapsedTime);
    if (request_id < 0 ||
        rpc_write(&ms, sizeof(ms)) < 0 ||
        rpc_write(&start, sizeof(start)) < 0 ||
        rpc_write(&end, sizeof(end)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ms, sizeof(ms)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const struct cudaExternalMemoryHandleDesc* memHandleDesc) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaImportExternalMemory);
    if (request_id < 0 ||
        rpc_write(&extMem_out, sizeof(extMem_out)) < 0 ||
        rpc_write(&memHandleDesc, sizeof(memHandleDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extMem_out, sizeof(extMem_out)) < 0 ||
        rpc_read(&memHandleDesc, sizeof(memHandleDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc* bufferDesc) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaExternalMemoryGetMappedBuffer);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&extMem, sizeof(extMem)) < 0 ||
        rpc_write(&bufferDesc, sizeof(bufferDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_read(&bufferDesc, sizeof(bufferDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaExternalMemoryGetMappedMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_write(&extMem, sizeof(extMem)) < 0 ||
        rpc_write(&mipmapDesc, sizeof(mipmapDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_read(&mipmapDesc, sizeof(mipmapDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroyExternalMemory);
    if (request_id < 0 ||
        rpc_write(&extMem, sizeof(extMem)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const struct cudaExternalSemaphoreHandleDesc* semHandleDesc) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaImportExternalSemaphore);
    if (request_id < 0 ||
        rpc_write(&extSem_out, sizeof(extSem_out)) < 0 ||
        rpc_write(&semHandleDesc, sizeof(semHandleDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extSem_out, sizeof(extSem_out)) < 0 ||
        rpc_read(&semHandleDesc, sizeof(semHandleDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSignalExternalSemaphoresAsync_v2);
    if (request_id < 0 ||
        rpc_write(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_write(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_write(&numExtSems, sizeof(numExtSems)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_read(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaWaitExternalSemaphoresAsync_v2);
    if (request_id < 0 ||
        rpc_write(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_write(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_write(&numExtSems, sizeof(numExtSems)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&extSemArray, sizeof(extSemArray)) < 0 ||
        rpc_read(&paramsArray, sizeof(paramsArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroyExternalSemaphore);
    if (request_id < 0 ||
        rpc_write(&extSem, sizeof(extSem)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchKernel);
    if (request_id < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&gridDim, sizeof(gridDim)) < 0 ||
        rpc_write(&blockDim, sizeof(blockDim)) < 0 ||
        rpc_write(&args, sizeof(args)) < 0 ||
        rpc_write(&sharedMem, sizeof(sharedMem)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_read(&args, sizeof(args)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchKernelExC);
    if (request_id < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&args, sizeof(args)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&config, sizeof(config)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_read(&args, sizeof(args)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchCooperativeKernel);
    if (request_id < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&gridDim, sizeof(gridDim)) < 0 ||
        rpc_write(&blockDim, sizeof(blockDim)) < 0 ||
        rpc_write(&args, sizeof(args)) < 0 ||
        rpc_write(&sharedMem, sizeof(sharedMem)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_read(&args, sizeof(args)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchCooperativeKernelMultiDevice);
    if (request_id < 0 ||
        rpc_write(&launchParamsList, sizeof(launchParamsList)) < 0 ||
        rpc_write(&numDevices, sizeof(numDevices)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&launchParamsList, sizeof(launchParamsList)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFuncSetCacheConfig);
    if (request_id < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&cacheConfig, sizeof(cacheConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFuncSetSharedMemConfig);
    if (request_id < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&config, sizeof(config)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFuncGetAttributes);
    if (request_id < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&attr, sizeof(attr)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFuncSetAttribute);
    if (request_id < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaSetDoubleForDevice(double* d) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDoubleForDevice);
    if (request_id < 0 ||
        rpc_write(&d, sizeof(d)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&d, sizeof(d)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaSetDoubleForHost(double* d) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaSetDoubleForHost);
    if (request_id < 0 ||
        rpc_write(&d, sizeof(d)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&d, sizeof(d)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaLaunchHostFunc);
    if (request_id < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_write(&fn, sizeof(fn)) < 0 ||
        rpc_write(&userData, sizeof(userData)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&userData, sizeof(userData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessor);
    if (request_id < 0 ||
        rpc_write(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&blockSize, sizeof(blockSize)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaOccupancyAvailableDynamicSMemPerBlock);
    if (request_id < 0 ||
        rpc_write(&dynamicSmemSize, sizeof(dynamicSmemSize)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_write(&blockSize, sizeof(blockSize)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dynamicSmemSize, sizeof(dynamicSmemSize)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    if (request_id < 0 ||
        rpc_write(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&blockSize, sizeof(blockSize)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numBlocks, sizeof(numBlocks)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaOccupancyMaxPotentialClusterSize(int* clusterSize, const void* func, const cudaLaunchConfig_t* launchConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaOccupancyMaxPotentialClusterSize);
    if (request_id < 0 ||
        rpc_write(&clusterSize, sizeof(clusterSize)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&launchConfig, sizeof(launchConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&clusterSize, sizeof(clusterSize)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_read(&launchConfig, sizeof(launchConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, const void* func, const cudaLaunchConfig_t* launchConfig) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaOccupancyMaxActiveClusters);
    if (request_id < 0 ||
        rpc_write(&numClusters, sizeof(numClusters)) < 0 ||
        rpc_write(&func, sizeof(func)) < 0 ||
        rpc_write(&launchConfig, sizeof(launchConfig)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&numClusters, sizeof(numClusters)) < 0 ||
        rpc_read(&func, sizeof(func)) < 0 ||
        rpc_read(&launchConfig, sizeof(launchConfig)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocManaged);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMalloc);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocHost);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocPitch);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&pitch, sizeof(pitch)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_read(&pitch, sizeof(pitch)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocArray);
    if (request_id < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&array, sizeof(array)) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFree(void* devPtr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFree);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFreeHost(void* ptr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeHost);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFreeArray(cudaArray_t array) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeArray);
    if (request_id < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmappedArray, sizeof(mipmappedArray)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostAlloc);
    if (request_id < 0 ||
        rpc_write(&pHost, sizeof(pHost)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pHost, sizeof(pHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostRegister);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaHostUnregister(void* ptr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostUnregister);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostGetDevicePointer);
    if (request_id < 0 ||
        rpc_write(&pDevice, sizeof(pDevice)) < 0 ||
        rpc_write(&pHost, sizeof(pHost)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pDevice, sizeof(pDevice)) < 0 ||
        rpc_read(&pHost, sizeof(pHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaHostGetFlags);
    if (request_id < 0 ||
        rpc_write(&pFlags, sizeof(pFlags)) < 0 ||
        rpc_write(&pHost, sizeof(pHost)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pFlags, sizeof(pFlags)) < 0 ||
        rpc_read(&pHost, sizeof(pHost)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMalloc3D);
    if (request_id < 0 ||
        rpc_write(&pitchedDevPtr, sizeof(pitchedDevPtr)) < 0 ||
        rpc_write(&extent, sizeof(extent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pitchedDevPtr, sizeof(pitchedDevPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMalloc3DArray);
    if (request_id < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&extent, sizeof(extent)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&array, sizeof(array)) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmappedArray, sizeof(mipmappedArray)) < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&extent, sizeof(extent)) < 0 ||
        rpc_write(&numLevels, sizeof(numLevels)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mipmappedArray, sizeof(mipmappedArray)) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetMipmappedArrayLevel);
    if (request_id < 0 ||
        rpc_write(&levelArray, sizeof(levelArray)) < 0 ||
        rpc_write(&mipmappedArray, sizeof(mipmappedArray)) < 0 ||
        rpc_write(&level, sizeof(level)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&levelArray, sizeof(levelArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy3D);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy3DPeer);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy3DAsync);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy3DPeerAsync);
    if (request_id < 0 ||
        rpc_write(&p, sizeof(p)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&p, sizeof(p)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemGetInfo);
    if (request_id < 0 ||
        rpc_write(&free, sizeof(free)) < 0 ||
        rpc_write(&total, sizeof(total)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&free, sizeof(free)) < 0 ||
        rpc_read(&total, sizeof(total)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetInfo);
    if (request_id < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&extent, sizeof(extent)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_read(&extent, sizeof(extent)) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetPlane);
    if (request_id < 0 ||
        rpc_write(&pPlaneArray, sizeof(pPlaneArray)) < 0 ||
        rpc_write(&hArray, sizeof(hArray)) < 0 ||
        rpc_write(&planeIdx, sizeof(planeIdx)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pPlaneArray, sizeof(pPlaneArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMipmappedArrayGetMemoryRequirements);
    if (request_id < 0 ||
        rpc_write(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_write(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memoryRequirements, sizeof(memoryRequirements)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaArray_t array) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMipmappedArrayGetSparseProperties);
    if (request_id < 0 ||
        rpc_write(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_write(&mipmap, sizeof(mipmap)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&sparseProperties, sizeof(sparseProperties)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyPeer);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2D);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&dpitch, sizeof(dpitch)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&spitch, sizeof(spitch)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DToArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&spitch, sizeof(spitch)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DFromArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&dpitch, sizeof(dpitch)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DArrayToArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&wOffsetDst, sizeof(wOffsetDst)) < 0 ||
        rpc_write(&hOffsetDst, sizeof(hOffsetDst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&wOffsetSrc, sizeof(wOffsetSrc)) < 0 ||
        rpc_write(&hOffsetSrc, sizeof(hOffsetSrc)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyToSymbol);
    if (request_id < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyFromSymbol);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyPeerAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&srcDevice, sizeof(srcDevice)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&dpitch, sizeof(dpitch)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&spitch, sizeof(spitch)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DToArrayAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&spitch, sizeof(spitch)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy2DFromArrayAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&dpitch, sizeof(dpitch)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyToSymbolAsync);
    if (request_id < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyFromSymbolAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset2D);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&pitch, sizeof(pitch)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset3D);
    if (request_id < 0 ||
        rpc_write(&pitchedDevPtr, sizeof(pitchedDevPtr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&extent, sizeof(extent)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemsetAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset2DAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&pitch, sizeof(pitch)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&width, sizeof(width)) < 0 ||
        rpc_write(&height, sizeof(height)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemset3DAsync);
    if (request_id < 0 ||
        rpc_write(&pitchedDevPtr, sizeof(pitchedDevPtr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_write(&extent, sizeof(extent)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetSymbolAddress);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetSymbolSize);
    if (request_id < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&size, sizeof(size)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPrefetchAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&dstDevice, sizeof(dstDevice)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemAdvise);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&advice, sizeof(advice)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemRangeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&dataSize, sizeof(dataSize)) < 0 ||
        rpc_write(&attribute, sizeof(attribute)) < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, enum cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemRangeGetAttributes);
    if (request_id < 0 ||
        rpc_write(&data, sizeof(data)) < 0 ||
        rpc_write(&dataSizes, sizeof(dataSizes)) < 0 ||
        rpc_write(&attributes, sizeof(attributes)) < 0 ||
        rpc_write(&numAttributes, sizeof(numAttributes)) < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&data, sizeof(data)) < 0 ||
        rpc_read(&dataSizes, sizeof(dataSizes)) < 0 ||
        rpc_read(&attributes, sizeof(attributes)) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyToArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyFromArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyArrayToArray);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&wOffsetDst, sizeof(wOffsetDst)) < 0 ||
        rpc_write(&hOffsetDst, sizeof(hOffsetDst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&wOffsetSrc, sizeof(wOffsetSrc)) < 0 ||
        rpc_write(&hOffsetSrc, sizeof(hOffsetSrc)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyToArrayAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyFromArrayAsync);
    if (request_id < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&wOffset, sizeof(wOffset)) < 0 ||
        rpc_write(&hOffset, sizeof(hOffset)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaFreeAsync);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolTrimTo);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&minBytesToKeep, sizeof(minBytesToKeep)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolSetAttribute);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolGetAttribute);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc* descList, size_t count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolSetAccess);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&descList, sizeof(descList)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&descList, sizeof(descList)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags* flags, cudaMemPool_t memPool, struct cudaMemLocation* location) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolGetAccess);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&location, sizeof(location)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_read(&location, sizeof(location)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const struct cudaMemPoolProps* poolProps) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolCreate);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&poolProps, sizeof(poolProps)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memPool, sizeof(memPool)) < 0 ||
        rpc_read(&poolProps, sizeof(poolProps)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolDestroy);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMallocFromPoolAsync);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolExportToShareableHandle);
    if (request_id < 0 ||
        rpc_write(&shareableHandle, sizeof(shareableHandle)) < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&handleType, sizeof(handleType)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&shareableHandle, sizeof(shareableHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolImportFromShareableHandle);
    if (request_id < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&shareableHandle, sizeof(shareableHandle)) < 0 ||
        rpc_write(&handleType, sizeof(handleType)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&memPool, sizeof(memPool)) < 0 ||
        rpc_read(&shareableHandle, sizeof(shareableHandle)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData* exportData, void* ptr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolExportPointer);
    if (request_id < 0 ||
        rpc_write(&exportData, sizeof(exportData)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&exportData, sizeof(exportData)) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData* exportData) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemPoolImportPointer);
    if (request_id < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&memPool, sizeof(memPool)) < 0 ||
        rpc_write(&exportData, sizeof(exportData)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_read(&exportData, sizeof(exportData)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaPointerGetAttributes);
    if (request_id < 0 ||
        rpc_write(&attributes, sizeof(attributes)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&attributes, sizeof(attributes)) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceCanAccessPeer);
    if (request_id < 0 ||
        rpc_write(&canAccessPeer, sizeof(canAccessPeer)) < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&peerDevice, sizeof(peerDevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&canAccessPeer, sizeof(canAccessPeer)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceEnablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerDevice, sizeof(peerDevice)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceDisablePeerAccess);
    if (request_id < 0 ||
        rpc_write(&peerDevice, sizeof(peerDevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsUnregisterResource);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsResourceSetMapFlags);
    if (request_id < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsMapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&resources, sizeof(resources)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&resources, sizeof(resources)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsUnmapResources);
    if (request_id < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&resources, sizeof(resources)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&resources, sizeof(resources)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsResourceGetMappedPointer);
    if (request_id < 0 ||
        rpc_write(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_write(&size, sizeof(size)) < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&devPtr, sizeof(devPtr)) < 0 ||
        rpc_read(&size, sizeof(size)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsSubResourceGetMappedArray);
    if (request_id < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_write(&arrayIndex, sizeof(arrayIndex)) < 0 ||
        rpc_write(&mipLevel, sizeof(mipLevel)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&array, sizeof(array)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphicsResourceGetMappedMipmappedArray);
    if (request_id < 0 ||
        rpc_write(&mipmappedArray, sizeof(mipmappedArray)) < 0 ||
        rpc_write(&resource, sizeof(resource)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&mipmappedArray, sizeof(mipmappedArray)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc* desc, cudaArray_const_t array) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetChannelDesc);
    if (request_id < 0 ||
        rpc_write(&desc, sizeof(desc)) < 0 ||
        rpc_write(&array, sizeof(array)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&desc, sizeof(desc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
    struct cudaChannelFormatDesc return_value;

    int request_id = rpc_start_request(RPC_cudaCreateChannelDesc);
    if (request_id < 0 ||
        rpc_write(&x, sizeof(x)) < 0 ||
        rpc_write(&y, sizeof(y)) < 0 ||
        rpc_write(&z, sizeof(z)) < 0 ||
        rpc_write(&w, sizeof(w)) < 0 ||
        rpc_write(&f, sizeof(f)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const struct cudaResourceDesc* pResDesc, const struct cudaTextureDesc* pTexDesc, const struct cudaResourceViewDesc* pResViewDesc) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaCreateTextureObject);
    if (request_id < 0 ||
        rpc_write(&pTexObject, sizeof(pTexObject)) < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_write(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_write(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pTexObject, sizeof(pTexObject)) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_read(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_read(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroyTextureObject);
    if (request_id < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetTextureObjectResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetTextureObjectTextureDesc);
    if (request_id < 0 ||
        rpc_write(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pTexDesc, sizeof(pTexDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetTextureObjectResourceViewDesc);
    if (request_id < 0 ||
        rpc_write(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_write(&texObject, sizeof(texObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pResViewDesc, sizeof(pResViewDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const struct cudaResourceDesc* pResDesc) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaCreateSurfaceObject);
    if (request_id < 0 ||
        rpc_write(&pSurfObject, sizeof(pSurfObject)) < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pSurfObject, sizeof(pSurfObject)) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDestroySurfaceObject);
    if (request_id < 0 ||
        rpc_write(&surfObject, sizeof(surfObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetSurfaceObjectResourceDesc);
    if (request_id < 0 ||
        rpc_write(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_write(&surfObject, sizeof(surfObject)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pResDesc, sizeof(pResDesc)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDriverGetVersion(int* driverVersion) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDriverGetVersion);
    if (request_id < 0 ||
        rpc_write(&driverVersion, sizeof(driverVersion)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&driverVersion, sizeof(driverVersion)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaRuntimeGetVersion);
    if (request_id < 0 ||
        rpc_write(&runtimeVersion, sizeof(runtimeVersion)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&runtimeVersion, sizeof(runtimeVersion)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphCreate);
    if (request_id < 0 ||
        rpc_write(&pGraph, sizeof(pGraph)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraph, sizeof(pGraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddKernelNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeCopyAttributes);
    if (request_id < 0 ||
        rpc_write(&hSrc, sizeof(hSrc)) < 0 ||
        rpc_write(&hDst, sizeof(hDst)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeGetAttribute);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value_out, sizeof(value_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value_out, sizeof(value_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphKernelNodeSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms* pCopyParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemcpyNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&pCopyParams, sizeof(pCopyParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&pCopyParams, sizeof(pCopyParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemcpyNodeToSymbol);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemcpyNodeFromSymbol);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemcpyNode1D);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemcpyNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemcpyNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemcpyNodeSetParamsToSymbol);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemcpyNodeSetParamsFromSymbol);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemcpyNodeSetParams1D);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemsetParams* pMemsetParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemsetNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&pMemsetParams, sizeof(pMemsetParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&pMemsetParams, sizeof(pMemsetParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemsetNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemsetNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaHostNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddHostNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphHostNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphHostNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddChildGraphNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&childGraph, sizeof(childGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphChildGraphNodeGetGraph);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pGraph, sizeof(pGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraph, sizeof(pGraph)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddEmptyNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddEventRecordNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventRecordNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&event_out, sizeof(event_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event_out, sizeof(event_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddEventWaitNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventWaitNodeGetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&event_out, sizeof(event_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&event_out, sizeof(event_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddExternalSemaphoresSignalNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams* params_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExternalSemaphoresSignalNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&params_out, sizeof(params_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&params_out, sizeof(params_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExternalSemaphoresSignalNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddExternalSemaphoresWaitNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams* params_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExternalSemaphoresWaitNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&params_out, sizeof(params_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&params_out, sizeof(params_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExternalSemaphoresWaitNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemAllocNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams* params_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemAllocNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&params_out, sizeof(params_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&params_out, sizeof(params_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddMemFreeNode);
    if (request_id < 0 ||
        rpc_write(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_write(&dptr, sizeof(dptr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphNode, sizeof(pGraphNode)) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&dptr, sizeof(dptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphMemFreeNodeGetParams);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&dptr_out, sizeof(dptr_out)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dptr_out, sizeof(dptr_out)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGraphMemTrim(int device) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGraphMemTrim);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceGetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaDeviceSetGraphMemAttribute);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(device)) < 0 ||
        rpc_write(&attr, sizeof(attr)) < 0 ||
        rpc_write(&value, sizeof(value)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&value, sizeof(value)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphClone);
    if (request_id < 0 ||
        rpc_write(&pGraphClone, sizeof(pGraphClone)) < 0 ||
        rpc_write(&originalGraph, sizeof(originalGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphClone, sizeof(pGraphClone)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeFindInClone);
    if (request_id < 0 ||
        rpc_write(&pNode, sizeof(pNode)) < 0 ||
        rpc_write(&originalNode, sizeof(originalNode)) < 0 ||
        rpc_write(&clonedGraph, sizeof(clonedGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNode, sizeof(pNode)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* pType) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetType);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pType, sizeof(pType)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pType, sizeof(pType)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphGetNodes);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&nodes, sizeof(nodes)) < 0 ||
        rpc_write(&numNodes, sizeof(numNodes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodes, sizeof(nodes)) < 0 ||
        rpc_read(&numNodes, sizeof(numNodes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphGetRootNodes);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&pRootNodes, sizeof(pRootNodes)) < 0 ||
        rpc_write(&pNumRootNodes, sizeof(pNumRootNodes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pRootNodes, sizeof(pRootNodes)) < 0 ||
        rpc_read(&pNumRootNodes, sizeof(pNumRootNodes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphGetEdges);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&from, sizeof(from)) < 0 ||
        rpc_write(&to, sizeof(to)) < 0 ||
        rpc_write(&numEdges, sizeof(numEdges)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&from, sizeof(from)) < 0 ||
        rpc_read(&to, sizeof(to)) < 0 ||
        rpc_read(&numEdges, sizeof(numEdges)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetDependencies);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_write(&pNumDependencies, sizeof(pNumDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pDependencies, sizeof(pDependencies)) < 0 ||
        rpc_read(&pNumDependencies, sizeof(pNumDependencies)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetDependentNodes);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pDependentNodes, sizeof(pDependentNodes)) < 0 ||
        rpc_write(&pNumDependentNodes, sizeof(pNumDependentNodes)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pDependentNodes, sizeof(pDependentNodes)) < 0 ||
        rpc_read(&pNumDependentNodes, sizeof(pNumDependentNodes)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphAddDependencies);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&from, sizeof(from)) < 0 ||
        rpc_write(&to, sizeof(to)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&from, sizeof(from)) < 0 ||
        rpc_read(&to, sizeof(to)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from, const cudaGraphNode_t* to, size_t numDependencies) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphRemoveDependencies);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&from, sizeof(from)) < 0 ||
        rpc_write(&to, sizeof(to)) < 0 ||
        rpc_write(&numDependencies, sizeof(numDependencies)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&from, sizeof(from)) < 0 ||
        rpc_read(&to, sizeof(to)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphDestroyNode);
    if (request_id < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphInstantiate);
    if (request_id < 0 ||
        rpc_write(&pGraphExec, sizeof(pGraphExec)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphExec, sizeof(pGraphExec)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphInstantiateWithFlags);
    if (request_id < 0 ||
        rpc_write(&pGraphExec, sizeof(pGraphExec)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphExec, sizeof(pGraphExec)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphInstantiateWithParams);
    if (request_id < 0 ||
        rpc_write(&pGraphExec, sizeof(pGraphExec)) < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&instantiateParams, sizeof(instantiateParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pGraphExec, sizeof(pGraphExec)) < 0 ||
        rpc_read(&instantiateParams, sizeof(instantiateParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecGetFlags);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(graphExec)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&flags, sizeof(flags)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecKernelNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecMemcpyNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecMemcpyNodeSetParamsToSymbol);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecMemcpyNodeSetParamsFromSymbol);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&offset, sizeof(offset)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecMemcpyNodeSetParams1D);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&dst, sizeof(dst)) < 0 ||
        rpc_write(&src, sizeof(src)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&kind, sizeof(kind)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&dst, sizeof(dst)) < 0 ||
        rpc_read(&src, sizeof(src)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecMemsetNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecHostNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&pNodeParams, sizeof(pNodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecChildGraphNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&node, sizeof(node)) < 0 ||
        rpc_write(&childGraph, sizeof(childGraph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecEventRecordNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecEventWaitNodeSetEvent);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&event, sizeof(event)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecExternalSemaphoresSignalNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams* nodeParams) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecExternalSemaphoresWaitNodeSetParams);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&nodeParams, sizeof(nodeParams)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeSetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphNodeGetEnabled);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hNode, sizeof(hNode)) < 0 ||
        rpc_write(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&isEnabled, sizeof(isEnabled)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecUpdate);
    if (request_id < 0 ||
        rpc_write(&hGraphExec, sizeof(hGraphExec)) < 0 ||
        rpc_write(&hGraph, sizeof(hGraph)) < 0 ||
        rpc_write(&resultInfo, sizeof(resultInfo)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&resultInfo, sizeof(resultInfo)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphUpload);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(graphExec)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphLaunch);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(graphExec)) < 0 ||
        rpc_write(&stream, sizeof(stream)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphExecDestroy);
    if (request_id < 0 ||
        rpc_write(&graphExec, sizeof(graphExec)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphDestroy);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphDebugDotPrint);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&path, sizeof(path)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&path, sizeof(path)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaUserObjectCreate);
    if (request_id < 0 ||
        rpc_write(&object_out, sizeof(object_out)) < 0 ||
        rpc_write(&ptr, sizeof(ptr)) < 0 ||
        rpc_write(&destroy, sizeof(destroy)) < 0 ||
        rpc_write(&initialRefcount, sizeof(initialRefcount)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&object_out, sizeof(object_out)) < 0 ||
        rpc_read(&ptr, sizeof(ptr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaUserObjectRetain);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaUserObjectRelease);
    if (request_id < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphRetainUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGraphReleaseUserObject);
    if (request_id < 0 ||
        rpc_write(&graph, sizeof(graph)) < 0 ||
        rpc_write(&object, sizeof(object)) < 0 ||
        rpc_write(&count, sizeof(count)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult* driverStatus) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetDriverEntryPoint);
    if (request_id < 0 ||
        rpc_write(&symbol, sizeof(symbol)) < 0 ||
        rpc_write(&funcPtr, sizeof(funcPtr)) < 0 ||
        rpc_write(&flags, sizeof(flags)) < 0 ||
        rpc_write(&driverStatus, sizeof(driverStatus)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&symbol, sizeof(symbol)) < 0 ||
        rpc_read(&funcPtr, sizeof(funcPtr)) < 0 ||
        rpc_read(&driverStatus, sizeof(driverStatus)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetExportTable);
    if (request_id < 0 ||
        rpc_write(&ppExportTable, sizeof(ppExportTable)) < 0 ||
        rpc_write(&pExportTableId, sizeof(pExportTableId)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&ppExportTable, sizeof(ppExportTable)) < 0 ||
        rpc_read(&pExportTableId, sizeof(pExportTableId)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

cudaError_t cudaGetFuncBySymbol(cudaFunction_t* functionPtr, const void* symbolPtr) {
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaGetFuncBySymbol);
    if (request_id < 0 ||
        rpc_write(&functionPtr, sizeof(functionPtr)) < 0 ||
        rpc_write(&symbolPtr, sizeof(symbolPtr)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(&functionPtr, sizeof(functionPtr)) < 0 ||
        rpc_read(&symbolPtr, sizeof(symbolPtr)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

