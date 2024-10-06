#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const size_t size);
extern int rpc_write(const void *conn, const void *data, const size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);

int handle_nvmlInit_v2(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlInit_v2();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlInitWithFlags(void *conn) {
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlInitWithFlags(flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlShutdown(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlShutdown();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlErrorString(void *conn) {
    nvmlReturn_t result;
    if (rpc_read(conn, &result, sizeof(result)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    const char* return_value = nvmlErrorString(result);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlSystemGetDriverVersion(void *conn) {
    unsigned int length;
    char* version;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetDriverVersion(version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;
}

int handle_nvmlSystemGetNVMLVersion(void *conn) {
    unsigned int length;
    char* version;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetNVMLVersion(version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;
}

int handle_nvmlSystemGetCudaDriverVersion(void *conn) {
    int* cudaDriverVersion;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetCudaDriverVersion(cudaDriverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &cudaDriverVersion, sizeof(cudaDriverVersion)) < 0)
        return -1;
}

int handle_nvmlSystemGetCudaDriverVersion_v2(void *conn) {
    int* cudaDriverVersion;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &cudaDriverVersion, sizeof(cudaDriverVersion)) < 0)
        return -1;
}

int handle_nvmlSystemGetProcessName(void *conn) {
    unsigned int pid;
    unsigned int length;
    char* name;
    if (rpc_read(conn, &pid, sizeof(pid)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetProcessName(pid, name, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, name, length * sizeof(name)) < 0)
        return -1;
}

int handle_nvmlUnitGetCount(void *conn) {
    unsigned int* unitCount;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetCount(unitCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &unitCount, sizeof(unitCount)) < 0)
        return -1;
}

int handle_nvmlUnitGetHandleByIndex(void *conn) {
    unsigned int index;
    nvmlUnit_t* unit;
    if (rpc_read(conn, &index, sizeof(index)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetHandleByIndex(index, unit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &unit, sizeof(unit)) < 0)
        return -1;
}

int handle_nvmlUnitGetUnitInfo(void *conn) {
    nvmlUnit_t unit;
    nvmlUnitInfo_t* info;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetUnitInfo(unit, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlUnitGetLedState(void *conn) {
    nvmlUnit_t unit;
    nvmlLedState_t* state;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetLedState(unit, state);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &state, sizeof(state)) < 0)
        return -1;
}

int handle_nvmlUnitGetPsuInfo(void *conn) {
    nvmlUnit_t unit;
    nvmlPSUInfo_t* psu;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetPsuInfo(unit, psu);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &psu, sizeof(psu)) < 0)
        return -1;
}

int handle_nvmlUnitGetTemperature(void *conn) {
    nvmlUnit_t unit;
    unsigned int type;
    unsigned int* temp;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetTemperature(unit, type, temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(temp)) < 0)
        return -1;
}

int handle_nvmlUnitGetFanSpeedInfo(void *conn) {
    nvmlUnit_t unit;
    nvmlUnitFanSpeeds_t* fanSpeeds;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetFanSpeedInfo(unit, fanSpeeds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fanSpeeds, sizeof(fanSpeeds)) < 0)
        return -1;
}

int handle_nvmlUnitGetDevices(void *conn) {
    nvmlUnit_t unit;
    unsigned int* deviceCount;
    nvmlDevice_t* devices;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    if (rpc_read(conn, &deviceCount, sizeof(deviceCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitGetDevices(unit, deviceCount, devices);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceCount, sizeof(deviceCount)) < 0)
        return -1;
    if (rpc_write(conn, &devices, sizeof(devices)) < 0)
        return -1;
}

int handle_nvmlSystemGetHicVersion(void *conn) {
    unsigned int* hwbcCount;
    nvmlHwbcEntry_t* hwbcEntries;
    if (rpc_read(conn, &hwbcCount, sizeof(hwbcCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetHicVersion(hwbcCount, hwbcEntries);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &hwbcCount, sizeof(hwbcCount)) < 0)
        return -1;
    if (rpc_write(conn, hwbcEntries, *hwbcCount * sizeof(hwbcEntries)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCount_v2(void *conn) {
    unsigned int* deviceCount;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCount_v2(deviceCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceCount, sizeof(deviceCount)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAttributes_v2(void *conn) {
    nvmlDevice_t device;
    nvmlDeviceAttributes_t* attributes;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAttributes_v2(device, attributes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
}

int handle_nvmlDeviceGetHandleByIndex_v2(void *conn) {
    unsigned int index;
    nvmlDevice_t* device;
    if (rpc_read(conn, &index, sizeof(index)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetHandleByIndex_v2(index, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_nvmlDeviceGetHandleBySerial(void *conn) {
    size_t serial_len;
    const char* serial;
    nvmlDevice_t* device;
    if (rpc_read(conn, &serial_len, sizeof(serial_len)) < 0)
        return -1;
    if (rpc_read(conn, &serial, serial_len) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetHandleBySerial(serial, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_nvmlDeviceGetHandleByUUID(void *conn) {
    size_t uuid_len;
    const char* uuid;
    nvmlDevice_t* device;
    if (rpc_read(conn, &uuid_len, sizeof(uuid_len)) < 0)
        return -1;
    if (rpc_read(conn, &uuid, uuid_len) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetHandleByUUID(uuid, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(void *conn) {
    size_t pciBusId_len;
    const char* pciBusId;
    nvmlDevice_t* device;
    if (rpc_read(conn, &pciBusId_len, sizeof(pciBusId_len)) < 0)
        return -1;
    if (rpc_read(conn, &pciBusId, pciBusId_len) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_nvmlDeviceGetName(void *conn) {
    nvmlDevice_t device;
    size_t name_len;
    char* name;
    unsigned int length;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetName(device, name, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    if (rpc_write(conn, &name, name_len) < 0)
        return -1;
}

int handle_nvmlDeviceGetBrand(void *conn) {
    nvmlDevice_t device;
    nvmlBrandType_t* type;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetBrand(device, type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &type, sizeof(type)) < 0)
        return -1;
}

int handle_nvmlDeviceGetIndex(void *conn) {
    nvmlDevice_t device;
    unsigned int* index;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetIndex(device, index);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &index, sizeof(index)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSerial(void *conn) {
    nvmlDevice_t device;
    size_t serial_len;
    char* serial;
    unsigned int length;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSerial(device, serial, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &serial_len, sizeof(serial_len)) < 0)
        return -1;
    if (rpc_write(conn, &serial, serial_len) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemoryAffinity(void *conn) {
    nvmlDevice_t device;
    unsigned int nodeSetSize;
    unsigned long* nodeSet;
    nvmlAffinityScope_t scope;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &nodeSetSize, sizeof(nodeSetSize)) < 0)
        return -1;
    if (rpc_read(conn, &scope, sizeof(scope)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, nodeSet, nodeSetSize * sizeof(nodeSet)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCpuAffinityWithinScope(void *conn) {
    nvmlDevice_t device;
    unsigned int cpuSetSize;
    unsigned long* cpuSet;
    nvmlAffinityScope_t scope;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &cpuSetSize, sizeof(cpuSetSize)) < 0)
        return -1;
    if (rpc_read(conn, &scope, sizeof(scope)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, cpuSet, cpuSetSize * sizeof(cpuSet)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCpuAffinity(void *conn) {
    nvmlDevice_t device;
    unsigned int cpuSetSize;
    unsigned long* cpuSet;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &cpuSetSize, sizeof(cpuSetSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, cpuSet, cpuSetSize * sizeof(cpuSet)) < 0)
        return -1;
}

int handle_nvmlDeviceSetCpuAffinity(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetCpuAffinity(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceClearCpuAffinity(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceClearCpuAffinity(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetTopologyCommonAncestor(void *conn) {
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuTopologyLevel_t* pathInfo;
    if (rpc_read(conn, &device1, sizeof(device1)) < 0)
        return -1;
    if (rpc_read(conn, &device2, sizeof(device2)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pathInfo, sizeof(pathInfo)) < 0)
        return -1;
}

int handle_nvmlDeviceGetTopologyNearestGpus(void *conn) {
    nvmlDevice_t device;
    nvmlGpuTopologyLevel_t level;
    unsigned int* count;
    nvmlDevice_t* deviceArray;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &level, sizeof(level)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, deviceArray, *count * sizeof(deviceArray)) < 0)
        return -1;
}

int handle_nvmlSystemGetTopologyGpuSet(void *conn) {
    unsigned int cpuNumber;
    unsigned int* count;
    nvmlDevice_t* deviceArray;
    if (rpc_read(conn, &cpuNumber, sizeof(cpuNumber)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, deviceArray, *count * sizeof(deviceArray)) < 0)
        return -1;
}

int handle_nvmlDeviceGetP2PStatus(void *conn) {
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuP2PCapsIndex_t p2pIndex;
    nvmlGpuP2PStatus_t* p2pStatus;
    if (rpc_read(conn, &device1, sizeof(device1)) < 0)
        return -1;
    if (rpc_read(conn, &device2, sizeof(device2)) < 0)
        return -1;
    if (rpc_read(conn, &p2pIndex, sizeof(p2pIndex)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p2pStatus, sizeof(p2pStatus)) < 0)
        return -1;
}

int handle_nvmlDeviceGetUUID(void *conn) {
    nvmlDevice_t device;
    size_t uuid_len;
    char* uuid;
    unsigned int length;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetUUID(device, uuid, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &uuid_len, sizeof(uuid_len)) < 0)
        return -1;
    if (rpc_write(conn, &uuid, uuid_len) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetMdevUUID(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    size_t mdevUuid_len;
    char* mdevUuid;
    unsigned int size;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mdevUuid_len, sizeof(mdevUuid_len)) < 0)
        return -1;
    if (rpc_write(conn, &mdevUuid, mdevUuid_len) < 0)
        return -1;
}

int handle_nvmlDeviceGetMinorNumber(void *conn) {
    nvmlDevice_t device;
    unsigned int* minorNumber;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMinorNumber(device, minorNumber);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minorNumber, sizeof(minorNumber)) < 0)
        return -1;
}

int handle_nvmlDeviceGetBoardPartNumber(void *conn) {
    nvmlDevice_t device;
    unsigned int length;
    char* partNumber;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetBoardPartNumber(device, partNumber, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, partNumber, length * sizeof(partNumber)) < 0)
        return -1;
}

int handle_nvmlDeviceGetInforomVersion(void *conn) {
    nvmlDevice_t device;
    nvmlInforomObject_t object;
    unsigned int length;
    char* version;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetInforomVersion(device, object, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;
}

int handle_nvmlDeviceGetInforomImageVersion(void *conn) {
    nvmlDevice_t device;
    unsigned int length;
    char* version;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetInforomImageVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;
}

int handle_nvmlDeviceGetInforomConfigurationChecksum(void *conn) {
    nvmlDevice_t device;
    unsigned int* checksum;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetInforomConfigurationChecksum(device, checksum);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &checksum, sizeof(checksum)) < 0)
        return -1;
}

int handle_nvmlDeviceValidateInforom(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceValidateInforom(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetDisplayMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* display;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDisplayMode(device, display);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &display, sizeof(display)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDisplayActive(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* isActive;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDisplayActive(device, isActive);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isActive, sizeof(isActive)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPersistenceMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPersistenceMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPciInfo_v3(void *conn) {
    nvmlDevice_t device;
    nvmlPciInfo_t* pci;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPciInfo_v3(device, pci);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pci, sizeof(pci)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(void *conn) {
    nvmlDevice_t device;
    unsigned int* maxLinkGen;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxLinkGen, sizeof(maxLinkGen)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuMaxPcieLinkGeneration(void *conn) {
    nvmlDevice_t device;
    unsigned int* maxLinkGenDevice;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuMaxPcieLinkGeneration(device, maxLinkGenDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxLinkGenDevice, sizeof(maxLinkGenDevice)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(void *conn) {
    nvmlDevice_t device;
    unsigned int* maxLinkWidth;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxLinkWidth, sizeof(maxLinkWidth)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(void *conn) {
    nvmlDevice_t device;
    unsigned int* currLinkGen;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currLinkGen, sizeof(currLinkGen)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(void *conn) {
    nvmlDevice_t device;
    unsigned int* currLinkWidth;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currLinkWidth, sizeof(currLinkWidth)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPcieThroughput(void *conn) {
    nvmlDevice_t device;
    nvmlPcieUtilCounter_t counter;
    unsigned int* value;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPcieThroughput(device, counter, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPcieReplayCounter(void *conn) {
    nvmlDevice_t device;
    unsigned int* value;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPcieReplayCounter(device, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_nvmlDeviceGetClockInfo(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int* clock;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetClockInfo(device, type, clock);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clock, sizeof(clock)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMaxClockInfo(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int* clock;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMaxClockInfo(device, type, clock);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clock, sizeof(clock)) < 0)
        return -1;
}

int handle_nvmlDeviceGetApplicationsClock(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int* clockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &clockType, sizeof(clockType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetApplicationsClock(device, clockType, clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(clockMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDefaultApplicationsClock(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int* clockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &clockType, sizeof(clockType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDefaultApplicationsClock(device, clockType, clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(clockMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceResetApplicationsClocks(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceResetApplicationsClocks(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetClock(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    nvmlClockId_t clockId;
    unsigned int* clockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &clockType, sizeof(clockType)) < 0)
        return -1;
    if (rpc_read(conn, &clockId, sizeof(clockId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetClock(device, clockType, clockId, clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(clockMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMaxCustomerBoostClock(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int* clockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &clockType, sizeof(clockType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(clockMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSupportedMemoryClocks(void *conn) {
    nvmlDevice_t device;
    unsigned int* count;
    unsigned int* clocksMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, clocksMHz, *count * sizeof(clocksMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(void *conn) {
    nvmlDevice_t device;
    unsigned int memoryClockMHz;
    unsigned int* count;
    unsigned int* clocksMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &memoryClockMHz, sizeof(memoryClockMHz)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count, clocksMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, clocksMHz, *count * sizeof(clocksMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* isEnabled;
    nvmlEnableState_t* defaultIsEnabled;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
    if (rpc_write(conn, &defaultIsEnabled, sizeof(defaultIsEnabled)) < 0)
        return -1;
}

int handle_nvmlDeviceSetAutoBoostedClocksEnabled(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t enabled;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &enabled, sizeof(enabled)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t enabled;
    unsigned int flags;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &enabled, sizeof(enabled)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetFanSpeed(void *conn) {
    nvmlDevice_t device;
    unsigned int* speed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetFanSpeed(device, speed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &speed, sizeof(speed)) < 0)
        return -1;
}

int handle_nvmlDeviceGetFanSpeed_v2(void *conn) {
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int* speed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &fan, sizeof(fan)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetFanSpeed_v2(device, fan, speed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &speed, sizeof(speed)) < 0)
        return -1;
}

int handle_nvmlDeviceGetTargetFanSpeed(void *conn) {
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int* targetSpeed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &fan, sizeof(fan)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTargetFanSpeed(device, fan, targetSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &targetSpeed, sizeof(targetSpeed)) < 0)
        return -1;
}

int handle_nvmlDeviceSetDefaultFanSpeed_v2(void *conn) {
    nvmlDevice_t device;
    unsigned int fan;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &fan, sizeof(fan)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetDefaultFanSpeed_v2(device, fan);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetMinMaxFanSpeed(void *conn) {
    nvmlDevice_t device;
    unsigned int* minSpeed;
    unsigned int* maxSpeed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMinMaxFanSpeed(device, minSpeed, maxSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minSpeed, sizeof(minSpeed)) < 0)
        return -1;
    if (rpc_write(conn, &maxSpeed, sizeof(maxSpeed)) < 0)
        return -1;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(void *conn) {
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t* policy;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &fan, sizeof(fan)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetFanControlPolicy_v2(device, fan, policy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &policy, sizeof(policy)) < 0)
        return -1;
}

int handle_nvmlDeviceSetFanControlPolicy(void *conn) {
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &fan, sizeof(fan)) < 0)
        return -1;
    if (rpc_read(conn, &policy, sizeof(policy)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetFanControlPolicy(device, fan, policy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetNumFans(void *conn) {
    nvmlDevice_t device;
    unsigned int* numFans;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNumFans(device, numFans);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numFans, sizeof(numFans)) < 0)
        return -1;
}

int handle_nvmlDeviceGetTemperature(void *conn) {
    nvmlDevice_t device;
    nvmlTemperatureSensors_t sensorType;
    unsigned int* temp;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &sensorType, sizeof(sensorType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTemperature(device, sensorType, temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(temp)) < 0)
        return -1;
}

int handle_nvmlDeviceGetTemperatureThreshold(void *conn) {
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    unsigned int* temp;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &thresholdType, sizeof(thresholdType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(temp)) < 0)
        return -1;
}

int handle_nvmlDeviceSetTemperatureThreshold(void *conn) {
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    int* temp;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &thresholdType, sizeof(thresholdType)) < 0)
        return -1;
    if (rpc_read(conn, &temp, sizeof(temp)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(temp)) < 0)
        return -1;
}

int handle_nvmlDeviceGetThermalSettings(void *conn) {
    nvmlDevice_t device;
    unsigned int sensorIndex;
    nvmlGpuThermalSettings_t* pThermalSettings;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &sensorIndex, sizeof(sensorIndex)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetThermalSettings(device, sensorIndex, pThermalSettings);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pThermalSettings, sizeof(pThermalSettings)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPerformanceState(void *conn) {
    nvmlDevice_t device;
    nvmlPstates_t* pState;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPerformanceState(device, pState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pState, sizeof(pState)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCurrentClocksThrottleReasons(void *conn) {
    nvmlDevice_t device;
    unsigned long long* clocksThrottleReasons;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clocksThrottleReasons, sizeof(clocksThrottleReasons)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSupportedClocksThrottleReasons(void *conn) {
    nvmlDevice_t device;
    unsigned long long* supportedClocksThrottleReasons;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSupportedClocksThrottleReasons(device, supportedClocksThrottleReasons);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &supportedClocksThrottleReasons, sizeof(supportedClocksThrottleReasons)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerState(void *conn) {
    nvmlDevice_t device;
    nvmlPstates_t* pState;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerState(device, pState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pState, sizeof(pState)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerManagementMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerManagementMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerManagementLimit(void *conn) {
    nvmlDevice_t device;
    unsigned int* limit;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerManagementLimit(device, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &limit, sizeof(limit)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerManagementLimitConstraints(void *conn) {
    nvmlDevice_t device;
    unsigned int* minLimit;
    unsigned int* maxLimit;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minLimit, sizeof(minLimit)) < 0)
        return -1;
    if (rpc_write(conn, &maxLimit, sizeof(maxLimit)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(void *conn) {
    nvmlDevice_t device;
    unsigned int* defaultLimit;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &defaultLimit, sizeof(defaultLimit)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerUsage(void *conn) {
    nvmlDevice_t device;
    unsigned int* power;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerUsage(device, power);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &power, sizeof(power)) < 0)
        return -1;
}

int handle_nvmlDeviceGetTotalEnergyConsumption(void *conn) {
    nvmlDevice_t device;
    unsigned long long* energy;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTotalEnergyConsumption(device, energy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &energy, sizeof(energy)) < 0)
        return -1;
}

int handle_nvmlDeviceGetEnforcedPowerLimit(void *conn) {
    nvmlDevice_t device;
    unsigned int* limit;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetEnforcedPowerLimit(device, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &limit, sizeof(limit)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuOperationMode(void *conn) {
    nvmlDevice_t device;
    nvmlGpuOperationMode_t* current;
    nvmlGpuOperationMode_t* pending;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuOperationMode(device, current, pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(current)) < 0)
        return -1;
    if (rpc_write(conn, &pending, sizeof(pending)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemoryInfo(void *conn) {
    nvmlDevice_t device;
    nvmlMemory_t* memory;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemoryInfo(device, memory);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memory, sizeof(memory)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemoryInfo_v2(void *conn) {
    nvmlDevice_t device;
    nvmlMemory_v2_t* memory;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemoryInfo_v2(device, memory);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memory, sizeof(memory)) < 0)
        return -1;
}

int handle_nvmlDeviceGetComputeMode(void *conn) {
    nvmlDevice_t device;
    nvmlComputeMode_t* mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetComputeMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCudaComputeCapability(void *conn) {
    nvmlDevice_t device;
    int* major;
    int* minor;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCudaComputeCapability(device, major, minor);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &major, sizeof(major)) < 0)
        return -1;
    if (rpc_write(conn, &minor, sizeof(minor)) < 0)
        return -1;
}

int handle_nvmlDeviceGetEccMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* current;
    nvmlEnableState_t* pending;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetEccMode(device, current, pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(current)) < 0)
        return -1;
    if (rpc_write(conn, &pending, sizeof(pending)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDefaultEccMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* defaultMode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDefaultEccMode(device, defaultMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &defaultMode, sizeof(defaultMode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetBoardId(void *conn) {
    nvmlDevice_t device;
    unsigned int* boardId;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetBoardId(device, boardId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &boardId, sizeof(boardId)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMultiGpuBoard(void *conn) {
    nvmlDevice_t device;
    unsigned int* multiGpuBool;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMultiGpuBoard(device, multiGpuBool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &multiGpuBool, sizeof(multiGpuBool)) < 0)
        return -1;
}

int handle_nvmlDeviceGetTotalEccErrors(void *conn) {
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    unsigned long long* eccCounts;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &errorType, sizeof(errorType)) < 0)
        return -1;
    if (rpc_read(conn, &counterType, sizeof(counterType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eccCounts, sizeof(eccCounts)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDetailedEccErrors(void *conn) {
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlEccErrorCounts_t* eccCounts;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &errorType, sizeof(errorType)) < 0)
        return -1;
    if (rpc_read(conn, &counterType, sizeof(counterType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, eccCounts);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eccCounts, sizeof(eccCounts)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemoryErrorCounter(void *conn) {
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlMemoryLocation_t locationType;
    unsigned long long* count;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &errorType, sizeof(errorType)) < 0)
        return -1;
    if (rpc_read(conn, &counterType, sizeof(counterType)) < 0)
        return -1;
    if (rpc_read(conn, &locationType, sizeof(locationType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
}

int handle_nvmlDeviceGetUtilizationRates(void *conn) {
    nvmlDevice_t device;
    nvmlUtilization_t* utilization;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetUtilizationRates(device, utilization);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &utilization, sizeof(utilization)) < 0)
        return -1;
}

int handle_nvmlDeviceGetEncoderUtilization(void *conn) {
    nvmlDevice_t device;
    unsigned int* utilization;
    unsigned int* samplingPeriodUs;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &utilization, sizeof(utilization)) < 0)
        return -1;
    if (rpc_write(conn, &samplingPeriodUs, sizeof(samplingPeriodUs)) < 0)
        return -1;
}

int handle_nvmlDeviceGetEncoderCapacity(void *conn) {
    nvmlDevice_t device;
    nvmlEncoderType_t encoderQueryType;
    unsigned int* encoderCapacity;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &encoderQueryType, sizeof(encoderQueryType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &encoderCapacity, sizeof(encoderCapacity)) < 0)
        return -1;
}

int handle_nvmlDeviceGetEncoderStats(void *conn) {
    nvmlDevice_t device;
    unsigned int* sessionCount;
    unsigned int* averageFps;
    unsigned int* averageLatency;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetEncoderStats(device, sessionCount, averageFps, averageLatency);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    if (rpc_write(conn, &averageFps, sizeof(averageFps)) < 0)
        return -1;
    if (rpc_write(conn, &averageLatency, sizeof(averageLatency)) < 0)
        return -1;
}

int handle_nvmlDeviceGetEncoderSessions(void *conn) {
    nvmlDevice_t device;
    unsigned int* sessionCount;
    nvmlEncoderSessionInfo_t* sessionInfos;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfos, *sessionCount * sizeof(sessionInfos)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDecoderUtilization(void *conn) {
    nvmlDevice_t device;
    unsigned int* utilization;
    unsigned int* samplingPeriodUs;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &utilization, sizeof(utilization)) < 0)
        return -1;
    if (rpc_write(conn, &samplingPeriodUs, sizeof(samplingPeriodUs)) < 0)
        return -1;
}

int handle_nvmlDeviceGetFBCStats(void *conn) {
    nvmlDevice_t device;
    nvmlFBCStats_t* fbcStats;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetFBCStats(device, fbcStats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbcStats, sizeof(fbcStats)) < 0)
        return -1;
}

int handle_nvmlDeviceGetFBCSessions(void *conn) {
    nvmlDevice_t device;
    unsigned int* sessionCount;
    nvmlFBCSessionInfo_t* sessionInfo;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfo, *sessionCount * sizeof(sessionInfo)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDriverModel(void *conn) {
    nvmlDevice_t device;
    nvmlDriverModel_t* current;
    nvmlDriverModel_t* pending;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDriverModel(device, current, pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(current)) < 0)
        return -1;
    if (rpc_write(conn, &pending, sizeof(pending)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVbiosVersion(void *conn) {
    nvmlDevice_t device;
    size_t version_len;
    char* version;
    unsigned int length;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVbiosVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version_len, sizeof(version_len)) < 0)
        return -1;
    if (rpc_write(conn, &version, version_len) < 0)
        return -1;
}

int handle_nvmlDeviceGetBridgeChipInfo(void *conn) {
    nvmlDevice_t device;
    nvmlBridgeChipHierarchy_t* bridgeHierarchy;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bridgeHierarchy, sizeof(bridgeHierarchy)) < 0)
        return -1;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(void *conn) {
    nvmlDevice_t device;
    unsigned int* infoCount;
    nvmlProcessInfo_t* infos;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &infoCount, sizeof(infoCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetComputeRunningProcesses_v3(device, infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &infoCount, sizeof(infoCount)) < 0)
        return -1;
    if (rpc_write(conn, infos, *infoCount * sizeof(infos)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(void *conn) {
    nvmlDevice_t device;
    unsigned int* infoCount;
    nvmlProcessInfo_t* infos;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &infoCount, sizeof(infoCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGraphicsRunningProcesses_v3(device, infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &infoCount, sizeof(infoCount)) < 0)
        return -1;
    if (rpc_write(conn, infos, *infoCount * sizeof(infos)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(void *conn) {
    nvmlDevice_t device;
    unsigned int* infoCount;
    nvmlProcessInfo_t* infos;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &infoCount, sizeof(infoCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &infoCount, sizeof(infoCount)) < 0)
        return -1;
    if (rpc_write(conn, infos, *infoCount * sizeof(infos)) < 0)
        return -1;
}

int handle_nvmlDeviceOnSameBoard(void *conn) {
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    int* onSameBoard;
    if (rpc_read(conn, &device1, sizeof(device1)) < 0)
        return -1;
    if (rpc_read(conn, &device2, sizeof(device2)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceOnSameBoard(device1, device2, onSameBoard);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &onSameBoard, sizeof(onSameBoard)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAPIRestriction(void *conn) {
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t* isRestricted;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &apiType, sizeof(apiType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAPIRestriction(device, apiType, isRestricted);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isRestricted, sizeof(isRestricted)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSamples(void *conn) {
    nvmlDevice_t device;
    nvmlSamplingType_t type;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t* sampleValType;
    unsigned int* sampleCount;
    nvmlSample_t* samples;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0)
        return -1;
    if (rpc_read(conn, &sampleCount, sizeof(sampleCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sampleValType, sizeof(sampleValType)) < 0)
        return -1;
    if (rpc_write(conn, &sampleCount, sizeof(sampleCount)) < 0)
        return -1;
    if (rpc_write(conn, samples, *sampleCount * sizeof(samples)) < 0)
        return -1;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(void *conn) {
    nvmlDevice_t device;
    nvmlBAR1Memory_t* bar1Memory;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bar1Memory, sizeof(bar1Memory)) < 0)
        return -1;
}

int handle_nvmlDeviceGetViolationStatus(void *conn) {
    nvmlDevice_t device;
    nvmlPerfPolicyType_t perfPolicyType;
    nvmlViolationTime_t* violTime;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &perfPolicyType, sizeof(perfPolicyType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetViolationStatus(device, perfPolicyType, violTime);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &violTime, sizeof(violTime)) < 0)
        return -1;
}

int handle_nvmlDeviceGetIrqNum(void *conn) {
    nvmlDevice_t device;
    unsigned int* irqNum;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetIrqNum(device, irqNum);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &irqNum, sizeof(irqNum)) < 0)
        return -1;
}

int handle_nvmlDeviceGetNumGpuCores(void *conn) {
    nvmlDevice_t device;
    unsigned int* numCores;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNumGpuCores(device, numCores);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numCores, sizeof(numCores)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPowerSource(void *conn) {
    nvmlDevice_t device;
    nvmlPowerSource_t* powerSource;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPowerSource(device, powerSource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &powerSource, sizeof(powerSource)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemoryBusWidth(void *conn) {
    nvmlDevice_t device;
    unsigned int* busWidth;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemoryBusWidth(device, busWidth);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &busWidth, sizeof(busWidth)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPcieLinkMaxSpeed(void *conn) {
    nvmlDevice_t device;
    unsigned int* maxSpeed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPcieLinkMaxSpeed(device, maxSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxSpeed, sizeof(maxSpeed)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPcieSpeed(void *conn) {
    nvmlDevice_t device;
    unsigned int* pcieSpeed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPcieSpeed(device, pcieSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pcieSpeed, sizeof(pcieSpeed)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAdaptiveClockInfoStatus(void *conn) {
    nvmlDevice_t device;
    unsigned int* adaptiveClockStatus;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAdaptiveClockInfoStatus(device, adaptiveClockStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &adaptiveClockStatus, sizeof(adaptiveClockStatus)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAccountingMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAccountingMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAccountingStats(void *conn) {
    nvmlDevice_t device;
    unsigned int pid;
    nvmlAccountingStats_t* stats;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &pid, sizeof(pid)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAccountingStats(device, pid, stats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &stats, sizeof(stats)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAccountingPids(void *conn) {
    nvmlDevice_t device;
    unsigned int* count;
    unsigned int* pids;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAccountingPids(device, count, pids);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, pids, *count * sizeof(pids)) < 0)
        return -1;
}

int handle_nvmlDeviceGetAccountingBufferSize(void *conn) {
    nvmlDevice_t device;
    unsigned int* bufferSize;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetAccountingBufferSize(device, bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
}

int handle_nvmlDeviceGetRetiredPages(void *conn) {
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int* pageCount;
    unsigned long long* addresses;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &cause, sizeof(cause)) < 0)
        return -1;
    if (rpc_read(conn, &pageCount, sizeof(pageCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pageCount, sizeof(pageCount)) < 0)
        return -1;
    if (rpc_write(conn, addresses, *pageCount * sizeof(addresses)) < 0)
        return -1;
}

int handle_nvmlDeviceGetRetiredPages_v2(void *conn) {
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int* pageCount;
    unsigned long long* addresses;
    unsigned long long* timestamps;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &cause, sizeof(cause)) < 0)
        return -1;
    if (rpc_read(conn, &pageCount, sizeof(pageCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses, timestamps);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pageCount, sizeof(pageCount)) < 0)
        return -1;
    if (rpc_write(conn, addresses, *pageCount * sizeof(addresses)) < 0)
        return -1;
    if (rpc_write(conn, timestamps, *pageCount * sizeof(timestamps)) < 0)
        return -1;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t* isPending;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetRetiredPagesPendingStatus(device, isPending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isPending, sizeof(isPending)) < 0)
        return -1;
}

int handle_nvmlDeviceGetRemappedRows(void *conn) {
    nvmlDevice_t device;
    unsigned int* corrRows;
    unsigned int* uncRows;
    unsigned int* isPending;
    unsigned int* failureOccurred;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending, failureOccurred);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &corrRows, sizeof(corrRows)) < 0)
        return -1;
    if (rpc_write(conn, &uncRows, sizeof(uncRows)) < 0)
        return -1;
    if (rpc_write(conn, &isPending, sizeof(isPending)) < 0)
        return -1;
    if (rpc_write(conn, &failureOccurred, sizeof(failureOccurred)) < 0)
        return -1;
}

int handle_nvmlDeviceGetRowRemapperHistogram(void *conn) {
    nvmlDevice_t device;
    nvmlRowRemapperHistogramValues_t* values;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetRowRemapperHistogram(device, values);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &values, sizeof(values)) < 0)
        return -1;
}

int handle_nvmlDeviceGetArchitecture(void *conn) {
    nvmlDevice_t device;
    nvmlDeviceArchitecture_t* arch;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetArchitecture(device, arch);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &arch, sizeof(arch)) < 0)
        return -1;
}

int handle_nvmlUnitSetLedState(void *conn) {
    nvmlUnit_t unit;
    nvmlLedColor_t color;
    if (rpc_read(conn, &unit, sizeof(unit)) < 0)
        return -1;
    if (rpc_read(conn, &color, sizeof(color)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlUnitSetLedState(unit, color);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetPersistenceMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetPersistenceMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetComputeMode(void *conn) {
    nvmlDevice_t device;
    nvmlComputeMode_t mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetComputeMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetEccMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t ecc;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &ecc, sizeof(ecc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetEccMode(device, ecc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceClearEccErrorCounts(void *conn) {
    nvmlDevice_t device;
    nvmlEccCounterType_t counterType;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &counterType, sizeof(counterType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceClearEccErrorCounts(device, counterType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetDriverModel(void *conn) {
    nvmlDevice_t device;
    nvmlDriverModel_t driverModel;
    unsigned int flags;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &driverModel, sizeof(driverModel)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetDriverModel(device, driverModel, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetGpuLockedClocks(void *conn) {
    nvmlDevice_t device;
    unsigned int minGpuClockMHz;
    unsigned int maxGpuClockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &minGpuClockMHz, sizeof(minGpuClockMHz)) < 0)
        return -1;
    if (rpc_read(conn, &maxGpuClockMHz, sizeof(maxGpuClockMHz)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceResetGpuLockedClocks(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceResetGpuLockedClocks(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetMemoryLockedClocks(void *conn) {
    nvmlDevice_t device;
    unsigned int minMemClockMHz;
    unsigned int maxMemClockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &minMemClockMHz, sizeof(minMemClockMHz)) < 0)
        return -1;
    if (rpc_read(conn, &maxMemClockMHz, sizeof(maxMemClockMHz)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceResetMemoryLockedClocks(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceResetMemoryLockedClocks(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetApplicationsClocks(void *conn) {
    nvmlDevice_t device;
    unsigned int memClockMHz;
    unsigned int graphicsClockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &memClockMHz, sizeof(memClockMHz)) < 0)
        return -1;
    if (rpc_read(conn, &graphicsClockMHz, sizeof(graphicsClockMHz)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetClkMonStatus(void *conn) {
    nvmlDevice_t device;
    nvmlClkMonStatus_t* status;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetClkMonStatus(device, status);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &status, sizeof(status)) < 0)
        return -1;
}

int handle_nvmlDeviceSetPowerManagementLimit(void *conn) {
    nvmlDevice_t device;
    unsigned int limit;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetPowerManagementLimit(device, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetGpuOperationMode(void *conn) {
    nvmlDevice_t device;
    nvmlGpuOperationMode_t mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetGpuOperationMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetAPIRestriction(void *conn) {
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &apiType, sizeof(apiType)) < 0)
        return -1;
    if (rpc_read(conn, &isRestricted, sizeof(isRestricted)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetAPIRestriction(device, apiType, isRestricted);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetAccountingMode(void *conn) {
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetAccountingMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceClearAccountingPids(void *conn) {
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceClearAccountingPids(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkState(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    nvmlEnableState_t* isActive;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkState(device, link, isActive);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isActive, sizeof(isActive)) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkVersion(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    unsigned int* version;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkVersion(device, link, version);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version, sizeof(version)) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkCapability(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    nvmlNvLinkCapability_t capability;
    unsigned int* capResult;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &capability, sizeof(capability)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkCapability(device, link, capability, capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(capResult)) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkRemotePciInfo_v2(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    nvmlPciInfo_t* pci;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pci, sizeof(pci)) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkErrorCounter(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    nvmlNvLinkErrorCounter_t counter;
    unsigned long long* counterValue;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &counterValue, sizeof(counterValue)) < 0)
        return -1;
}

int handle_nvmlDeviceResetNvLinkErrorCounters(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceResetNvLinkErrorCounters(device, link);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetNvLinkUtilizationControl(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t* control;
    unsigned int reset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    if (rpc_read(conn, &control, sizeof(control)) < 0)
        return -1;
    if (rpc_read(conn, &reset, sizeof(reset)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control, reset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkUtilizationControl(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t* control;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, control);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &control, sizeof(control)) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkUtilizationCounter(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    unsigned long long* rxcounter;
    unsigned long long* txcounter;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, rxcounter, txcounter);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &rxcounter, sizeof(rxcounter)) < 0)
        return -1;
    if (rpc_write(conn, &txcounter, sizeof(txcounter)) < 0)
        return -1;
}

int handle_nvmlDeviceFreezeNvLinkUtilizationCounter(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlEnableState_t freeze;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    if (rpc_read(conn, &freeze, sizeof(freeze)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceResetNvLinkUtilizationCounter(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    if (rpc_read(conn, &counter, sizeof(counter)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetNvLinkRemoteDeviceType(void *conn) {
    nvmlDevice_t device;
    unsigned int link;
    nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &link, sizeof(link)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, pNvLinkDeviceType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNvLinkDeviceType, sizeof(pNvLinkDeviceType)) < 0)
        return -1;
}

int handle_nvmlEventSetCreate(void *conn) {
    nvmlEventSet_t* set;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlEventSetCreate(set);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &set, sizeof(set)) < 0)
        return -1;
}

int handle_nvmlDeviceRegisterEvents(void *conn) {
    nvmlDevice_t device;
    unsigned long long eventTypes;
    nvmlEventSet_t set;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &eventTypes, sizeof(eventTypes)) < 0)
        return -1;
    if (rpc_read(conn, &set, sizeof(set)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceRegisterEvents(device, eventTypes, set);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetSupportedEventTypes(void *conn) {
    nvmlDevice_t device;
    unsigned long long* eventTypes;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSupportedEventTypes(device, eventTypes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eventTypes, sizeof(eventTypes)) < 0)
        return -1;
}

int handle_nvmlEventSetWait_v2(void *conn) {
    nvmlEventSet_t set;
    nvmlEventData_t* data;
    unsigned int timeoutms;
    if (rpc_read(conn, &set, sizeof(set)) < 0)
        return -1;
    if (rpc_read(conn, &timeoutms, sizeof(timeoutms)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlEventSetWait_v2(set, data, timeoutms);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
}

int handle_nvmlEventSetFree(void *conn) {
    nvmlEventSet_t set;
    if (rpc_read(conn, &set, sizeof(set)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlEventSetFree(set);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceModifyDrainState(void *conn) {
    nvmlPciInfo_t* pciInfo;
    nvmlEnableState_t newState;
    if (rpc_read(conn, &pciInfo, sizeof(pciInfo)) < 0)
        return -1;
    if (rpc_read(conn, &newState, sizeof(newState)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceModifyDrainState(pciInfo, newState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceQueryDrainState(void *conn) {
    nvmlPciInfo_t* pciInfo;
    nvmlEnableState_t* currentState;
    if (rpc_read(conn, &pciInfo, sizeof(pciInfo)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceQueryDrainState(pciInfo, currentState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currentState, sizeof(currentState)) < 0)
        return -1;
}

int handle_nvmlDeviceRemoveGpu_v2(void *conn) {
    nvmlPciInfo_t* pciInfo;
    nvmlDetachGpuState_t gpuState;
    nvmlPcieLinkState_t linkState;
    if (rpc_read(conn, &pciInfo, sizeof(pciInfo)) < 0)
        return -1;
    if (rpc_read(conn, &gpuState, sizeof(gpuState)) < 0)
        return -1;
    if (rpc_read(conn, &linkState, sizeof(linkState)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceRemoveGpu_v2(pciInfo, gpuState, linkState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceDiscoverGpus(void *conn) {
    nvmlPciInfo_t* pciInfo;
    if (rpc_read(conn, &pciInfo, sizeof(pciInfo)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceDiscoverGpus(pciInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetFieldValues(void *conn) {
    nvmlDevice_t device;
    int valuesCount;
    nvmlFieldValue_t* values;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &valuesCount, sizeof(valuesCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetFieldValues(device, valuesCount, values);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, values, valuesCount * sizeof(values)) < 0)
        return -1;
}

int handle_nvmlDeviceClearFieldValues(void *conn) {
    nvmlDevice_t device;
    int valuesCount;
    nvmlFieldValue_t* values;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &valuesCount, sizeof(valuesCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceClearFieldValues(device, valuesCount, values);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, values, valuesCount * sizeof(values)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVirtualizationMode(void *conn) {
    nvmlDevice_t device;
    nvmlGpuVirtualizationMode_t* pVirtualMode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVirtualizationMode(device, pVirtualMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pVirtualMode, sizeof(pVirtualMode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetHostVgpuMode(void *conn) {
    nvmlDevice_t device;
    nvmlHostVgpuMode_t* pHostVgpuMode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHostVgpuMode, sizeof(pHostVgpuMode)) < 0)
        return -1;
}

int handle_nvmlDeviceSetVirtualizationMode(void *conn) {
    nvmlDevice_t device;
    nvmlGpuVirtualizationMode_t virtualMode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &virtualMode, sizeof(virtualMode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetVirtualizationMode(device, virtualMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetGridLicensableFeatures_v4(void *conn) {
    nvmlDevice_t device;
    nvmlGridLicensableFeatures_t* pGridLicensableFeatures;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGridLicensableFeatures_v4(device, pGridLicensableFeatures);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGridLicensableFeatures, sizeof(pGridLicensableFeatures)) < 0)
        return -1;
}

int handle_nvmlDeviceGetProcessUtilization(void *conn) {
    nvmlDevice_t device;
    unsigned int* processSamplesCount;
    nvmlProcessUtilizationSample_t* utilization;
    unsigned long long lastSeenTimeStamp;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &processSamplesCount, sizeof(processSamplesCount)) < 0)
        return -1;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount, lastSeenTimeStamp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &processSamplesCount, sizeof(processSamplesCount)) < 0)
        return -1;
    if (rpc_write(conn, utilization, *processSamplesCount * sizeof(utilization)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGspFirmwareVersion(void *conn) {
    nvmlDevice_t device;
    char* version;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGspFirmwareVersion(device, version);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version, sizeof(version)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGspFirmwareMode(void *conn) {
    nvmlDevice_t device;
    unsigned int* isEnabled;
    unsigned int* defaultMode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGspFirmwareMode(device, isEnabled, defaultMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
    if (rpc_write(conn, &defaultMode, sizeof(defaultMode)) < 0)
        return -1;
}

int handle_nvmlGetVgpuDriverCapabilities(void *conn) {
    nvmlVgpuDriverCapability_t capability;
    unsigned int* capResult;
    if (rpc_read(conn, &capability, sizeof(capability)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGetVgpuDriverCapabilities(capability, capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(capResult)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuCapabilities(void *conn) {
    nvmlDevice_t device;
    nvmlDeviceVgpuCapability_t capability;
    unsigned int* capResult;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &capability, sizeof(capability)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuCapabilities(device, capability, capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(capResult)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSupportedVgpus(void *conn) {
    nvmlDevice_t device;
    unsigned int* vgpuCount;
    nvmlVgpuTypeId_t* vgpuTypeIds;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &vgpuCount, sizeof(vgpuCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuCount, sizeof(vgpuCount)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeIds, *vgpuCount * sizeof(vgpuTypeIds)) < 0)
        return -1;
}

int handle_nvmlDeviceGetCreatableVgpus(void *conn) {
    nvmlDevice_t device;
    unsigned int* vgpuCount;
    nvmlVgpuTypeId_t* vgpuTypeIds;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &vgpuCount, sizeof(vgpuCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuCount, sizeof(vgpuCount)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeIds, *vgpuCount * sizeof(vgpuTypeIds)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetClass(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* size;
    char* vgpuTypeClass;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeClass, *size * sizeof(vgpuTypeClass)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetName(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* size;
    char* vgpuTypeName;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeName, *size * sizeof(vgpuTypeName)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* gpuInstanceProfileId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstanceProfileId, sizeof(gpuInstanceProfileId)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetDeviceID(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long* deviceID;
    unsigned long long* subsystemID;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceID, sizeof(deviceID)) < 0)
        return -1;
    if (rpc_write(conn, &subsystemID, sizeof(subsystemID)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetFramebufferSize(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long* fbSize;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbSize, sizeof(fbSize)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetNumDisplayHeads(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* numDisplayHeads;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numDisplayHeads, sizeof(numDisplayHeads)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetResolution(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int displayIndex;
    unsigned int* xdim;
    unsigned int* ydim;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    if (rpc_read(conn, &displayIndex, sizeof(displayIndex)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &xdim, sizeof(xdim)) < 0)
        return -1;
    if (rpc_write(conn, &ydim, sizeof(ydim)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetLicense(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;
    char* vgpuTypeLicenseString;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeLicenseString, size * sizeof(vgpuTypeLicenseString)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetFrameRateLimit(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* frameRateLimit;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &frameRateLimit, sizeof(frameRateLimit)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetMaxInstances(void *conn) {
    nvmlDevice_t device;
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* vgpuInstanceCount;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuInstanceCount, sizeof(vgpuInstanceCount)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetMaxInstancesPerVm(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int* vgpuInstanceCountPerVm;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuInstanceCountPerVm, sizeof(vgpuInstanceCountPerVm)) < 0)
        return -1;
}

int handle_nvmlDeviceGetActiveVgpus(void *conn) {
    nvmlDevice_t device;
    unsigned int* vgpuCount;
    nvmlVgpuInstance_t* vgpuInstances;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &vgpuCount, sizeof(vgpuCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuCount, sizeof(vgpuCount)) < 0)
        return -1;
    if (rpc_write(conn, vgpuInstances, *vgpuCount * sizeof(vgpuInstances)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetVmID(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    char* vmId;
    nvmlVgpuVmIdType_t* vmIdType;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, vmId, size * sizeof(vmId)) < 0)
        return -1;
    if (rpc_write(conn, &vmIdType, sizeof(vmIdType)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetUUID(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    char* uuid;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, size * sizeof(uuid)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetVmDriverVersion(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int length;
    char* version;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetFbUsage(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned long long* fbUsage;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbUsage, sizeof(fbUsage)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetLicenseStatus(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* licensed;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &licensed, sizeof(licensed)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetType(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuTypeId_t* vgpuTypeId;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetFrameRateLimit(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* frameRateLimit;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &frameRateLimit, sizeof(frameRateLimit)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetEccMode(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t* eccMode;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eccMode, sizeof(eccMode)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetEncoderCapacity(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* encoderCapacity;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &encoderCapacity, sizeof(encoderCapacity)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceSetEncoderCapacity(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int encoderCapacity;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &encoderCapacity, sizeof(encoderCapacity)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetEncoderStats(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* sessionCount;
    unsigned int* averageFps;
    unsigned int* averageLatency;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps, averageLatency);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    if (rpc_write(conn, &averageFps, sizeof(averageFps)) < 0)
        return -1;
    if (rpc_write(conn, &averageLatency, sizeof(averageLatency)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetEncoderSessions(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* sessionCount;
    nvmlEncoderSessionInfo_t* sessionInfo;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfo, *sessionCount * sizeof(sessionInfo)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetFBCStats(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    nvmlFBCStats_t* fbcStats;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbcStats, sizeof(fbcStats)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetFBCSessions(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* sessionCount;
    nvmlFBCSessionInfo_t* sessionInfo;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(sessionCount)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfo, *sessionCount * sizeof(sessionInfo)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetGpuInstanceId(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* gpuInstanceId;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstanceId, sizeof(gpuInstanceId)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetGpuPciId(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* length;
    char* vgpuPciId;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &length, sizeof(length)) < 0)
        return -1;
    if (rpc_write(conn, vgpuPciId, *length * sizeof(vgpuPciId)) < 0)
        return -1;
}

int handle_nvmlVgpuTypeGetCapabilities(void *conn) {
    nvmlVgpuTypeId_t vgpuTypeId;
    nvmlVgpuCapability_t capability;
    unsigned int* capResult;
    if (rpc_read(conn, &vgpuTypeId, sizeof(vgpuTypeId)) < 0)
        return -1;
    if (rpc_read(conn, &capability, sizeof(capability)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(capResult)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetMetadata(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* bufferSize;
    nvmlVgpuMetadata_t* vgpuMetadata;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
    if (rpc_write(conn, vgpuMetadata, *bufferSize * sizeof(vgpuMetadata)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuMetadata(void *conn) {
    nvmlDevice_t device;
    unsigned int* bufferSize;
    nvmlVgpuPgpuMetadata_t* pgpuMetadata;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
    if (rpc_write(conn, pgpuMetadata, *bufferSize * sizeof(pgpuMetadata)) < 0)
        return -1;
}

int handle_nvmlGetVgpuCompatibility(void *conn) {
    nvmlVgpuMetadata_t* vgpuMetadata;
    nvmlVgpuPgpuMetadata_t* pgpuMetadata;
    nvmlVgpuPgpuCompatibility_t* compatibilityInfo;
    if (rpc_read(conn, &vgpuMetadata, sizeof(vgpuMetadata)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuMetadata, sizeof(vgpuMetadata)) < 0)
        return -1;
    if (rpc_write(conn, &pgpuMetadata, sizeof(pgpuMetadata)) < 0)
        return -1;
    if (rpc_write(conn, &compatibilityInfo, sizeof(compatibilityInfo)) < 0)
        return -1;
}

int handle_nvmlDeviceGetPgpuMetadataString(void *conn) {
    nvmlDevice_t device;
    unsigned int* bufferSize;
    char* pgpuMetadata;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(bufferSize)) < 0)
        return -1;
    if (rpc_write(conn, pgpuMetadata, *bufferSize * sizeof(pgpuMetadata)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuSchedulerLog(void *conn) {
    nvmlDevice_t device;
    nvmlVgpuSchedulerLog_t* pSchedulerLog;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuSchedulerLog(device, pSchedulerLog);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSchedulerLog, sizeof(pSchedulerLog)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuSchedulerState(void *conn) {
    nvmlDevice_t device;
    nvmlVgpuSchedulerGetState_t* pSchedulerState;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuSchedulerState(device, pSchedulerState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSchedulerState, sizeof(pSchedulerState)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuSchedulerCapabilities(void *conn) {
    nvmlDevice_t device;
    nvmlVgpuSchedulerCapabilities_t* pCapabilities;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuSchedulerCapabilities(device, pCapabilities);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCapabilities, sizeof(pCapabilities)) < 0)
        return -1;
}

int handle_nvmlGetVgpuVersion(void *conn) {
    nvmlVgpuVersion_t* supported;
    nvmlVgpuVersion_t* current;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGetVgpuVersion(supported, current);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &supported, sizeof(supported)) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(current)) < 0)
        return -1;
}

int handle_nvmlSetVgpuVersion(void *conn) {
    nvmlVgpuVersion_t* vgpuVersion;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlSetVgpuVersion(vgpuVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuVersion, sizeof(vgpuVersion)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuUtilization(void *conn) {
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t* sampleValType;
    unsigned int* vgpuInstanceSamplesCount;
    nvmlVgpuInstanceUtilizationSample_t* utilizationSamples;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0)
        return -1;
    if (rpc_read(conn, &sampleValType, sizeof(sampleValType)) < 0)
        return -1;
    if (rpc_read(conn, &vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sampleValType, sizeof(sampleValType)) < 0)
        return -1;
    if (rpc_write(conn, &vgpuInstanceSamplesCount, sizeof(vgpuInstanceSamplesCount)) < 0)
        return -1;
    if (rpc_write(conn, utilizationSamples, *vgpuInstanceSamplesCount * sizeof(utilizationSamples)) < 0)
        return -1;
}

int handle_nvmlDeviceGetVgpuProcessUtilization(void *conn) {
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    unsigned int* vgpuProcessSamplesCount;
    nvmlVgpuProcessUtilizationSample_t* utilizationSamples;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(lastSeenTimeStamp)) < 0)
        return -1;
    if (rpc_read(conn, &vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuProcessSamplesCount, sizeof(vgpuProcessSamplesCount)) < 0)
        return -1;
    if (rpc_write(conn, utilizationSamples, *vgpuProcessSamplesCount * sizeof(utilizationSamples)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetAccountingMode(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t* mode;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetAccountingPids(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int* count;
    unsigned int* pids;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, pids, *count * sizeof(pids)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetAccountingStats(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int pid;
    nvmlAccountingStats_t* stats;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &pid, sizeof(pid)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &stats, sizeof(stats)) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceClearAccountingPids(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceClearAccountingPids(vgpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlVgpuInstanceGetLicenseInfo_v2(void *conn) {
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuLicenseInfo_t* licenseInfo;
    if (rpc_read(conn, &vgpuInstance, sizeof(vgpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, licenseInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &licenseInfo, sizeof(licenseInfo)) < 0)
        return -1;
}

int handle_nvmlGetExcludedDeviceCount(void *conn) {
    unsigned int* deviceCount;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGetExcludedDeviceCount(deviceCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceCount, sizeof(deviceCount)) < 0)
        return -1;
}

int handle_nvmlGetExcludedDeviceInfoByIndex(void *conn) {
    unsigned int index;
    nvmlExcludedDeviceInfo_t* info;
    if (rpc_read(conn, &index, sizeof(index)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGetExcludedDeviceInfoByIndex(index, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlDeviceSetMigMode(void *conn) {
    nvmlDevice_t device;
    unsigned int mode;
    nvmlReturn_t* activationStatus;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetMigMode(device, mode, activationStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &activationStatus, sizeof(activationStatus)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMigMode(void *conn) {
    nvmlDevice_t device;
    unsigned int* currentMode;
    unsigned int* pendingMode;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMigMode(device, currentMode, pendingMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currentMode, sizeof(currentMode)) < 0)
        return -1;
    if (rpc_write(conn, &pendingMode, sizeof(pendingMode)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfo(void *conn) {
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_t* info;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profile, sizeof(profile)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfoV(void *conn) {
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_v2_t* info;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profile, sizeof(profile)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstancePossiblePlacements_v2(void *conn) {
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int* count;
    nvmlGpuInstancePlacement_t* placements;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, placements, *count * sizeof(placements)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstanceRemainingCapacity(void *conn) {
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int* count;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
}

int handle_nvmlDeviceCreateGpuInstance(void *conn) {
    nvmlDevice_t device;
    unsigned int profileId;
    nvmlGpuInstance_t* gpuInstance;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
}

int handle_nvmlDeviceCreateGpuInstanceWithPlacement(void *conn) {
    nvmlDevice_t device;
    unsigned int profileId;
    const nvmlGpuInstancePlacement_t* placement;
    nvmlGpuInstance_t* gpuInstance;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement, gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &placement, sizeof(placement)) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceDestroy(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceDestroy(gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstances(void *conn) {
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int* count;
    nvmlGpuInstance_t* gpuInstances;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, gpuInstances, *count * sizeof(gpuInstances)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstanceById(void *conn) {
    nvmlDevice_t device;
    unsigned int id;
    nvmlGpuInstance_t* gpuInstance;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &id, sizeof(id)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstanceById(device, id, gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetInfo(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    nvmlGpuInstanceInfo_t* info;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetInfo(gpuInstance, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfo(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_t* info;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profile, sizeof(profile)) < 0)
        return -1;
    if (rpc_read(conn, &engProfile, sizeof(engProfile)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_v2_t* info;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profile, sizeof(profile)) < 0)
        return -1;
    if (rpc_read(conn, &engProfile, sizeof(engProfile)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile, engProfile, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int* count;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int* count;
    nvmlComputeInstancePlacement_t* placements;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placements, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, placements, *count * sizeof(placements)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceCreateComputeInstance(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    nvmlComputeInstance_t* computeInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &computeInstance, sizeof(computeInstance)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceCreateComputeInstanceWithPlacement(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    const nvmlComputeInstancePlacement_t* placement;
    nvmlComputeInstance_t* computeInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    if (rpc_read(conn, &placement, sizeof(placement)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, placement, computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &computeInstance, sizeof(computeInstance)) < 0)
        return -1;
}

int handle_nvmlComputeInstanceDestroy(void *conn) {
    nvmlComputeInstance_t computeInstance;
    if (rpc_read(conn, &computeInstance, sizeof(computeInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlComputeInstanceDestroy(computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetComputeInstances(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int* count;
    nvmlComputeInstance_t* computeInstances;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &profileId, sizeof(profileId)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_write(conn, computeInstances, *count * sizeof(computeInstances)) < 0)
        return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceById(void *conn) {
    nvmlGpuInstance_t gpuInstance;
    unsigned int id;
    nvmlComputeInstance_t* computeInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(gpuInstance)) < 0)
        return -1;
    if (rpc_read(conn, &id, sizeof(id)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &computeInstance, sizeof(computeInstance)) < 0)
        return -1;
}

int handle_nvmlComputeInstanceGetInfo_v2(void *conn) {
    nvmlComputeInstance_t computeInstance;
    nvmlComputeInstanceInfo_t* info;
    if (rpc_read(conn, &computeInstance, sizeof(computeInstance)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlComputeInstanceGetInfo_v2(computeInstance, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_nvmlDeviceIsMigDeviceHandle(void *conn) {
    nvmlDevice_t device;
    unsigned int* isMigDevice;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceIsMigDeviceHandle(device, isMigDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isMigDevice, sizeof(isMigDevice)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuInstanceId(void *conn) {
    nvmlDevice_t device;
    unsigned int* id;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuInstanceId(device, id);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &id, sizeof(id)) < 0)
        return -1;
}

int handle_nvmlDeviceGetComputeInstanceId(void *conn) {
    nvmlDevice_t device;
    unsigned int* id;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetComputeInstanceId(device, id);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &id, sizeof(id)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMaxMigDeviceCount(void *conn) {
    nvmlDevice_t device;
    unsigned int* count;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMaxMigDeviceCount(device, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMigDeviceHandleByIndex(void *conn) {
    nvmlDevice_t device;
    unsigned int index;
    nvmlDevice_t* migDevice;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &index, sizeof(index)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &migDevice, sizeof(migDevice)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDeviceHandleFromMigDeviceHandle(void *conn) {
    nvmlDevice_t migDevice;
    nvmlDevice_t* device;
    if (rpc_read(conn, &migDevice, sizeof(migDevice)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_nvmlDeviceGetBusType(void *conn) {
    nvmlDevice_t device;
    nvmlBusType_t* type;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetBusType(device, type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &type, sizeof(type)) < 0)
        return -1;
}

int handle_nvmlDeviceGetDynamicPstatesInfo(void *conn) {
    nvmlDevice_t device;
    nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetDynamicPstatesInfo(device, pDynamicPstatesInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDynamicPstatesInfo, sizeof(pDynamicPstatesInfo)) < 0)
        return -1;
}

int handle_nvmlDeviceSetFanSpeed_v2(void *conn) {
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int speed;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &fan, sizeof(fan)) < 0)
        return -1;
    if (rpc_read(conn, &speed, sizeof(speed)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetFanSpeed_v2(device, fan, speed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpcClkVfOffset(void *conn) {
    nvmlDevice_t device;
    int* offset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpcClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &offset, sizeof(offset)) < 0)
        return -1;
}

int handle_nvmlDeviceSetGpcClkVfOffset(void *conn) {
    nvmlDevice_t device;
    int offset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetGpcClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &offset, sizeof(offset)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemClkVfOffset(void *conn) {
    nvmlDevice_t device;
    int* offset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &offset, sizeof(offset)) < 0)
        return -1;
}

int handle_nvmlDeviceSetMemClkVfOffset(void *conn) {
    nvmlDevice_t device;
    int offset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetMemClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceGetMinMaxClockOfPState(void *conn) {
    nvmlDevice_t device;
    nvmlClockType_t type;
    nvmlPstates_t pstate;
    unsigned int* minClockMHz;
    unsigned int* maxClockMHz;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    if (rpc_read(conn, &pstate, sizeof(pstate)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, minClockMHz, maxClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minClockMHz, sizeof(minClockMHz)) < 0)
        return -1;
    if (rpc_write(conn, &maxClockMHz, sizeof(maxClockMHz)) < 0)
        return -1;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(void *conn) {
    nvmlDevice_t device;
    unsigned int size;
    nvmlPstates_t* pstates;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetSupportedPerformanceStates(device, pstates, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, pstates, size * sizeof(pstates)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(void *conn) {
    nvmlDevice_t device;
    int* minOffset;
    int* maxOffset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpcClkMinMaxVfOffset(device, minOffset, maxOffset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minOffset, sizeof(minOffset)) < 0)
        return -1;
    if (rpc_write(conn, &maxOffset, sizeof(maxOffset)) < 0)
        return -1;
}

int handle_nvmlDeviceGetMemClkMinMaxVfOffset(void *conn) {
    nvmlDevice_t device;
    int* minOffset;
    int* maxOffset;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minOffset, sizeof(minOffset)) < 0)
        return -1;
    if (rpc_write(conn, &maxOffset, sizeof(maxOffset)) < 0)
        return -1;
}

int handle_nvmlDeviceGetGpuFabricInfo(void *conn) {
    nvmlDevice_t device;
    nvmlGpuFabricInfo_t* gpuFabricInfo;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceGetGpuFabricInfo(device, gpuFabricInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuFabricInfo, sizeof(gpuFabricInfo)) < 0)
        return -1;
}

int handle_nvmlGpmMetricsGet(void *conn) {
    nvmlGpmMetricsGet_t* metricsGet;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpmMetricsGet(metricsGet);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &metricsGet, sizeof(metricsGet)) < 0)
        return -1;
}

int handle_nvmlGpmSampleFree(void *conn) {
    nvmlGpmSample_t gpmSample;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpmSampleFree(gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpmSample, sizeof(gpmSample)) < 0)
        return -1;
}

int handle_nvmlGpmSampleAlloc(void *conn) {
    nvmlGpmSample_t* gpmSample;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpmSampleAlloc(gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpmSample, sizeof(gpmSample)) < 0)
        return -1;
}

int handle_nvmlGpmSampleGet(void *conn) {
    nvmlDevice_t device;
    nvmlGpmSample_t gpmSample;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpmSampleGet(device, gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;
}

int handle_nvmlGpmMigSampleGet(void *conn) {
    nvmlDevice_t device;
    unsigned int gpuInstanceId;
    nvmlGpmSample_t gpmSample;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &gpuInstanceId, sizeof(gpuInstanceId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;
}

int handle_nvmlGpmQueryDeviceSupport(void *conn) {
    nvmlDevice_t device;
    nvmlGpmSupport_t* gpmSupport;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlGpmQueryDeviceSupport(device, gpmSupport);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpmSupport, sizeof(gpmSupport)) < 0)
        return -1;
}

int handle_nvmlDeviceCcuGetStreamState(void *conn) {
    nvmlDevice_t device;
    unsigned int* state;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceCcuGetStreamState(device, state);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &state, sizeof(state)) < 0)
        return -1;
}

int handle_nvmlDeviceCcuSetStreamState(void *conn) {
    nvmlDevice_t device;
    unsigned int state;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &state, sizeof(state)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceCcuSetStreamState(device, state);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold(void *conn) {
    nvmlDevice_t device;
    nvmlNvLinkPowerThres_t* info;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t return_value = nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(info)) < 0)
        return -1;
}

int handle_cuInit(void *conn) {
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuInit(Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDriverGetVersion(void *conn) {
    int* driverVersion;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDriverGetVersion(driverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &driverVersion, sizeof(driverVersion)) < 0)
        return -1;
}

int handle_cuDeviceGet(void *conn) {
    CUdevice* device;
    int ordinal;
    if (rpc_read(conn, &ordinal, sizeof(ordinal)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGet(device, ordinal);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_cuDeviceGetCount(void *conn) {
    int* count;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetCount(count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
}

int handle_cuDeviceGetName(void *conn) {
    int len;
    char* name;
    CUdevice dev;
    if (rpc_read(conn, &len, sizeof(len)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetName(name, len, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, name, len * sizeof(name)) < 0)
        return -1;
}

int handle_cuDeviceGetUuid(void *conn) {
    CUuuid* uuid;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetUuid(uuid, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, 16) < 0)
        return -1;
}

int handle_cuDeviceGetUuid_v2(void *conn) {
    CUuuid* uuid;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetUuid_v2(uuid, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, 16) < 0)
        return -1;
}

int handle_cuDeviceGetLuid(void *conn) {
    size_t luid_len;
    char* luid;
    unsigned int* deviceNodeMask;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetLuid(luid, deviceNodeMask, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &luid_len, sizeof(luid_len)) < 0)
        return -1;
    if (rpc_write(conn, &luid, luid_len) < 0)
        return -1;
    if (rpc_write(conn, &deviceNodeMask, sizeof(deviceNodeMask)) < 0)
        return -1;
}

int handle_cuDeviceTotalMem_v2(void *conn) {
    size_t* bytes;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceTotalMem_v2(bytes, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
}

int handle_cuDeviceGetTexture1DLinearMaxWidth(void *conn) {
    size_t* maxWidthInElements;
    CUarray_format format;
    unsigned numChannels;
    CUdevice dev;
    if (rpc_read(conn, &format, sizeof(format)) < 0)
        return -1;
    if (rpc_read(conn, &numChannels, sizeof(numChannels)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxWidthInElements, sizeof(maxWidthInElements)) < 0)
        return -1;
}

int handle_cuDeviceGetAttribute(void *conn) {
    int* pi;
    CUdevice_attribute attrib;
    CUdevice dev;
    if (rpc_read(conn, &attrib, sizeof(attrib)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetAttribute(pi, attrib, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(pi)) < 0)
        return -1;
}

int handle_cuDeviceSetMemPool(void *conn) {
    CUdevice dev;
    CUmemoryPool pool;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceSetMemPool(dev, pool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDeviceGetMemPool(void *conn) {
    CUmemoryPool* pool;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetMemPool(pool, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool, sizeof(pool)) < 0)
        return -1;
}

int handle_cuDeviceGetDefaultMemPool(void *conn) {
    CUmemoryPool* pool_out;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetDefaultMemPool(pool_out, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool_out, sizeof(pool_out)) < 0)
        return -1;
}

int handle_cuDeviceGetExecAffinitySupport(void *conn) {
    int* pi;
    CUexecAffinityType type;
    CUdevice dev;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetExecAffinitySupport(pi, type, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(pi)) < 0)
        return -1;
}

int handle_cuFlushGPUDirectRDMAWrites(void *conn) {
    CUflushGPUDirectRDMAWritesTarget target;
    CUflushGPUDirectRDMAWritesScope scope;
    if (rpc_read(conn, &target, sizeof(target)) < 0)
        return -1;
    if (rpc_read(conn, &scope, sizeof(scope)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFlushGPUDirectRDMAWrites(target, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDeviceGetProperties(void *conn) {
    CUdevprop* prop;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetProperties(prop, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(prop)) < 0)
        return -1;
}

int handle_cuDeviceComputeCapability(void *conn) {
    int* major;
    int* minor;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceComputeCapability(major, minor, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &major, sizeof(major)) < 0)
        return -1;
    if (rpc_write(conn, &minor, sizeof(minor)) < 0)
        return -1;
}

int handle_cuDevicePrimaryCtxRetain(void *conn) {
    CUcontext* pctx;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDevicePrimaryCtxRetain(pctx, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
}

int handle_cuDevicePrimaryCtxRelease_v2(void *conn) {
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDevicePrimaryCtxRelease_v2(dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDevicePrimaryCtxSetFlags_v2(void *conn) {
    CUdevice dev;
    unsigned int flags;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDevicePrimaryCtxSetFlags_v2(dev, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDevicePrimaryCtxGetState(void *conn) {
    CUdevice dev;
    unsigned int* flags;
    int* active;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDevicePrimaryCtxGetState(dev, flags, active);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_write(conn, &active, sizeof(active)) < 0)
        return -1;
}

int handle_cuDevicePrimaryCtxReset_v2(void *conn) {
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDevicePrimaryCtxReset_v2(dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxCreate_v2(void *conn) {
    CUcontext* pctx;
    unsigned int flags;
    CUdevice dev;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxCreate_v2(pctx, flags, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
}

int handle_cuCtxCreate_v3(void *conn) {
    CUcontext* pctx;
    int numParams;
    CUexecAffinityParam* paramsArray;
    unsigned int flags;
    CUdevice dev;
    if (rpc_read(conn, &numParams, sizeof(numParams)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
    if (rpc_write(conn, paramsArray, numParams * sizeof(paramsArray)) < 0)
        return -1;
}

int handle_cuCtxDestroy_v2(void *conn) {
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxDestroy_v2(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxPushCurrent_v2(void *conn) {
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxPushCurrent_v2(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxPopCurrent_v2(void *conn) {
    CUcontext* pctx;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxPopCurrent_v2(pctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
}

int handle_cuCtxSetCurrent(void *conn) {
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxSetCurrent(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxGetCurrent(void *conn) {
    CUcontext* pctx;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetCurrent(pctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
}

int handle_cuCtxGetDevice(void *conn) {
    CUdevice* device;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetDevice(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_cuCtxGetFlags(void *conn) {
    unsigned int* flags;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetFlags(flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cuCtxGetId(void *conn) {
    CUcontext ctx;
    unsigned long long* ctxId;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetId(ctx, ctxId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ctxId, sizeof(ctxId)) < 0)
        return -1;
}

int handle_cuCtxSynchronize(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxSynchronize();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxSetLimit(void *conn) {
    CUlimit limit;
    size_t value;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxGetLimit(void *conn) {
    size_t* pvalue;
    CUlimit limit;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetLimit(pvalue, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pvalue, sizeof(pvalue)) < 0)
        return -1;
}

int handle_cuCtxGetCacheConfig(void *conn) {
    CUfunc_cache* pconfig;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetCacheConfig(pconfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pconfig, sizeof(pconfig)) < 0)
        return -1;
}

int handle_cuCtxSetCacheConfig(void *conn) {
    CUfunc_cache config;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxSetCacheConfig(config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxGetSharedMemConfig(void *conn) {
    CUsharedconfig* pConfig;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetSharedMemConfig(pConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pConfig, sizeof(pConfig)) < 0)
        return -1;
}

int handle_cuCtxSetSharedMemConfig(void *conn) {
    CUsharedconfig config;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxSetSharedMemConfig(config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxGetApiVersion(void *conn) {
    CUcontext ctx;
    unsigned int* version;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetApiVersion(ctx, version);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version, sizeof(version)) < 0)
        return -1;
}

int handle_cuCtxGetStreamPriorityRange(void *conn) {
    int* leastPriority;
    int* greatestPriority;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetStreamPriorityRange(leastPriority, greatestPriority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &leastPriority, sizeof(leastPriority)) < 0)
        return -1;
    if (rpc_write(conn, &greatestPriority, sizeof(greatestPriority)) < 0)
        return -1;
}

int handle_cuCtxResetPersistingL2Cache(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxResetPersistingL2Cache();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxGetExecAffinity(void *conn) {
    CUexecAffinityParam* pExecAffinity;
    CUexecAffinityType type;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxGetExecAffinity(pExecAffinity, type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pExecAffinity, sizeof(pExecAffinity)) < 0)
        return -1;
}

int handle_cuCtxAttach(void *conn) {
    CUcontext* pctx;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxAttach(pctx, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
}

int handle_cuCtxDetach(void *conn) {
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxDetach(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuModuleLoad(void *conn) {
    CUmodule* module;
    size_t fname_len;
    const char* fname;
    if (rpc_read(conn, &fname_len, sizeof(fname_len)) < 0)
        return -1;
    if (rpc_read(conn, &fname, fname_len) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleLoad(module, fname);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &module, sizeof(module)) < 0)
        return -1;
}

int handle_cuModuleLoadDataEx(void *conn) {
    CUmodule* module;
    size_t image_len;
    const void* image;
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    if (rpc_read(conn, &image_len, sizeof(image_len)) < 0)
        return -1;
    if (rpc_read(conn, &image, image_len) < 0)
        return -1;
    if (rpc_read(conn, &numOptions, sizeof(numOptions)) < 0)
        return -1;
    if (rpc_read(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_read(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleLoadDataEx(module, image, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &module, sizeof(module)) < 0)
        return -1;
    if (rpc_write(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
}

int handle_cuModuleUnload(void *conn) {
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleUnload(hmod);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuModuleGetLoadingMode(void *conn) {
    CUmoduleLoadingMode* mode;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleGetLoadingMode(mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_cuModuleGetFunction(void *conn) {
    CUfunction* hfunc;
    CUmodule hmod;
    const char* name;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleGetFunction(hfunc, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuModuleGetGlobal_v2(void *conn) {
    CUdeviceptr* dptr;
    size_t* bytes;
    CUmodule hmod;
    const char* name;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    if (rpc_read(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleGetGlobal_v2(dptr, bytes, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuLinkCreate_v2(void *conn) {
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    CUlinkState* stateOut;
    if (rpc_read(conn, &numOptions, sizeof(numOptions)) < 0)
        return -1;
    if (rpc_read(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_read(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
    if (rpc_read(conn, &stateOut, sizeof(stateOut)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLinkCreate_v2(numOptions, options, optionValues, stateOut);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_write(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
    if (rpc_write(conn, &stateOut, sizeof(stateOut)) < 0)
        return -1;
}

int handle_cuLinkAddData_v2(void *conn) {
    CUlinkState state;
    CUjitInputType type;
    void* data;
    size_t size;
    const char* name;
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    if (rpc_read(conn, &state, sizeof(state)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    if (rpc_read(conn, &numOptions, sizeof(numOptions)) < 0)
        return -1;
    if (rpc_read(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_read(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
    if (rpc_write(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_write(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
}

int handle_cuLinkAddFile_v2(void *conn) {
    CUlinkState state;
    CUjitInputType type;
    const char* path;
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    if (rpc_read(conn, &state, sizeof(state)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    if (rpc_read(conn, &path, sizeof(path)) < 0)
        return -1;
    if (rpc_read(conn, &numOptions, sizeof(numOptions)) < 0)
        return -1;
    if (rpc_read(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_read(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &path, sizeof(path)) < 0)
        return -1;
    if (rpc_write(conn, &options, sizeof(options)) < 0)
        return -1;
    if (rpc_write(conn, &optionValues, sizeof(optionValues)) < 0)
        return -1;
}

int handle_cuLinkComplete(void *conn) {
    CUlinkState state;
    void** cubinOut;
    size_t* sizeOut;
    if (rpc_read(conn, &state, sizeof(state)) < 0)
        return -1;
    if (rpc_read(conn, &cubinOut, sizeof(cubinOut)) < 0)
        return -1;
    if (rpc_read(conn, &sizeOut, sizeof(sizeOut)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLinkComplete(state, cubinOut, sizeOut);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &cubinOut, sizeof(cubinOut)) < 0)
        return -1;
    if (rpc_write(conn, &sizeOut, sizeof(sizeOut)) < 0)
        return -1;
}

int handle_cuLinkDestroy(void *conn) {
    CUlinkState state;
    if (rpc_read(conn, &state, sizeof(state)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLinkDestroy(state);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuModuleGetTexRef(void *conn) {
    CUtexref* pTexRef;
    CUmodule hmod;
    const char* name;
    if (rpc_read(conn, &pTexRef, sizeof(pTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleGetTexRef(pTexRef, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexRef, sizeof(pTexRef)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuModuleGetSurfRef(void *conn) {
    CUsurfref* pSurfRef;
    CUmodule hmod;
    const char* name;
    if (rpc_read(conn, &pSurfRef, sizeof(pSurfRef)) < 0)
        return -1;
    if (rpc_read(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuModuleGetSurfRef(pSurfRef, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSurfRef, sizeof(pSurfRef)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuLibraryLoadData(void *conn) {
    CUlibrary* library;
    const void* code;
    CUjit_option* jitOptions;
    void** jitOptionsValues;
    unsigned int numJitOptions;
    CUlibraryOption* libraryOptions;
    void** libraryOptionValues;
    unsigned int numLibraryOptions;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_read(conn, &code, sizeof(code)) < 0)
        return -1;
    if (rpc_read(conn, &jitOptions, sizeof(jitOptions)) < 0)
        return -1;
    if (rpc_read(conn, &jitOptionsValues, sizeof(jitOptionsValues)) < 0)
        return -1;
    if (rpc_read(conn, &numJitOptions, sizeof(numJitOptions)) < 0)
        return -1;
    if (rpc_read(conn, &libraryOptions, sizeof(libraryOptions)) < 0)
        return -1;
    if (rpc_read(conn, &libraryOptionValues, sizeof(libraryOptionValues)) < 0)
        return -1;
    if (rpc_read(conn, &numLibraryOptions, sizeof(numLibraryOptions)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_write(conn, &code, sizeof(code)) < 0)
        return -1;
    if (rpc_write(conn, &jitOptions, sizeof(jitOptions)) < 0)
        return -1;
    if (rpc_write(conn, &jitOptionsValues, sizeof(jitOptionsValues)) < 0)
        return -1;
    if (rpc_write(conn, &libraryOptions, sizeof(libraryOptions)) < 0)
        return -1;
    if (rpc_write(conn, &libraryOptionValues, sizeof(libraryOptionValues)) < 0)
        return -1;
}

int handle_cuLibraryLoadFromFile(void *conn) {
    CUlibrary* library;
    const char* fileName;
    CUjit_option* jitOptions;
    void** jitOptionsValues;
    unsigned int numJitOptions;
    CUlibraryOption* libraryOptions;
    void** libraryOptionValues;
    unsigned int numLibraryOptions;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_read(conn, &fileName, sizeof(fileName)) < 0)
        return -1;
    if (rpc_read(conn, &jitOptions, sizeof(jitOptions)) < 0)
        return -1;
    if (rpc_read(conn, &jitOptionsValues, sizeof(jitOptionsValues)) < 0)
        return -1;
    if (rpc_read(conn, &numJitOptions, sizeof(numJitOptions)) < 0)
        return -1;
    if (rpc_read(conn, &libraryOptions, sizeof(libraryOptions)) < 0)
        return -1;
    if (rpc_read(conn, &libraryOptionValues, sizeof(libraryOptionValues)) < 0)
        return -1;
    if (rpc_read(conn, &numLibraryOptions, sizeof(numLibraryOptions)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_write(conn, &fileName, sizeof(fileName)) < 0)
        return -1;
    if (rpc_write(conn, &jitOptions, sizeof(jitOptions)) < 0)
        return -1;
    if (rpc_write(conn, &jitOptionsValues, sizeof(jitOptionsValues)) < 0)
        return -1;
    if (rpc_write(conn, &libraryOptions, sizeof(libraryOptions)) < 0)
        return -1;
    if (rpc_write(conn, &libraryOptionValues, sizeof(libraryOptionValues)) < 0)
        return -1;
}

int handle_cuLibraryUnload(void *conn) {
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryUnload(library);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuLibraryGetKernel(void *conn) {
    CUkernel* pKernel;
    CUlibrary library;
    const char* name;
    if (rpc_read(conn, &pKernel, sizeof(pKernel)) < 0)
        return -1;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryGetKernel(pKernel, library, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pKernel, sizeof(pKernel)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuLibraryGetModule(void *conn) {
    CUmodule* pMod;
    CUlibrary library;
    if (rpc_read(conn, &pMod, sizeof(pMod)) < 0)
        return -1;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryGetModule(pMod, library);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pMod, sizeof(pMod)) < 0)
        return -1;
}

int handle_cuKernelGetFunction(void *conn) {
    CUfunction* pFunc;
    CUkernel kernel;
    if (rpc_read(conn, &pFunc, sizeof(pFunc)) < 0)
        return -1;
    if (rpc_read(conn, &kernel, sizeof(kernel)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuKernelGetFunction(pFunc, kernel);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFunc, sizeof(pFunc)) < 0)
        return -1;
}

int handle_cuLibraryGetGlobal(void *conn) {
    CUdeviceptr* dptr;
    size_t* bytes;
    CUlibrary library;
    const char* name;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryGetGlobal(dptr, bytes, library, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuLibraryGetManaged(void *conn) {
    CUdeviceptr* dptr;
    size_t* bytes;
    CUlibrary library;
    const char* name;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_read(conn, &name, sizeof(name)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryGetManaged(dptr, bytes, library, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    if (rpc_write(conn, &name, sizeof(name)) < 0)
        return -1;
}

int handle_cuLibraryGetUnifiedFunction(void *conn) {
    void** fptr;
    CUlibrary library;
    const char* symbol;
    if (rpc_read(conn, &fptr, sizeof(fptr)) < 0)
        return -1;
    if (rpc_read(conn, &library, sizeof(library)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLibraryGetUnifiedFunction(fptr, library, symbol);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fptr, sizeof(fptr)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cuKernelGetAttribute(void *conn) {
    int* pi;
    CUfunction_attribute attrib;
    CUkernel kernel;
    CUdevice dev;
    if (rpc_read(conn, &pi, sizeof(pi)) < 0)
        return -1;
    if (rpc_read(conn, &attrib, sizeof(attrib)) < 0)
        return -1;
    if (rpc_read(conn, &kernel, sizeof(kernel)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuKernelGetAttribute(pi, attrib, kernel, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(pi)) < 0)
        return -1;
}

int handle_cuKernelSetAttribute(void *conn) {
    CUfunction_attribute attrib;
    int val;
    CUkernel kernel;
    CUdevice dev;
    if (rpc_read(conn, &attrib, sizeof(attrib)) < 0)
        return -1;
    if (rpc_read(conn, &val, sizeof(val)) < 0)
        return -1;
    if (rpc_read(conn, &kernel, sizeof(kernel)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuKernelSetAttribute(attrib, val, kernel, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuKernelSetCacheConfig(void *conn) {
    CUkernel kernel;
    CUfunc_cache config;
    CUdevice dev;
    if (rpc_read(conn, &kernel, sizeof(kernel)) < 0)
        return -1;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuKernelSetCacheConfig(kernel, config, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemGetInfo_v2(void *conn) {
    size_t* free;
    size_t* total;
    if (rpc_read(conn, &free, sizeof(free)) < 0)
        return -1;
    if (rpc_read(conn, &total, sizeof(total)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemGetInfo_v2(free, total);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &free, sizeof(free)) < 0)
        return -1;
    if (rpc_write(conn, &total, sizeof(total)) < 0)
        return -1;
}

int handle_cuMemAlloc_v2(void *conn) {
    CUdeviceptr* dptr;
    size_t bytesize;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAlloc_v2(dptr, bytesize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
}

int handle_cuMemAllocPitch_v2(void *conn) {
    CUdeviceptr* dptr;
    size_t* pPitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &pPitch, sizeof(pPitch)) < 0)
        return -1;
    if (rpc_read(conn, &WidthInBytes, sizeof(WidthInBytes)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    if (rpc_read(conn, &ElementSizeBytes, sizeof(ElementSizeBytes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_write(conn, &pPitch, sizeof(pPitch)) < 0)
        return -1;
}

int handle_cuMemFree_v2(void *conn) {
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemFree_v2(dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemGetAddressRange_v2(void *conn) {
    CUdeviceptr* pbase;
    size_t* psize;
    CUdeviceptr dptr;
    if (rpc_read(conn, &pbase, sizeof(pbase)) < 0)
        return -1;
    if (rpc_read(conn, &psize, sizeof(psize)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemGetAddressRange_v2(pbase, psize, dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pbase, sizeof(pbase)) < 0)
        return -1;
    if (rpc_write(conn, &psize, sizeof(psize)) < 0)
        return -1;
}

int handle_cuMemAllocHost_v2(void *conn) {
    void** pp;
    size_t bytesize;
    if (rpc_read(conn, &pp, sizeof(pp)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAllocHost_v2(pp, bytesize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pp, sizeof(pp)) < 0)
        return -1;
}

int handle_cuMemFreeHost(void *conn) {
    void* p;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemFreeHost(p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cuMemHostAlloc(void *conn) {
    void** pp;
    size_t bytesize;
    unsigned int Flags;
    if (rpc_read(conn, &pp, sizeof(pp)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemHostAlloc(pp, bytesize, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pp, sizeof(pp)) < 0)
        return -1;
}

int handle_cuMemHostGetDevicePointer_v2(void *conn) {
    CUdeviceptr* pdptr;
    void* p;
    unsigned int Flags;
    if (rpc_read(conn, &pdptr, sizeof(pdptr)) < 0)
        return -1;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemHostGetDevicePointer_v2(pdptr, p, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pdptr, sizeof(pdptr)) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cuMemHostGetFlags(void *conn) {
    unsigned int* pFlags;
    void* p;
    if (rpc_read(conn, &pFlags, sizeof(pFlags)) < 0)
        return -1;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemHostGetFlags(pFlags, p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFlags, sizeof(pFlags)) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cuMemAllocManaged(void *conn) {
    CUdeviceptr* dptr;
    size_t bytesize;
    unsigned int flags;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAllocManaged(dptr, bytesize, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
}

int handle_cuDeviceGetByPCIBusId(void *conn) {
    CUdevice* dev;
    const char* pciBusId;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    if (rpc_read(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetByPCIBusId(dev, pciBusId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dev, sizeof(dev)) < 0)
        return -1;
    if (rpc_write(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
}

int handle_cuDeviceGetPCIBusId(void *conn) {
    char* pciBusId;
    int len;
    CUdevice dev;
    if (rpc_read(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
    if (rpc_read(conn, &len, sizeof(len)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetPCIBusId(pciBusId, len, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
}

int handle_cuIpcGetEventHandle(void *conn) {
    CUipcEventHandle* pHandle;
    CUevent event;
    if (rpc_read(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuIpcGetEventHandle(pHandle, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
}

int handle_cuIpcOpenEventHandle(void *conn) {
    CUevent* phEvent;
    CUipcEventHandle handle;
    if (rpc_read(conn, &phEvent, sizeof(phEvent)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuIpcOpenEventHandle(phEvent, handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phEvent, sizeof(phEvent)) < 0)
        return -1;
}

int handle_cuIpcGetMemHandle(void *conn) {
    CUipcMemHandle* pHandle;
    CUdeviceptr dptr;
    if (rpc_read(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuIpcGetMemHandle(pHandle, dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
}

int handle_cuIpcOpenMemHandle_v2(void *conn) {
    CUdeviceptr* pdptr;
    CUipcMemHandle handle;
    unsigned int Flags;
    if (rpc_read(conn, &pdptr, sizeof(pdptr)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuIpcOpenMemHandle_v2(pdptr, handle, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pdptr, sizeof(pdptr)) < 0)
        return -1;
}

int handle_cuIpcCloseMemHandle(void *conn) {
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuIpcCloseMemHandle(dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemHostRegister_v2(void *conn) {
    void* p;
    size_t bytesize;
    unsigned int Flags;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemHostRegister_v2(p, bytesize, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cuMemHostUnregister(void *conn) {
    void* p;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemHostUnregister(p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cuMemcpy(void *conn) {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy(dst, src, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyPeer(void *conn) {
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstContext, sizeof(dstContext)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcContext, sizeof(srcContext)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyHtoD_v2(void *conn) {
    CUdeviceptr dstDevice;
    const void* srcHost;
    size_t ByteCount;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
}

int handle_cuMemcpyDtoH_v2(void *conn) {
    void* dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
}

int handle_cuMemcpyDtoD_v2(void *conn) {
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyDtoA_v2(void *conn) {
    CUarray dstArray;
    size_t dstOffset;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    if (rpc_read(conn, &dstArray, sizeof(dstArray)) < 0)
        return -1;
    if (rpc_read(conn, &dstOffset, sizeof(dstOffset)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyAtoD_v2(void *conn) {
    CUdeviceptr dstDevice;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0)
        return -1;
    if (rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyHtoA_v2(void *conn) {
    CUarray dstArray;
    size_t dstOffset;
    const void* srcHost;
    size_t ByteCount;
    if (rpc_read(conn, &dstArray, sizeof(dstArray)) < 0)
        return -1;
    if (rpc_read(conn, &dstOffset, sizeof(dstOffset)) < 0)
        return -1;
    if (rpc_read(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
}

int handle_cuMemcpyAtoH_v2(void *conn) {
    void* dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
    if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0)
        return -1;
    if (rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
}

int handle_cuMemcpyAtoA_v2(void *conn) {
    CUarray dstArray;
    size_t dstOffset;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    if (rpc_read(conn, &dstArray, sizeof(dstArray)) < 0)
        return -1;
    if (rpc_read(conn, &dstOffset, sizeof(dstOffset)) < 0)
        return -1;
    if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0)
        return -1;
    if (rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpy2D_v2(void *conn) {
    const CUDA_MEMCPY2D* pCopy;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy2D_v2(pCopy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemcpy2DUnaligned_v2(void *conn) {
    const CUDA_MEMCPY2D* pCopy;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy2DUnaligned_v2(pCopy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemcpy3D_v2(void *conn) {
    const CUDA_MEMCPY3D* pCopy;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy3D_v2(pCopy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemcpy3DPeer(void *conn) {
    const CUDA_MEMCPY3D_PEER* pCopy;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy3DPeer(pCopy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemcpyAsync(void *conn) {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyAsync(dst, src, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyPeerAsync(void *conn) {
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstContext, sizeof(dstContext)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcContext, sizeof(srcContext)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyHtoDAsync_v2(void *conn) {
    CUdeviceptr dstDevice;
    const void* srcHost;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
}

int handle_cuMemcpyDtoHAsync_v2(void *conn) {
    void* dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
}

int handle_cuMemcpyDtoDAsync_v2(void *conn) {
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemcpyHtoAAsync_v2(void *conn) {
    CUarray dstArray;
    size_t dstOffset;
    const void* srcHost;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dstArray, sizeof(dstArray)) < 0)
        return -1;
    if (rpc_read(conn, &dstOffset, sizeof(dstOffset)) < 0)
        return -1;
    if (rpc_read(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &srcHost, sizeof(srcHost)) < 0)
        return -1;
}

int handle_cuMemcpyAtoHAsync_v2(void *conn) {
    void* dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    CUstream hStream;
    if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
    if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0)
        return -1;
    if (rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0)
        return -1;
    if (rpc_read(conn, &ByteCount, sizeof(ByteCount)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(dstHost)) < 0)
        return -1;
}

int handle_cuMemcpy2DAsync_v2(void *conn) {
    const CUDA_MEMCPY2D* pCopy;
    CUstream hStream;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy2DAsync_v2(pCopy, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemcpy3DAsync_v2(void *conn) {
    const CUDA_MEMCPY3D* pCopy;
    CUstream hStream;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy3DAsync_v2(pCopy, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemcpy3DPeerAsync(void *conn) {
    const CUDA_MEMCPY3D_PEER* pCopy;
    CUstream hStream;
    if (rpc_read(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemcpy3DPeerAsync(pCopy, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCopy, sizeof(pCopy)) < 0)
        return -1;
}

int handle_cuMemsetD8_v2(void *conn) {
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &uc, sizeof(uc)) < 0)
        return -1;
    if (rpc_read(conn, &N, sizeof(N)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD8_v2(dstDevice, uc, N);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD16_v2(void *conn) {
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &us, sizeof(us)) < 0)
        return -1;
    if (rpc_read(conn, &N, sizeof(N)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD16_v2(dstDevice, us, N);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD32_v2(void *conn) {
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ui, sizeof(ui)) < 0)
        return -1;
    if (rpc_read(conn, &N, sizeof(N)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD32_v2(dstDevice, ui, N);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD2D8_v2(void *conn) {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstPitch, sizeof(dstPitch)) < 0)
        return -1;
    if (rpc_read(conn, &uc, sizeof(uc)) < 0)
        return -1;
    if (rpc_read(conn, &Width, sizeof(Width)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD2D16_v2(void *conn) {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstPitch, sizeof(dstPitch)) < 0)
        return -1;
    if (rpc_read(conn, &us, sizeof(us)) < 0)
        return -1;
    if (rpc_read(conn, &Width, sizeof(Width)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD2D32_v2(void *conn) {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstPitch, sizeof(dstPitch)) < 0)
        return -1;
    if (rpc_read(conn, &ui, sizeof(ui)) < 0)
        return -1;
    if (rpc_read(conn, &Width, sizeof(Width)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD8Async(void *conn) {
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &uc, sizeof(uc)) < 0)
        return -1;
    if (rpc_read(conn, &N, sizeof(N)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD8Async(dstDevice, uc, N, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD16Async(void *conn) {
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &us, sizeof(us)) < 0)
        return -1;
    if (rpc_read(conn, &N, sizeof(N)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD16Async(dstDevice, us, N, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD32Async(void *conn) {
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &ui, sizeof(ui)) < 0)
        return -1;
    if (rpc_read(conn, &N, sizeof(N)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD32Async(dstDevice, ui, N, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD2D8Async(void *conn) {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstPitch, sizeof(dstPitch)) < 0)
        return -1;
    if (rpc_read(conn, &uc, sizeof(uc)) < 0)
        return -1;
    if (rpc_read(conn, &Width, sizeof(Width)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD2D16Async(void *conn) {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstPitch, sizeof(dstPitch)) < 0)
        return -1;
    if (rpc_read(conn, &us, sizeof(us)) < 0)
        return -1;
    if (rpc_read(conn, &Width, sizeof(Width)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemsetD2D32Async(void *conn) {
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    CUstream hStream;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstPitch, sizeof(dstPitch)) < 0)
        return -1;
    if (rpc_read(conn, &ui, sizeof(ui)) < 0)
        return -1;
    if (rpc_read(conn, &Width, sizeof(Width)) < 0)
        return -1;
    if (rpc_read(conn, &Height, sizeof(Height)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuArrayCreate_v2(void *conn) {
    CUarray* pHandle;
    const CUDA_ARRAY_DESCRIPTOR* pAllocateArray;
    if (rpc_read(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_read(conn, &pAllocateArray, sizeof(pAllocateArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArrayCreate_v2(pHandle, pAllocateArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_write(conn, &pAllocateArray, sizeof(pAllocateArray)) < 0)
        return -1;
}

int handle_cuArrayGetDescriptor_v2(void *conn) {
    CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor;
    CUarray hArray;
    if (rpc_read(conn, &pArrayDescriptor, sizeof(pArrayDescriptor)) < 0)
        return -1;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArrayGetDescriptor_v2(pArrayDescriptor, hArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pArrayDescriptor, sizeof(pArrayDescriptor)) < 0)
        return -1;
}

int handle_cuArrayGetSparseProperties(void *conn) {
    CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties;
    CUarray array;
    if (rpc_read(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArrayGetSparseProperties(sparseProperties, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
}

int handle_cuMipmappedArrayGetSparseProperties(void *conn) {
    CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties;
    CUmipmappedArray mipmap;
    if (rpc_read(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
    if (rpc_read(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
}

int handle_cuArrayGetMemoryRequirements(void *conn) {
    CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements;
    CUarray array;
    CUdevice device;
    if (rpc_read(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArrayGetMemoryRequirements(memoryRequirements, array, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
}

int handle_cuMipmappedArrayGetMemoryRequirements(void *conn) {
    CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements;
    CUmipmappedArray mipmap;
    CUdevice device;
    if (rpc_read(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
    if (rpc_read(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
}

int handle_cuArrayGetPlane(void *conn) {
    CUarray* pPlaneArray;
    CUarray hArray;
    unsigned int planeIdx;
    if (rpc_read(conn, &pPlaneArray, sizeof(pPlaneArray)) < 0)
        return -1;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    if (rpc_read(conn, &planeIdx, sizeof(planeIdx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArrayGetPlane(pPlaneArray, hArray, planeIdx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pPlaneArray, sizeof(pPlaneArray)) < 0)
        return -1;
}

int handle_cuArrayDestroy(void *conn) {
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArrayDestroy(hArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuArray3DCreate_v2(void *conn) {
    CUarray* pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray;
    if (rpc_read(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_read(conn, &pAllocateArray, sizeof(pAllocateArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArray3DCreate_v2(pHandle, pAllocateArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_write(conn, &pAllocateArray, sizeof(pAllocateArray)) < 0)
        return -1;
}

int handle_cuArray3DGetDescriptor_v2(void *conn) {
    CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor;
    CUarray hArray;
    if (rpc_read(conn, &pArrayDescriptor, sizeof(pArrayDescriptor)) < 0)
        return -1;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pArrayDescriptor, sizeof(pArrayDescriptor)) < 0)
        return -1;
}

int handle_cuMipmappedArrayCreate(void *conn) {
    CUmipmappedArray* pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc;
    unsigned int numMipmapLevels;
    if (rpc_read(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_read(conn, &pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc)) < 0)
        return -1;
    if (rpc_read(conn, &numMipmapLevels, sizeof(numMipmapLevels)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(pHandle)) < 0)
        return -1;
    if (rpc_write(conn, &pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc)) < 0)
        return -1;
}

int handle_cuMipmappedArrayGetLevel(void *conn) {
    CUarray* pLevelArray;
    CUmipmappedArray hMipmappedArray;
    unsigned int level;
    if (rpc_read(conn, &pLevelArray, sizeof(pLevelArray)) < 0)
        return -1;
    if (rpc_read(conn, &hMipmappedArray, sizeof(hMipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &level, sizeof(level)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pLevelArray, sizeof(pLevelArray)) < 0)
        return -1;
}

int handle_cuMipmappedArrayDestroy(void *conn) {
    CUmipmappedArray hMipmappedArray;
    if (rpc_read(conn, &hMipmappedArray, sizeof(hMipmappedArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMipmappedArrayDestroy(hMipmappedArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemGetHandleForAddressRange(void *conn) {
    void* handle;
    CUdeviceptr dptr;
    size_t size;
    CUmemRangeHandleType handleType;
    unsigned long long flags;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &handleType, sizeof(handleType)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
}

int handle_cuMemAddressReserve(void *conn) {
    CUdeviceptr* ptr;
    size_t size;
    size_t alignment;
    CUdeviceptr addr;
    unsigned long long flags;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &alignment, sizeof(alignment)) < 0)
        return -1;
    if (rpc_read(conn, &addr, sizeof(addr)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAddressReserve(ptr, size, alignment, addr, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cuMemAddressFree(void *conn) {
    CUdeviceptr ptr;
    size_t size;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAddressFree(ptr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemCreate(void *conn) {
    CUmemGenericAllocationHandle* handle;
    size_t size;
    const CUmemAllocationProp* prop;
    unsigned long long flags;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &prop, sizeof(prop)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemCreate(handle, size, prop, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(prop)) < 0)
        return -1;
}

int handle_cuMemRelease(void *conn) {
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemRelease(handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemMap(void *conn) {
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUmemGenericAllocationHandle handle;
    unsigned long long flags;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemMap(ptr, size, offset, handle, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemMapArrayAsync(void *conn) {
    CUarrayMapInfo* mapInfoList;
    unsigned int count;
    CUstream hStream;
    if (rpc_read(conn, &mapInfoList, sizeof(mapInfoList)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemMapArrayAsync(mapInfoList, count, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mapInfoList, sizeof(mapInfoList)) < 0)
        return -1;
}

int handle_cuMemUnmap(void *conn) {
    CUdeviceptr ptr;
    size_t size;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemUnmap(ptr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemSetAccess(void *conn) {
    CUdeviceptr ptr;
    size_t size;
    const CUmemAccessDesc* desc;
    size_t count;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemSetAccess(ptr, size, desc, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
}

int handle_cuMemGetAccess(void *conn) {
    unsigned long long* flags;
    const CUmemLocation* location;
    CUdeviceptr ptr;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &location, sizeof(location)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemGetAccess(flags, location, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_write(conn, &location, sizeof(location)) < 0)
        return -1;
}

int handle_cuMemExportToShareableHandle(void *conn) {
    void* shareableHandle;
    CUmemGenericAllocationHandle handle;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
    if (rpc_read(conn, &shareableHandle, sizeof(shareableHandle)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &handleType, sizeof(handleType)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &shareableHandle, sizeof(shareableHandle)) < 0)
        return -1;
}

int handle_cuMemImportFromShareableHandle(void *conn) {
    CUmemGenericAllocationHandle* handle;
    void* osHandle;
    CUmemAllocationHandleType shHandleType;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &osHandle, sizeof(osHandle)) < 0)
        return -1;
    if (rpc_read(conn, &shHandleType, sizeof(shHandleType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemImportFromShareableHandle(handle, osHandle, shHandleType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_write(conn, &osHandle, sizeof(osHandle)) < 0)
        return -1;
}

int handle_cuMemGetAllocationGranularity(void *conn) {
    size_t* granularity;
    const CUmemAllocationProp* prop;
    CUmemAllocationGranularity_flags option;
    if (rpc_read(conn, &granularity, sizeof(granularity)) < 0)
        return -1;
    if (rpc_read(conn, &prop, sizeof(prop)) < 0)
        return -1;
    if (rpc_read(conn, &option, sizeof(option)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemGetAllocationGranularity(granularity, prop, option);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &granularity, sizeof(granularity)) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(prop)) < 0)
        return -1;
}

int handle_cuMemGetAllocationPropertiesFromHandle(void *conn) {
    CUmemAllocationProp* prop;
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &prop, sizeof(prop)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemGetAllocationPropertiesFromHandle(prop, handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(prop)) < 0)
        return -1;
}

int handle_cuMemRetainAllocationHandle(void *conn) {
    CUmemGenericAllocationHandle* handle;
    void* addr;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &addr, sizeof(addr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemRetainAllocationHandle(handle, addr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_write(conn, &addr, sizeof(addr)) < 0)
        return -1;
}

int handle_cuMemFreeAsync(void *conn) {
    CUdeviceptr dptr;
    CUstream hStream;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemFreeAsync(dptr, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemAllocAsync(void *conn) {
    CUdeviceptr* dptr;
    size_t bytesize;
    CUstream hStream;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAllocAsync(dptr, bytesize, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
}

int handle_cuMemPoolTrimTo(void *conn) {
    CUmemoryPool pool;
    size_t minBytesToKeep;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &minBytesToKeep, sizeof(minBytesToKeep)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolTrimTo(pool, minBytesToKeep);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemPoolSetAttribute(void *conn) {
    CUmemoryPool pool;
    CUmemPool_attribute attr;
    void* value;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolSetAttribute(pool, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuMemPoolGetAttribute(void *conn) {
    CUmemoryPool pool;
    CUmemPool_attribute attr;
    void* value;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolGetAttribute(pool, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuMemPoolSetAccess(void *conn) {
    CUmemoryPool pool;
    const CUmemAccessDesc* map;
    size_t count;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &map, sizeof(map)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolSetAccess(pool, map, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &map, sizeof(map)) < 0)
        return -1;
}

int handle_cuMemPoolGetAccess(void *conn) {
    CUmemAccess_flags* flags;
    CUmemoryPool memPool;
    CUmemLocation* location;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &location, sizeof(location)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolGetAccess(flags, memPool, location);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_write(conn, &location, sizeof(location)) < 0)
        return -1;
}

int handle_cuMemPoolCreate(void *conn) {
    CUmemoryPool* pool;
    const CUmemPoolProps* poolProps;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &poolProps, sizeof(poolProps)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolCreate(pool, poolProps);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_write(conn, &poolProps, sizeof(poolProps)) < 0)
        return -1;
}

int handle_cuMemPoolDestroy(void *conn) {
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolDestroy(pool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemAllocFromPoolAsync(void *conn) {
    CUdeviceptr* dptr;
    size_t bytesize;
    CUmemoryPool pool;
    CUstream hStream;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
        return -1;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
}

int handle_cuMemPoolExportToShareableHandle(void *conn) {
    void* handle_out;
    CUmemoryPool pool;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
    if (rpc_read(conn, &handle_out, sizeof(handle_out)) < 0)
        return -1;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &handleType, sizeof(handleType)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle_out, sizeof(handle_out)) < 0)
        return -1;
}

int handle_cuMemPoolImportFromShareableHandle(void *conn) {
    CUmemoryPool* pool_out;
    void* handle;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
    if (rpc_read(conn, &pool_out, sizeof(pool_out)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &handleType, sizeof(handleType)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool_out, sizeof(pool_out)) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
}

int handle_cuMemPoolExportPointer(void *conn) {
    CUmemPoolPtrExportData* shareData_out;
    CUdeviceptr ptr;
    if (rpc_read(conn, &shareData_out, sizeof(shareData_out)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolExportPointer(shareData_out, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &shareData_out, sizeof(shareData_out)) < 0)
        return -1;
}

int handle_cuMemPoolImportPointer(void *conn) {
    CUdeviceptr* ptr_out;
    CUmemoryPool pool;
    CUmemPoolPtrExportData* shareData;
    if (rpc_read(conn, &ptr_out, sizeof(ptr_out)) < 0)
        return -1;
    if (rpc_read(conn, &pool, sizeof(pool)) < 0)
        return -1;
    if (rpc_read(conn, &shareData, sizeof(shareData)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPoolImportPointer(ptr_out, pool, shareData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr_out, sizeof(ptr_out)) < 0)
        return -1;
    if (rpc_write(conn, &shareData, sizeof(shareData)) < 0)
        return -1;
}

int handle_cuPointerGetAttribute(void *conn) {
    void* data;
    CUpointer_attribute attribute;
    CUdeviceptr ptr;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &attribute, sizeof(attribute)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuPointerGetAttribute(data, attribute, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
}

int handle_cuMemPrefetchAsync(void *conn) {
    CUdeviceptr devPtr;
    size_t count;
    CUdevice dstDevice;
    CUstream hStream;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemAdvise(void *conn) {
    CUdeviceptr devPtr;
    size_t count;
    CUmem_advise advice;
    CUdevice device;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &advice, sizeof(advice)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemAdvise(devPtr, count, advice, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuMemRangeGetAttribute(void *conn) {
    void* data;
    size_t dataSize;
    CUmem_range_attribute attribute;
    CUdeviceptr devPtr;
    size_t count;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &dataSize, sizeof(dataSize)) < 0)
        return -1;
    if (rpc_read(conn, &attribute, sizeof(attribute)) < 0)
        return -1;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
}

int handle_cuMemRangeGetAttributes(void *conn) {
    void** data;
    size_t* dataSizes;
    CUmem_range_attribute* attributes;
    size_t numAttributes;
    CUdeviceptr devPtr;
    size_t count;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &dataSizes, sizeof(dataSizes)) < 0)
        return -1;
    if (rpc_read(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_read(conn, &numAttributes, sizeof(numAttributes)) < 0)
        return -1;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_write(conn, &dataSizes, sizeof(dataSizes)) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
}

int handle_cuPointerSetAttribute(void *conn) {
    const void* value;
    CUpointer_attribute attribute;
    CUdeviceptr ptr;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &attribute, sizeof(attribute)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuPointerSetAttribute(value, attribute, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuPointerGetAttributes(void *conn) {
    unsigned int numAttributes;
    CUpointer_attribute* attributes;
    void** data;
    CUdeviceptr ptr;
    if (rpc_read(conn, &numAttributes, sizeof(numAttributes)) < 0)
        return -1;
    if (rpc_read(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuPointerGetAttributes(numAttributes, attributes, data, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
}

int handle_cuStreamCreate(void *conn) {
    CUstream* phStream;
    unsigned int Flags;
    if (rpc_read(conn, &phStream, sizeof(phStream)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamCreate(phStream, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phStream, sizeof(phStream)) < 0)
        return -1;
}

int handle_cuStreamCreateWithPriority(void *conn) {
    CUstream* phStream;
    unsigned int flags;
    int priority;
    if (rpc_read(conn, &phStream, sizeof(phStream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &priority, sizeof(priority)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamCreateWithPriority(phStream, flags, priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phStream, sizeof(phStream)) < 0)
        return -1;
}

int handle_cuStreamGetPriority(void *conn) {
    CUstream hStream;
    int* priority;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &priority, sizeof(priority)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamGetPriority(hStream, priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &priority, sizeof(priority)) < 0)
        return -1;
}

int handle_cuStreamGetFlags(void *conn) {
    CUstream hStream;
    unsigned int* flags;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamGetFlags(hStream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cuStreamGetId(void *conn) {
    CUstream hStream;
    unsigned long long* streamId;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &streamId, sizeof(streamId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamGetId(hStream, streamId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &streamId, sizeof(streamId)) < 0)
        return -1;
}

int handle_cuStreamGetCtx(void *conn) {
    CUstream hStream;
    CUcontext* pctx;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamGetCtx(hStream, pctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(pctx)) < 0)
        return -1;
}

int handle_cuStreamWaitEvent(void *conn) {
    CUstream hStream;
    CUevent hEvent;
    unsigned int Flags;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &hEvent, sizeof(hEvent)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamWaitEvent(hStream, hEvent, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamAddCallback(void *conn) {
    CUstream hStream;
    CUstreamCallback callback;
    void* userData;
    unsigned int flags;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &callback, sizeof(callback)) < 0)
        return -1;
    if (rpc_read(conn, &userData, sizeof(userData)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamAddCallback(hStream, callback, userData, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(userData)) < 0)
        return -1;
}

int handle_cuStreamBeginCapture_v2(void *conn) {
    CUstream hStream;
    CUstreamCaptureMode mode;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamBeginCapture_v2(hStream, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuThreadExchangeStreamCaptureMode(void *conn) {
    CUstreamCaptureMode* mode;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuThreadExchangeStreamCaptureMode(mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_cuStreamEndCapture(void *conn) {
    CUstream hStream;
    CUgraph* phGraph;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &phGraph, sizeof(phGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamEndCapture(hStream, phGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraph, sizeof(phGraph)) < 0)
        return -1;
}

int handle_cuStreamIsCapturing(void *conn) {
    CUstream hStream;
    CUstreamCaptureStatus* captureStatus;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &captureStatus, sizeof(captureStatus)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamIsCapturing(hStream, captureStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &captureStatus, sizeof(captureStatus)) < 0)
        return -1;
}

int handle_cuStreamGetCaptureInfo_v2(void *conn) {
    CUstream hStream;
    CUstreamCaptureStatus* captureStatus_out;
    cuuint64_t* id_out;
    CUgraph* graph_out;
    const CUgraphNode** dependencies_out;
    size_t* numDependencies_out;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &captureStatus_out, sizeof(captureStatus_out)) < 0)
        return -1;
    if (rpc_read(conn, &id_out, sizeof(id_out)) < 0)
        return -1;
    if (rpc_read(conn, &graph_out, sizeof(graph_out)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies_out, sizeof(dependencies_out)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies_out, sizeof(numDependencies_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &captureStatus_out, sizeof(captureStatus_out)) < 0)
        return -1;
    if (rpc_write(conn, &id_out, sizeof(id_out)) < 0)
        return -1;
    if (rpc_write(conn, &graph_out, sizeof(graph_out)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies_out, sizeof(dependencies_out)) < 0)
        return -1;
    if (rpc_write(conn, &numDependencies_out, sizeof(numDependencies_out)) < 0)
        return -1;
}

int handle_cuStreamUpdateCaptureDependencies(void *conn) {
    CUstream hStream;
    CUgraphNode* dependencies;
    size_t numDependencies;
    unsigned int flags;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cuStreamAttachMemAsync(void *conn) {
    CUstream hStream;
    CUdeviceptr dptr;
    size_t length;
    unsigned int flags;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamAttachMemAsync(hStream, dptr, length, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamQuery(void *conn) {
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamQuery(hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamSynchronize(void *conn) {
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamSynchronize(hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamDestroy_v2(void *conn) {
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamDestroy_v2(hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamCopyAttributes(void *conn) {
    CUstream dst;
    CUstream src;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamGetAttribute(void *conn) {
    CUstream hStream;
    CUstreamAttrID attr;
    CUstreamAttrValue* value_out;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamGetAttribute(hStream, attr, value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
}

int handle_cuStreamSetAttribute(void *conn) {
    CUstream hStream;
    CUstreamAttrID attr;
    const CUstreamAttrValue* value;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamSetAttribute(hStream, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuEventCreate(void *conn) {
    CUevent* phEvent;
    unsigned int Flags;
    if (rpc_read(conn, &phEvent, sizeof(phEvent)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventCreate(phEvent, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phEvent, sizeof(phEvent)) < 0)
        return -1;
}

int handle_cuEventRecord(void *conn) {
    CUevent hEvent;
    CUstream hStream;
    if (rpc_read(conn, &hEvent, sizeof(hEvent)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventRecord(hEvent, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuEventRecordWithFlags(void *conn) {
    CUevent hEvent;
    CUstream hStream;
    unsigned int flags;
    if (rpc_read(conn, &hEvent, sizeof(hEvent)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventRecordWithFlags(hEvent, hStream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuEventQuery(void *conn) {
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(hEvent)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventQuery(hEvent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuEventSynchronize(void *conn) {
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(hEvent)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventSynchronize(hEvent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuEventDestroy_v2(void *conn) {
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(hEvent)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventDestroy_v2(hEvent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuEventElapsedTime(void *conn) {
    float* pMilliseconds;
    CUevent hStart;
    CUevent hEnd;
    if (rpc_read(conn, &pMilliseconds, sizeof(pMilliseconds)) < 0)
        return -1;
    if (rpc_read(conn, &hStart, sizeof(hStart)) < 0)
        return -1;
    if (rpc_read(conn, &hEnd, sizeof(hEnd)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuEventElapsedTime(pMilliseconds, hStart, hEnd);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pMilliseconds, sizeof(pMilliseconds)) < 0)
        return -1;
}

int handle_cuImportExternalMemory(void *conn) {
    CUexternalMemory* extMem_out;
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc;
    if (rpc_read(conn, &extMem_out, sizeof(extMem_out)) < 0)
        return -1;
    if (rpc_read(conn, &memHandleDesc, sizeof(memHandleDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuImportExternalMemory(extMem_out, memHandleDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extMem_out, sizeof(extMem_out)) < 0)
        return -1;
    if (rpc_write(conn, &memHandleDesc, sizeof(memHandleDesc)) < 0)
        return -1;
}

int handle_cuExternalMemoryGetMappedBuffer(void *conn) {
    CUdeviceptr* devPtr;
    CUexternalMemory extMem;
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &extMem, sizeof(extMem)) < 0)
        return -1;
    if (rpc_read(conn, &bufferDesc, sizeof(bufferDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_write(conn, &bufferDesc, sizeof(bufferDesc)) < 0)
        return -1;
}

int handle_cuExternalMemoryGetMappedMipmappedArray(void *conn) {
    CUmipmappedArray* mipmap;
    CUexternalMemory extMem;
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc;
    if (rpc_read(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    if (rpc_read(conn, &extMem, sizeof(extMem)) < 0)
        return -1;
    if (rpc_read(conn, &mipmapDesc, sizeof(mipmapDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    if (rpc_write(conn, &mipmapDesc, sizeof(mipmapDesc)) < 0)
        return -1;
}

int handle_cuDestroyExternalMemory(void *conn) {
    CUexternalMemory extMem;
    if (rpc_read(conn, &extMem, sizeof(extMem)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDestroyExternalMemory(extMem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuImportExternalSemaphore(void *conn) {
    CUexternalSemaphore* extSem_out;
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc;
    if (rpc_read(conn, &extSem_out, sizeof(extSem_out)) < 0)
        return -1;
    if (rpc_read(conn, &semHandleDesc, sizeof(semHandleDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuImportExternalSemaphore(extSem_out, semHandleDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extSem_out, sizeof(extSem_out)) < 0)
        return -1;
    if (rpc_write(conn, &semHandleDesc, sizeof(semHandleDesc)) < 0)
        return -1;
}

int handle_cuSignalExternalSemaphoresAsync(void *conn) {
    const CUexternalSemaphore* extSemArray;
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray;
    unsigned int numExtSems;
    CUstream stream;
    if (rpc_read(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_read(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
    if (rpc_read(conn, &numExtSems, sizeof(numExtSems)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_write(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
}

int handle_cuWaitExternalSemaphoresAsync(void *conn) {
    const CUexternalSemaphore* extSemArray;
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray;
    unsigned int numExtSems;
    CUstream stream;
    if (rpc_read(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_read(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
    if (rpc_read(conn, &numExtSems, sizeof(numExtSems)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_write(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
}

int handle_cuDestroyExternalSemaphore(void *conn) {
    CUexternalSemaphore extSem;
    if (rpc_read(conn, &extSem, sizeof(extSem)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDestroyExternalSemaphore(extSem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamWaitValue32_v2(void *conn) {
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &addr, sizeof(addr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamWaitValue32_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamWaitValue64_v2(void *conn) {
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &addr, sizeof(addr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamWaitValue64_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamWriteValue32_v2(void *conn) {
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &addr, sizeof(addr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamWriteValue32_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamWriteValue64_v2(void *conn) {
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &addr, sizeof(addr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamWriteValue64_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuStreamBatchMemOp_v2(void *conn) {
    CUstream stream;
    unsigned int count;
    CUstreamBatchMemOpParams* paramArray;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &paramArray, sizeof(paramArray)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuStreamBatchMemOp_v2(stream, count, paramArray, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &paramArray, sizeof(paramArray)) < 0)
        return -1;
}

int handle_cuFuncGetAttribute(void *conn) {
    int* pi;
    CUfunction_attribute attrib;
    CUfunction hfunc;
    if (rpc_read(conn, &pi, sizeof(pi)) < 0)
        return -1;
    if (rpc_read(conn, &attrib, sizeof(attrib)) < 0)
        return -1;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncGetAttribute(pi, attrib, hfunc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(pi)) < 0)
        return -1;
}

int handle_cuFuncSetAttribute(void *conn) {
    CUfunction hfunc;
    CUfunction_attribute attrib;
    int value;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &attrib, sizeof(attrib)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncSetAttribute(hfunc, attrib, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuFuncSetCacheConfig(void *conn) {
    CUfunction hfunc;
    CUfunc_cache config;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncSetCacheConfig(hfunc, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuFuncSetSharedMemConfig(void *conn) {
    CUfunction hfunc;
    CUsharedconfig config;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncSetSharedMemConfig(hfunc, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuFuncGetModule(void *conn) {
    CUmodule* hmod;
    CUfunction hfunc;
    if (rpc_read(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncGetModule(hmod, hfunc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &hmod, sizeof(hmod)) < 0)
        return -1;
}

int handle_cuLaunchKernel(void *conn) {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void** kernelParams;
    void** extra;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    if (rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0)
        return -1;
    if (rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0)
        return -1;
    if (rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0)
        return -1;
    if (rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0)
        return -1;
    if (rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0)
        return -1;
    if (rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0)
        return -1;
    if (rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &kernelParams, sizeof(kernelParams)) < 0)
        return -1;
    if (rpc_read(conn, &extra, sizeof(extra)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuLaunchKernelEx(void *conn) {
    const CUlaunchConfig* config;
    CUfunction f;
    void** kernelParams;
    void** extra;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    if (rpc_read(conn, &kernelParams, sizeof(kernelParams)) < 0)
        return -1;
    if (rpc_read(conn, &extra, sizeof(extra)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchKernelEx(config, f, kernelParams, extra);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuLaunchCooperativeKernel(void *conn) {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void** kernelParams;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    if (rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0)
        return -1;
    if (rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0)
        return -1;
    if (rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0)
        return -1;
    if (rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0)
        return -1;
    if (rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0)
        return -1;
    if (rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0)
        return -1;
    if (rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &kernelParams, sizeof(kernelParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &kernelParams, sizeof(kernelParams)) < 0)
        return -1;
}

int handle_cuLaunchCooperativeKernelMultiDevice(void *conn) {
    CUDA_LAUNCH_PARAMS* launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
    if (rpc_read(conn, &launchParamsList, sizeof(launchParamsList)) < 0)
        return -1;
    if (rpc_read(conn, &numDevices, sizeof(numDevices)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &launchParamsList, sizeof(launchParamsList)) < 0)
        return -1;
}

int handle_cuLaunchHostFunc(void *conn) {
    CUstream hStream;
    CUhostFn fn;
    void* userData;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &fn, sizeof(fn)) < 0)
        return -1;
    if (rpc_read(conn, &userData, sizeof(userData)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchHostFunc(hStream, fn, userData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(userData)) < 0)
        return -1;
}

int handle_cuFuncSetBlockShape(void *conn) {
    CUfunction hfunc;
    int x;
    int y;
    int z;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &x, sizeof(x)) < 0)
        return -1;
    if (rpc_read(conn, &y, sizeof(y)) < 0)
        return -1;
    if (rpc_read(conn, &z, sizeof(z)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncSetBlockShape(hfunc, x, y, z);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuFuncSetSharedSize(void *conn) {
    CUfunction hfunc;
    unsigned int bytes;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuFuncSetSharedSize(hfunc, bytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuParamSetSize(void *conn) {
    CUfunction hfunc;
    unsigned int numbytes;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &numbytes, sizeof(numbytes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuParamSetSize(hfunc, numbytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuParamSeti(void *conn) {
    CUfunction hfunc;
    int offset;
    unsigned int value;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuParamSeti(hfunc, offset, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuParamSetf(void *conn) {
    CUfunction hfunc;
    int offset;
    float value;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuParamSetf(hfunc, offset, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuParamSetv(void *conn) {
    CUfunction hfunc;
    int offset;
    void* ptr;
    unsigned int numbytes;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &numbytes, sizeof(numbytes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuParamSetv(hfunc, offset, ptr, numbytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cuLaunch(void *conn) {
    CUfunction f;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunch(f);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuLaunchGrid(void *conn) {
    CUfunction f;
    int grid_width;
    int grid_height;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    if (rpc_read(conn, &grid_width, sizeof(grid_width)) < 0)
        return -1;
    if (rpc_read(conn, &grid_height, sizeof(grid_height)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchGrid(f, grid_width, grid_height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuLaunchGridAsync(void *conn) {
    CUfunction f;
    int grid_width;
    int grid_height;
    CUstream hStream;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    if (rpc_read(conn, &grid_width, sizeof(grid_width)) < 0)
        return -1;
    if (rpc_read(conn, &grid_height, sizeof(grid_height)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuLaunchGridAsync(f, grid_width, grid_height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuParamSetTexRef(void *conn) {
    CUfunction hfunc;
    int texunit;
    CUtexref hTexRef;
    if (rpc_read(conn, &hfunc, sizeof(hfunc)) < 0)
        return -1;
    if (rpc_read(conn, &texunit, sizeof(texunit)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuParamSetTexRef(hfunc, texunit, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphCreate(void *conn) {
    CUgraph* phGraph;
    unsigned int flags;
    if (rpc_read(conn, &phGraph, sizeof(phGraph)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphCreate(phGraph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraph, sizeof(phGraph)) < 0)
        return -1;
}

int handle_cuGraphAddKernelNode_v2(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_KERNEL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphKernelNodeGetParams_v2(void *conn) {
    CUgraphNode hNode;
    CUDA_KERNEL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphKernelNodeGetParams_v2(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphKernelNodeSetParams_v2(void *conn) {
    CUgraphNode hNode;
    const CUDA_KERNEL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphKernelNodeSetParams_v2(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddMemcpyNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_MEMCPY3D* copyParams;
    CUcontext ctx;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &copyParams, sizeof(copyParams)) < 0)
        return -1;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &copyParams, sizeof(copyParams)) < 0)
        return -1;
}

int handle_cuGraphMemcpyNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_MEMCPY3D* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphMemcpyNodeGetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphMemcpyNodeSetParams(void *conn) {
    CUgraphNode hNode;
    const CUDA_MEMCPY3D* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphMemcpyNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddMemsetNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_MEMSET_NODE_PARAMS* memsetParams;
    CUcontext ctx;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &memsetParams, sizeof(memsetParams)) < 0)
        return -1;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &memsetParams, sizeof(memsetParams)) < 0)
        return -1;
}

int handle_cuGraphMemsetNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_MEMSET_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphMemsetNodeGetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphMemsetNodeSetParams(void *conn) {
    CUgraphNode hNode;
    const CUDA_MEMSET_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphMemsetNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddHostNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_HOST_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphHostNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_HOST_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphHostNodeGetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphHostNodeSetParams(void *conn) {
    CUgraphNode hNode;
    const CUDA_HOST_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphHostNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddChildGraphNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    CUgraph childGraph;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &childGraph, sizeof(childGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cuGraphChildGraphNodeGetGraph(void *conn) {
    CUgraphNode hNode;
    CUgraph* phGraph;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &phGraph, sizeof(phGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphChildGraphNodeGetGraph(hNode, phGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraph, sizeof(phGraph)) < 0)
        return -1;
}

int handle_cuGraphAddEmptyNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cuGraphAddEventRecordNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    CUevent event;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cuGraphEventRecordNodeGetEvent(void *conn) {
    CUgraphNode hNode;
    CUevent* event_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphEventRecordNodeGetEvent(hNode, event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
}

int handle_cuGraphEventRecordNodeSetEvent(void *conn) {
    CUgraphNode hNode;
    CUevent event;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphEventRecordNodeSetEvent(hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphAddEventWaitNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    CUevent event;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cuGraphEventWaitNodeGetEvent(void *conn) {
    CUgraphNode hNode;
    CUevent* event_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphEventWaitNodeGetEvent(hNode, event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
}

int handle_cuGraphEventWaitNodeSetEvent(void *conn) {
    CUgraphNode hNode;
    CUevent event;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphEventWaitNodeSetEvent(hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphAddExternalSemaphoresSignalNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphExternalSemaphoresSignalNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
}

int handle_cuGraphExternalSemaphoresSignalNodeSetParams(void *conn) {
    CUgraphNode hNode;
    const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddExternalSemaphoresWaitNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphExternalSemaphoresWaitNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
}

int handle_cuGraphExternalSemaphoresWaitNodeSetParams(void *conn) {
    CUgraphNode hNode;
    const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddBatchMemOpNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphBatchMemOpNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams_out, sizeof(nodeParams_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams_out, sizeof(nodeParams_out)) < 0)
        return -1;
}

int handle_cuGraphBatchMemOpNodeSetParams(void *conn) {
    CUgraphNode hNode;
    const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphBatchMemOpNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphExecBatchMemOpNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphAddMemAllocNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphMemAllocNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUDA_MEM_ALLOC_NODE_PARAMS* params_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphMemAllocNodeGetParams(hNode, params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
}

int handle_cuGraphAddMemFreeNode(void *conn) {
    CUgraphNode* phGraphNode;
    CUgraph hGraph;
    const CUgraphNode* dependencies;
    size_t numDependencies;
    CUdeviceptr dptr;
    if (rpc_read(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphNode, sizeof(phGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cuGraphMemFreeNodeGetParams(void *conn) {
    CUgraphNode hNode;
    CUdeviceptr* dptr_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &dptr_out, sizeof(dptr_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphMemFreeNodeGetParams(hNode, dptr_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr_out, sizeof(dptr_out)) < 0)
        return -1;
}

int handle_cuDeviceGraphMemTrim(void *conn) {
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGraphMemTrim(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDeviceGetGraphMemAttribute(void *conn) {
    CUdevice device;
    CUgraphMem_attribute attr;
    void* value;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetGraphMemAttribute(device, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuDeviceSetGraphMemAttribute(void *conn) {
    CUdevice device;
    CUgraphMem_attribute attr;
    void* value;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceSetGraphMemAttribute(device, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuGraphClone(void *conn) {
    CUgraph* phGraphClone;
    CUgraph originalGraph;
    if (rpc_read(conn, &phGraphClone, sizeof(phGraphClone)) < 0)
        return -1;
    if (rpc_read(conn, &originalGraph, sizeof(originalGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphClone(phGraphClone, originalGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphClone, sizeof(phGraphClone)) < 0)
        return -1;
}

int handle_cuGraphNodeFindInClone(void *conn) {
    CUgraphNode* phNode;
    CUgraphNode hOriginalNode;
    CUgraph hClonedGraph;
    if (rpc_read(conn, &phNode, sizeof(phNode)) < 0)
        return -1;
    if (rpc_read(conn, &hOriginalNode, sizeof(hOriginalNode)) < 0)
        return -1;
    if (rpc_read(conn, &hClonedGraph, sizeof(hClonedGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phNode, sizeof(phNode)) < 0)
        return -1;
}

int handle_cuGraphNodeGetType(void *conn) {
    CUgraphNode hNode;
    CUgraphNodeType* type;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &type, sizeof(type)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphNodeGetType(hNode, type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &type, sizeof(type)) < 0)
        return -1;
}

int handle_cuGraphGetNodes(void *conn) {
    CUgraph hGraph;
    CUgraphNode* nodes;
    size_t* numNodes;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &nodes, sizeof(nodes)) < 0)
        return -1;
    if (rpc_read(conn, &numNodes, sizeof(numNodes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphGetNodes(hGraph, nodes, numNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodes, sizeof(nodes)) < 0)
        return -1;
    if (rpc_write(conn, &numNodes, sizeof(numNodes)) < 0)
        return -1;
}

int handle_cuGraphGetRootNodes(void *conn) {
    CUgraph hGraph;
    CUgraphNode* rootNodes;
    size_t* numRootNodes;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &rootNodes, sizeof(rootNodes)) < 0)
        return -1;
    if (rpc_read(conn, &numRootNodes, sizeof(numRootNodes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &rootNodes, sizeof(rootNodes)) < 0)
        return -1;
    if (rpc_write(conn, &numRootNodes, sizeof(numRootNodes)) < 0)
        return -1;
}

int handle_cuGraphGetEdges(void *conn) {
    CUgraph hGraph;
    CUgraphNode* from;
    CUgraphNode* to;
    size_t* numEdges;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_read(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_read(conn, &numEdges, sizeof(numEdges)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphGetEdges(hGraph, from, to, numEdges);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_write(conn, &numEdges, sizeof(numEdges)) < 0)
        return -1;
}

int handle_cuGraphNodeGetDependencies(void *conn) {
    CUgraphNode hNode;
    CUgraphNode* dependencies;
    size_t* numDependencies;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphNodeGetDependencies(hNode, dependencies, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_write(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
}

int handle_cuGraphNodeGetDependentNodes(void *conn) {
    CUgraphNode hNode;
    CUgraphNode* dependentNodes;
    size_t* numDependentNodes;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &dependentNodes, sizeof(dependentNodes)) < 0)
        return -1;
    if (rpc_read(conn, &numDependentNodes, sizeof(numDependentNodes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependentNodes, sizeof(dependentNodes)) < 0)
        return -1;
    if (rpc_write(conn, &numDependentNodes, sizeof(numDependentNodes)) < 0)
        return -1;
}

int handle_cuGraphAddDependencies(void *conn) {
    CUgraph hGraph;
    const CUgraphNode* from;
    const CUgraphNode* to;
    size_t numDependencies;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_read(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphAddDependencies(hGraph, from, to, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(to)) < 0)
        return -1;
}

int handle_cuGraphRemoveDependencies(void *conn) {
    CUgraph hGraph;
    const CUgraphNode* from;
    const CUgraphNode* to;
    size_t numDependencies;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_read(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphRemoveDependencies(hGraph, from, to, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(to)) < 0)
        return -1;
}

int handle_cuGraphDestroyNode(void *conn) {
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphDestroyNode(hNode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphInstantiateWithFlags(void *conn) {
    CUgraphExec* phGraphExec;
    CUgraph hGraph;
    unsigned long long flags;
    if (rpc_read(conn, &phGraphExec, sizeof(phGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphExec, sizeof(phGraphExec)) < 0)
        return -1;
}

int handle_cuGraphInstantiateWithParams(void *conn) {
    CUgraphExec* phGraphExec;
    CUgraph hGraph;
    CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams;
    if (rpc_read(conn, &phGraphExec, sizeof(phGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &instantiateParams, sizeof(instantiateParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphExec, sizeof(phGraphExec)) < 0)
        return -1;
    if (rpc_write(conn, &instantiateParams, sizeof(instantiateParams)) < 0)
        return -1;
}

int handle_cuGraphExecGetFlags(void *conn) {
    CUgraphExec hGraphExec;
    cuuint64_t* flags;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecGetFlags(hGraphExec, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cuGraphExecKernelNodeSetParams_v2(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_KERNEL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphExecMemcpyNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_MEMCPY3D* copyParams;
    CUcontext ctx;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &copyParams, sizeof(copyParams)) < 0)
        return -1;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &copyParams, sizeof(copyParams)) < 0)
        return -1;
}

int handle_cuGraphExecMemsetNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_MEMSET_NODE_PARAMS* memsetParams;
    CUcontext ctx;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &memsetParams, sizeof(memsetParams)) < 0)
        return -1;
    if (rpc_read(conn, &ctx, sizeof(ctx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memsetParams, sizeof(memsetParams)) < 0)
        return -1;
}

int handle_cuGraphExecHostNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_HOST_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphExecChildGraphNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    CUgraph childGraph;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &childGraph, sizeof(childGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphExecEventRecordNodeSetEvent(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    CUevent event;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphExecEventWaitNodeSetEvent(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    CUevent event;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphExecExternalSemaphoresSignalNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphExecExternalSemaphoresWaitNodeSetParams(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cuGraphNodeSetEnabled(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    unsigned int isEnabled;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphNodeGetEnabled(void *conn) {
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    unsigned int* isEnabled;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
}

int handle_cuGraphUpload(void *conn) {
    CUgraphExec hGraphExec;
    CUstream hStream;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphUpload(hGraphExec, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphLaunch(void *conn) {
    CUgraphExec hGraphExec;
    CUstream hStream;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphLaunch(hGraphExec, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphExecDestroy(void *conn) {
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecDestroy(hGraphExec);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphDestroy(void *conn) {
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphDestroy(hGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphExecUpdate_v2(void *conn) {
    CUgraphExec hGraphExec;
    CUgraph hGraph;
    CUgraphExecUpdateResultInfo* resultInfo;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &resultInfo, sizeof(resultInfo)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resultInfo, sizeof(resultInfo)) < 0)
        return -1;
}

int handle_cuGraphKernelNodeCopyAttributes(void *conn) {
    CUgraphNode dst;
    CUgraphNode src;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphKernelNodeCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphKernelNodeGetAttribute(void *conn) {
    CUgraphNode hNode;
    CUkernelNodeAttrID attr;
    CUkernelNodeAttrValue* value_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphKernelNodeGetAttribute(hNode, attr, value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
}

int handle_cuGraphKernelNodeSetAttribute(void *conn) {
    CUgraphNode hNode;
    CUkernelNodeAttrID attr;
    const CUkernelNodeAttrValue* value;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphKernelNodeSetAttribute(hNode, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuGraphDebugDotPrint(void *conn) {
    CUgraph hGraph;
    const char* path;
    unsigned int flags;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &path, sizeof(path)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphDebugDotPrint(hGraph, path, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &path, sizeof(path)) < 0)
        return -1;
}

int handle_cuUserObjectCreate(void *conn) {
    CUuserObject* object_out;
    void* ptr;
    CUhostFn destroy;
    unsigned int initialRefcount;
    unsigned int flags;
    if (rpc_read(conn, &object_out, sizeof(object_out)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &destroy, sizeof(destroy)) < 0)
        return -1;
    if (rpc_read(conn, &initialRefcount, sizeof(initialRefcount)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &object_out, sizeof(object_out)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cuUserObjectRetain(void *conn) {
    CUuserObject object;
    unsigned int count;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuUserObjectRetain(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuUserObjectRelease(void *conn) {
    CUuserObject object;
    unsigned int count;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuUserObjectRelease(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphRetainUserObject(void *conn) {
    CUgraph graph;
    CUuserObject object;
    unsigned int count;
    unsigned int flags;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphRetainUserObject(graph, object, count, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphReleaseUserObject(void *conn) {
    CUgraph graph;
    CUuserObject object;
    unsigned int count;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphReleaseUserObject(graph, object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(void *conn) {
    int* numBlocks;
    CUfunction func;
    int blockSize;
    size_t dynamicSMemSize;
    if (rpc_read(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &blockSize, sizeof(blockSize)) < 0)
        return -1;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *conn) {
    int* numBlocks;
    CUfunction func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
    if (rpc_read(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &blockSize, sizeof(blockSize)) < 0)
        return -1;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
}

int handle_cuOccupancyAvailableDynamicSMemPerBlock(void *conn) {
    size_t* dynamicSmemSize;
    CUfunction func;
    int numBlocks;
    int blockSize;
    if (rpc_read(conn, &dynamicSmemSize, sizeof(dynamicSmemSize)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_read(conn, &blockSize, sizeof(blockSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dynamicSmemSize, sizeof(dynamicSmemSize)) < 0)
        return -1;
}

int handle_cuOccupancyMaxPotentialClusterSize(void *conn) {
    int* clusterSize;
    CUfunction func;
    const CUlaunchConfig* config;
    if (rpc_read(conn, &clusterSize, sizeof(clusterSize)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuOccupancyMaxPotentialClusterSize(clusterSize, func, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clusterSize, sizeof(clusterSize)) < 0)
        return -1;
    if (rpc_write(conn, &config, sizeof(config)) < 0)
        return -1;
}

int handle_cuOccupancyMaxActiveClusters(void *conn) {
    int* numClusters;
    CUfunction func;
    const CUlaunchConfig* config;
    if (rpc_read(conn, &numClusters, sizeof(numClusters)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuOccupancyMaxActiveClusters(numClusters, func, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numClusters, sizeof(numClusters)) < 0)
        return -1;
    if (rpc_write(conn, &config, sizeof(config)) < 0)
        return -1;
}

int handle_cuTexRefSetArray(void *conn) {
    CUtexref hTexRef;
    CUarray hArray;
    unsigned int Flags;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetArray(hTexRef, hArray, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetMipmappedArray(void *conn) {
    CUtexref hTexRef;
    CUmipmappedArray hMipmappedArray;
    unsigned int Flags;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &hMipmappedArray, sizeof(hMipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetAddress_v2(void *conn) {
    size_t* ByteOffset;
    CUtexref hTexRef;
    CUdeviceptr dptr;
    size_t bytes;
    if (rpc_read(conn, &ByteOffset, sizeof(ByteOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &bytes, sizeof(bytes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ByteOffset, sizeof(ByteOffset)) < 0)
        return -1;
}

int handle_cuTexRefSetAddress2D_v3(void *conn) {
    CUtexref hTexRef;
    const CUDA_ARRAY_DESCRIPTOR* desc;
    CUdeviceptr dptr;
    size_t Pitch;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    if (rpc_read(conn, &Pitch, sizeof(Pitch)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
}

int handle_cuTexRefSetFormat(void *conn) {
    CUtexref hTexRef;
    CUarray_format fmt;
    int NumPackedComponents;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &fmt, sizeof(fmt)) < 0)
        return -1;
    if (rpc_read(conn, &NumPackedComponents, sizeof(NumPackedComponents)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetAddressMode(void *conn) {
    CUtexref hTexRef;
    int dim;
    CUaddress_mode am;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &dim, sizeof(dim)) < 0)
        return -1;
    if (rpc_read(conn, &am, sizeof(am)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetAddressMode(hTexRef, dim, am);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetFilterMode(void *conn) {
    CUtexref hTexRef;
    CUfilter_mode fm;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &fm, sizeof(fm)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetFilterMode(hTexRef, fm);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetMipmapFilterMode(void *conn) {
    CUtexref hTexRef;
    CUfilter_mode fm;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &fm, sizeof(fm)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetMipmapFilterMode(hTexRef, fm);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetMipmapLevelBias(void *conn) {
    CUtexref hTexRef;
    float bias;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &bias, sizeof(bias)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetMipmapLevelBias(hTexRef, bias);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetMipmapLevelClamp(void *conn) {
    CUtexref hTexRef;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &minMipmapLevelClamp, sizeof(minMipmapLevelClamp)) < 0)
        return -1;
    if (rpc_read(conn, &maxMipmapLevelClamp, sizeof(maxMipmapLevelClamp)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetMaxAnisotropy(void *conn) {
    CUtexref hTexRef;
    unsigned int maxAniso;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &maxAniso, sizeof(maxAniso)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetMaxAnisotropy(hTexRef, maxAniso);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefSetBorderColor(void *conn) {
    CUtexref hTexRef;
    float* pBorderColor;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &pBorderColor, sizeof(pBorderColor)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetBorderColor(hTexRef, pBorderColor);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pBorderColor, sizeof(pBorderColor)) < 0)
        return -1;
}

int handle_cuTexRefSetFlags(void *conn) {
    CUtexref hTexRef;
    unsigned int Flags;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefSetFlags(hTexRef, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexRefGetAddress_v2(void *conn) {
    CUdeviceptr* pdptr;
    CUtexref hTexRef;
    if (rpc_read(conn, &pdptr, sizeof(pdptr)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetAddress_v2(pdptr, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pdptr, sizeof(pdptr)) < 0)
        return -1;
}

int handle_cuTexRefGetArray(void *conn) {
    CUarray* phArray;
    CUtexref hTexRef;
    if (rpc_read(conn, &phArray, sizeof(phArray)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetArray(phArray, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phArray, sizeof(phArray)) < 0)
        return -1;
}

int handle_cuTexRefGetMipmappedArray(void *conn) {
    CUmipmappedArray* phMipmappedArray;
    CUtexref hTexRef;
    if (rpc_read(conn, &phMipmappedArray, sizeof(phMipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phMipmappedArray, sizeof(phMipmappedArray)) < 0)
        return -1;
}

int handle_cuTexRefGetAddressMode(void *conn) {
    CUaddress_mode* pam;
    CUtexref hTexRef;
    int dim;
    if (rpc_read(conn, &pam, sizeof(pam)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    if (rpc_read(conn, &dim, sizeof(dim)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetAddressMode(pam, hTexRef, dim);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pam, sizeof(pam)) < 0)
        return -1;
}

int handle_cuTexRefGetFilterMode(void *conn) {
    CUfilter_mode* pfm;
    CUtexref hTexRef;
    if (rpc_read(conn, &pfm, sizeof(pfm)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetFilterMode(pfm, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pfm, sizeof(pfm)) < 0)
        return -1;
}

int handle_cuTexRefGetFormat(void *conn) {
    CUarray_format* pFormat;
    int* pNumChannels;
    CUtexref hTexRef;
    if (rpc_read(conn, &pFormat, sizeof(pFormat)) < 0)
        return -1;
    if (rpc_read(conn, &pNumChannels, sizeof(pNumChannels)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetFormat(pFormat, pNumChannels, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFormat, sizeof(pFormat)) < 0)
        return -1;
    if (rpc_write(conn, &pNumChannels, sizeof(pNumChannels)) < 0)
        return -1;
}

int handle_cuTexRefGetMipmapFilterMode(void *conn) {
    CUfilter_mode* pfm;
    CUtexref hTexRef;
    if (rpc_read(conn, &pfm, sizeof(pfm)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetMipmapFilterMode(pfm, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pfm, sizeof(pfm)) < 0)
        return -1;
}

int handle_cuTexRefGetMipmapLevelBias(void *conn) {
    float* pbias;
    CUtexref hTexRef;
    if (rpc_read(conn, &pbias, sizeof(pbias)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetMipmapLevelBias(pbias, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pbias, sizeof(pbias)) < 0)
        return -1;
}

int handle_cuTexRefGetMipmapLevelClamp(void *conn) {
    float* pminMipmapLevelClamp;
    float* pmaxMipmapLevelClamp;
    CUtexref hTexRef;
    if (rpc_read(conn, &pminMipmapLevelClamp, sizeof(pminMipmapLevelClamp)) < 0)
        return -1;
    if (rpc_read(conn, &pmaxMipmapLevelClamp, sizeof(pmaxMipmapLevelClamp)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pminMipmapLevelClamp, sizeof(pminMipmapLevelClamp)) < 0)
        return -1;
    if (rpc_write(conn, &pmaxMipmapLevelClamp, sizeof(pmaxMipmapLevelClamp)) < 0)
        return -1;
}

int handle_cuTexRefGetMaxAnisotropy(void *conn) {
    int* pmaxAniso;
    CUtexref hTexRef;
    if (rpc_read(conn, &pmaxAniso, sizeof(pmaxAniso)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pmaxAniso, sizeof(pmaxAniso)) < 0)
        return -1;
}

int handle_cuTexRefGetBorderColor(void *conn) {
    float* pBorderColor;
    CUtexref hTexRef;
    if (rpc_read(conn, &pBorderColor, sizeof(pBorderColor)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetBorderColor(pBorderColor, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pBorderColor, sizeof(pBorderColor)) < 0)
        return -1;
}

int handle_cuTexRefGetFlags(void *conn) {
    unsigned int* pFlags;
    CUtexref hTexRef;
    if (rpc_read(conn, &pFlags, sizeof(pFlags)) < 0)
        return -1;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefGetFlags(pFlags, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFlags, sizeof(pFlags)) < 0)
        return -1;
}

int handle_cuTexRefCreate(void *conn) {
    CUtexref* pTexRef;
    if (rpc_read(conn, &pTexRef, sizeof(pTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefCreate(pTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexRef, sizeof(pTexRef)) < 0)
        return -1;
}

int handle_cuTexRefDestroy(void *conn) {
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(hTexRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexRefDestroy(hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuSurfRefSetArray(void *conn) {
    CUsurfref hSurfRef;
    CUarray hArray;
    unsigned int Flags;
    if (rpc_read(conn, &hSurfRef, sizeof(hSurfRef)) < 0)
        return -1;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuSurfRefSetArray(hSurfRef, hArray, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuSurfRefGetArray(void *conn) {
    CUarray* phArray;
    CUsurfref hSurfRef;
    if (rpc_read(conn, &phArray, sizeof(phArray)) < 0)
        return -1;
    if (rpc_read(conn, &hSurfRef, sizeof(hSurfRef)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuSurfRefGetArray(phArray, hSurfRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phArray, sizeof(phArray)) < 0)
        return -1;
}

int handle_cuTexObjectCreate(void *conn) {
    CUtexObject* pTexObject;
    const CUDA_RESOURCE_DESC* pResDesc;
    const CUDA_TEXTURE_DESC* pTexDesc;
    const CUDA_RESOURCE_VIEW_DESC* pResViewDesc;
    if (rpc_read(conn, &pTexObject, sizeof(pTexObject)) < 0)
        return -1;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_read(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
    if (rpc_read(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexObject, sizeof(pTexObject)) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_write(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
    if (rpc_write(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
}

int handle_cuTexObjectDestroy(void *conn) {
    CUtexObject texObject;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexObjectDestroy(texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuTexObjectGetResourceDesc(void *conn) {
    CUDA_RESOURCE_DESC* pResDesc;
    CUtexObject texObject;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexObjectGetResourceDesc(pResDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
}

int handle_cuTexObjectGetTextureDesc(void *conn) {
    CUDA_TEXTURE_DESC* pTexDesc;
    CUtexObject texObject;
    if (rpc_read(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexObjectGetTextureDesc(pTexDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
}

int handle_cuTexObjectGetResourceViewDesc(void *conn) {
    CUDA_RESOURCE_VIEW_DESC* pResViewDesc;
    CUtexObject texObject;
    if (rpc_read(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTexObjectGetResourceViewDesc(pResViewDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
}

int handle_cuSurfObjectCreate(void *conn) {
    CUsurfObject* pSurfObject;
    const CUDA_RESOURCE_DESC* pResDesc;
    if (rpc_read(conn, &pSurfObject, sizeof(pSurfObject)) < 0)
        return -1;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuSurfObjectCreate(pSurfObject, pResDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSurfObject, sizeof(pSurfObject)) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
}

int handle_cuSurfObjectDestroy(void *conn) {
    CUsurfObject surfObject;
    if (rpc_read(conn, &surfObject, sizeof(surfObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuSurfObjectDestroy(surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuSurfObjectGetResourceDesc(void *conn) {
    CUDA_RESOURCE_DESC* pResDesc;
    CUsurfObject surfObject;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_read(conn, &surfObject, sizeof(surfObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuSurfObjectGetResourceDesc(pResDesc, surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
}

int handle_cuTensorMapEncodeTiled(void *conn) {
    CUtensorMap* tensorMap;
    CUtensorMapDataType tensorDataType;
    cuuint32_t tensorRank;
    void* globalAddress;
    const cuuint64_t* globalDim;
    const cuuint64_t* globalStrides;
    const cuuint32_t* boxDim;
    const cuuint32_t* elementStrides;
    CUtensorMapInterleave interleave;
    CUtensorMapSwizzle swizzle;
    CUtensorMapL2promotion l2Promotion;
    CUtensorMapFloatOOBfill oobFill;
    if (rpc_read(conn, &tensorMap, sizeof(tensorMap)) < 0)
        return -1;
    if (rpc_read(conn, &tensorDataType, sizeof(tensorDataType)) < 0)
        return -1;
    if (rpc_read(conn, &tensorRank, sizeof(tensorRank)) < 0)
        return -1;
    if (rpc_read(conn, &globalAddress, sizeof(globalAddress)) < 0)
        return -1;
    if (rpc_read(conn, &globalDim, sizeof(globalDim)) < 0)
        return -1;
    if (rpc_read(conn, &globalStrides, sizeof(globalStrides)) < 0)
        return -1;
    if (rpc_read(conn, &boxDim, sizeof(boxDim)) < 0)
        return -1;
    if (rpc_read(conn, &elementStrides, sizeof(elementStrides)) < 0)
        return -1;
    if (rpc_read(conn, &interleave, sizeof(interleave)) < 0)
        return -1;
    if (rpc_read(conn, &swizzle, sizeof(swizzle)) < 0)
        return -1;
    if (rpc_read(conn, &l2Promotion, sizeof(l2Promotion)) < 0)
        return -1;
    if (rpc_read(conn, &oobFill, sizeof(oobFill)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &tensorMap, sizeof(tensorMap)) < 0)
        return -1;
    if (rpc_write(conn, &globalAddress, sizeof(globalAddress)) < 0)
        return -1;
    if (rpc_write(conn, &globalDim, sizeof(globalDim)) < 0)
        return -1;
    if (rpc_write(conn, &globalStrides, sizeof(globalStrides)) < 0)
        return -1;
    if (rpc_write(conn, &boxDim, sizeof(boxDim)) < 0)
        return -1;
    if (rpc_write(conn, &elementStrides, sizeof(elementStrides)) < 0)
        return -1;
}

int handle_cuTensorMapEncodeIm2col(void *conn) {
    CUtensorMap* tensorMap;
    CUtensorMapDataType tensorDataType;
    cuuint32_t tensorRank;
    void* globalAddress;
    const cuuint64_t* globalDim;
    const cuuint64_t* globalStrides;
    const int* pixelBoxLowerCorner;
    const int* pixelBoxUpperCorner;
    cuuint32_t channelsPerPixel;
    cuuint32_t pixelsPerColumn;
    const cuuint32_t* elementStrides;
    CUtensorMapInterleave interleave;
    CUtensorMapSwizzle swizzle;
    CUtensorMapL2promotion l2Promotion;
    CUtensorMapFloatOOBfill oobFill;
    if (rpc_read(conn, &tensorMap, sizeof(tensorMap)) < 0)
        return -1;
    if (rpc_read(conn, &tensorDataType, sizeof(tensorDataType)) < 0)
        return -1;
    if (rpc_read(conn, &tensorRank, sizeof(tensorRank)) < 0)
        return -1;
    if (rpc_read(conn, &globalAddress, sizeof(globalAddress)) < 0)
        return -1;
    if (rpc_read(conn, &globalDim, sizeof(globalDim)) < 0)
        return -1;
    if (rpc_read(conn, &globalStrides, sizeof(globalStrides)) < 0)
        return -1;
    if (rpc_read(conn, &pixelBoxLowerCorner, sizeof(pixelBoxLowerCorner)) < 0)
        return -1;
    if (rpc_read(conn, &pixelBoxUpperCorner, sizeof(pixelBoxUpperCorner)) < 0)
        return -1;
    if (rpc_read(conn, &channelsPerPixel, sizeof(channelsPerPixel)) < 0)
        return -1;
    if (rpc_read(conn, &pixelsPerColumn, sizeof(pixelsPerColumn)) < 0)
        return -1;
    if (rpc_read(conn, &elementStrides, sizeof(elementStrides)) < 0)
        return -1;
    if (rpc_read(conn, &interleave, sizeof(interleave)) < 0)
        return -1;
    if (rpc_read(conn, &swizzle, sizeof(swizzle)) < 0)
        return -1;
    if (rpc_read(conn, &l2Promotion, sizeof(l2Promotion)) < 0)
        return -1;
    if (rpc_read(conn, &oobFill, sizeof(oobFill)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTensorMapEncodeIm2col(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &tensorMap, sizeof(tensorMap)) < 0)
        return -1;
    if (rpc_write(conn, &globalAddress, sizeof(globalAddress)) < 0)
        return -1;
    if (rpc_write(conn, &globalDim, sizeof(globalDim)) < 0)
        return -1;
    if (rpc_write(conn, &globalStrides, sizeof(globalStrides)) < 0)
        return -1;
    if (rpc_write(conn, &pixelBoxLowerCorner, sizeof(pixelBoxLowerCorner)) < 0)
        return -1;
    if (rpc_write(conn, &pixelBoxUpperCorner, sizeof(pixelBoxUpperCorner)) < 0)
        return -1;
    if (rpc_write(conn, &elementStrides, sizeof(elementStrides)) < 0)
        return -1;
}

int handle_cuTensorMapReplaceAddress(void *conn) {
    CUtensorMap* tensorMap;
    void* globalAddress;
    if (rpc_read(conn, &tensorMap, sizeof(tensorMap)) < 0)
        return -1;
    if (rpc_read(conn, &globalAddress, sizeof(globalAddress)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuTensorMapReplaceAddress(tensorMap, globalAddress);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &tensorMap, sizeof(tensorMap)) < 0)
        return -1;
    if (rpc_write(conn, &globalAddress, sizeof(globalAddress)) < 0)
        return -1;
}

int handle_cuDeviceCanAccessPeer(void *conn) {
    int* canAccessPeer;
    CUdevice dev;
    CUdevice peerDev;
    if (rpc_read(conn, &canAccessPeer, sizeof(canAccessPeer)) < 0)
        return -1;
    if (rpc_read(conn, &dev, sizeof(dev)) < 0)
        return -1;
    if (rpc_read(conn, &peerDev, sizeof(peerDev)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &canAccessPeer, sizeof(canAccessPeer)) < 0)
        return -1;
}

int handle_cuCtxEnablePeerAccess(void *conn) {
    CUcontext peerContext;
    unsigned int Flags;
    if (rpc_read(conn, &peerContext, sizeof(peerContext)) < 0)
        return -1;
    if (rpc_read(conn, &Flags, sizeof(Flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxEnablePeerAccess(peerContext, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuCtxDisablePeerAccess(void *conn) {
    CUcontext peerContext;
    if (rpc_read(conn, &peerContext, sizeof(peerContext)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuCtxDisablePeerAccess(peerContext);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuDeviceGetP2PAttribute(void *conn) {
    int* value;
    CUdevice_P2PAttribute attrib;
    CUdevice srcDevice;
    CUdevice dstDevice;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &attrib, sizeof(attrib)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cuGraphicsUnregisterResource(void *conn) {
    CUgraphicsResource resource;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsUnregisterResource(resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphicsSubResourceGetMappedArray(void *conn) {
    CUarray* pArray;
    CUgraphicsResource resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
    if (rpc_read(conn, &pArray, sizeof(pArray)) < 0)
        return -1;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    if (rpc_read(conn, &arrayIndex, sizeof(arrayIndex)) < 0)
        return -1;
    if (rpc_read(conn, &mipLevel, sizeof(mipLevel)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pArray, sizeof(pArray)) < 0)
        return -1;
}

int handle_cuGraphicsResourceGetMappedMipmappedArray(void *conn) {
    CUmipmappedArray* pMipmappedArray;
    CUgraphicsResource resource;
    if (rpc_read(conn, &pMipmappedArray, sizeof(pMipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pMipmappedArray, sizeof(pMipmappedArray)) < 0)
        return -1;
}

int handle_cuGraphicsResourceGetMappedPointer_v2(void *conn) {
    CUdeviceptr* pDevPtr;
    size_t* pSize;
    CUgraphicsResource resource;
    if (rpc_read(conn, &pDevPtr, sizeof(pDevPtr)) < 0)
        return -1;
    if (rpc_read(conn, &pSize, sizeof(pSize)) < 0)
        return -1;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDevPtr, sizeof(pDevPtr)) < 0)
        return -1;
    if (rpc_write(conn, &pSize, sizeof(pSize)) < 0)
        return -1;
}

int handle_cuGraphicsResourceSetMapFlags_v2(void *conn) {
    CUgraphicsResource resource;
    unsigned int flags;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsResourceSetMapFlags_v2(resource, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cuGraphicsMapResources(void *conn) {
    unsigned int count;
    CUgraphicsResource* resources;
    CUstream hStream;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &resources, sizeof(resources)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsMapResources(count, resources, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(resources)) < 0)
        return -1;
}

int handle_cuGraphicsUnmapResources(void *conn) {
    unsigned int count;
    CUgraphicsResource* resources;
    CUstream hStream;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &resources, sizeof(resources)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGraphicsUnmapResources(count, resources, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(resources)) < 0)
        return -1;
}

int handle_cuGetProcAddress_v2(void *conn) {
    const char* symbol;
    void** pfn;
    int cudaVersion;
    cuuint64_t flags;
    CUdriverProcAddressQueryResult* symbolStatus;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &pfn, sizeof(pfn)) < 0)
        return -1;
    if (rpc_read(conn, &cudaVersion, sizeof(cudaVersion)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &symbolStatus, sizeof(symbolStatus)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &pfn, sizeof(pfn)) < 0)
        return -1;
    if (rpc_write(conn, &symbolStatus, sizeof(symbolStatus)) < 0)
        return -1;
}

int handle_cuGetExportTable(void *conn) {
    const void** ppExportTable;
    const CUuuid* pExportTableId;
    if (rpc_read(conn, &ppExportTable, sizeof(ppExportTable)) < 0)
        return -1;
    if (rpc_read(conn, &pExportTableId, sizeof(pExportTableId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult return_value = cuGetExportTable(ppExportTable, pExportTableId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ppExportTable, sizeof(ppExportTable)) < 0)
        return -1;
    if (rpc_write(conn, &pExportTableId, sizeof(pExportTableId)) < 0)
        return -1;
}

int handle_cudaDeviceReset(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceReset();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceSynchronize(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceSynchronize();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceSetLimit(void *conn) {
    enum cudaLimit limit;
    size_t value;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceGetLimit(void *conn) {
    size_t* pValue;
    enum cudaLimit limit;
    if (rpc_read(conn, &pValue, sizeof(pValue)) < 0)
        return -1;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetLimit(pValue, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pValue, sizeof(pValue)) < 0)
        return -1;
}

int handle_cudaDeviceGetTexture1DLinearMaxWidth(void *conn) {
    size_t* maxWidthInElements;
    const struct cudaChannelFormatDesc* fmtDesc;
    int device;
    if (rpc_read(conn, &maxWidthInElements, sizeof(maxWidthInElements)) < 0)
        return -1;
    if (rpc_read(conn, &fmtDesc, sizeof(fmtDesc)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxWidthInElements, sizeof(maxWidthInElements)) < 0)
        return -1;
    if (rpc_write(conn, &fmtDesc, sizeof(fmtDesc)) < 0)
        return -1;
}

int handle_cudaDeviceGetCacheConfig(void *conn) {
    enum cudaFuncCache* pCacheConfig;
    if (rpc_read(conn, &pCacheConfig, sizeof(pCacheConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetCacheConfig(pCacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCacheConfig, sizeof(pCacheConfig)) < 0)
        return -1;
}

int handle_cudaDeviceGetStreamPriorityRange(void *conn) {
    int* leastPriority;
    int* greatestPriority;
    if (rpc_read(conn, &leastPriority, sizeof(leastPriority)) < 0)
        return -1;
    if (rpc_read(conn, &greatestPriority, sizeof(greatestPriority)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &leastPriority, sizeof(leastPriority)) < 0)
        return -1;
    if (rpc_write(conn, &greatestPriority, sizeof(greatestPriority)) < 0)
        return -1;
}

int handle_cudaDeviceSetCacheConfig(void *conn) {
    enum cudaFuncCache cacheConfig;
    if (rpc_read(conn, &cacheConfig, sizeof(cacheConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceSetCacheConfig(cacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceGetSharedMemConfig(void *conn) {
    enum cudaSharedMemConfig* pConfig;
    if (rpc_read(conn, &pConfig, sizeof(pConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetSharedMemConfig(pConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pConfig, sizeof(pConfig)) < 0)
        return -1;
}

int handle_cudaDeviceSetSharedMemConfig(void *conn) {
    enum cudaSharedMemConfig config;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceSetSharedMemConfig(config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceGetByPCIBusId(void *conn) {
    int* device;
    const char* pciBusId;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetByPCIBusId(device, pciBusId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_write(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
}

int handle_cudaDeviceGetPCIBusId(void *conn) {
    char* pciBusId;
    int len;
    int device;
    if (rpc_read(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
    if (rpc_read(conn, &len, sizeof(len)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetPCIBusId(pciBusId, len, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pciBusId, sizeof(pciBusId)) < 0)
        return -1;
}

int handle_cudaIpcGetEventHandle(void *conn) {
    cudaIpcEventHandle_t* handle;
    cudaEvent_t event;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaIpcGetEventHandle(handle, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
}

int handle_cudaIpcOpenEventHandle(void *conn) {
    cudaEvent_t* event;
    cudaIpcEventHandle_t handle;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaIpcOpenEventHandle(event, handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event, sizeof(event)) < 0)
        return -1;
}

int handle_cudaIpcGetMemHandle(void *conn) {
    cudaIpcMemHandle_t* handle;
    void* devPtr;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaIpcGetMemHandle(handle, devPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaIpcOpenMemHandle(void *conn) {
    void** devPtr;
    cudaIpcMemHandle_t handle;
    unsigned int flags;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &handle, sizeof(handle)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaIpcOpenMemHandle(devPtr, handle, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaIpcCloseMemHandle(void *conn) {
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaIpcCloseMemHandle(devPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaDeviceFlushGPUDirectRDMAWrites(void *conn) {
    enum cudaFlushGPUDirectRDMAWritesTarget target;
    enum cudaFlushGPUDirectRDMAWritesScope scope;
    if (rpc_read(conn, &target, sizeof(target)) < 0)
        return -1;
    if (rpc_read(conn, &scope, sizeof(scope)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceFlushGPUDirectRDMAWrites(target, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaThreadExit(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadExit();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaThreadSynchronize(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadSynchronize();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaThreadSetLimit(void *conn) {
    enum cudaLimit limit;
    size_t value;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaThreadGetLimit(void *conn) {
    size_t* pValue;
    enum cudaLimit limit;
    if (rpc_read(conn, &pValue, sizeof(pValue)) < 0)
        return -1;
    if (rpc_read(conn, &limit, sizeof(limit)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadGetLimit(pValue, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pValue, sizeof(pValue)) < 0)
        return -1;
}

int handle_cudaThreadGetCacheConfig(void *conn) {
    enum cudaFuncCache* pCacheConfig;
    if (rpc_read(conn, &pCacheConfig, sizeof(pCacheConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadGetCacheConfig(pCacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCacheConfig, sizeof(pCacheConfig)) < 0)
        return -1;
}

int handle_cudaThreadSetCacheConfig(void *conn) {
    enum cudaFuncCache cacheConfig;
    if (rpc_read(conn, &cacheConfig, sizeof(cacheConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadSetCacheConfig(cacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetLastError(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetLastError();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaPeekAtLastError(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaPeekAtLastError();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetErrorName(void *conn) {
    cudaError_t error;
    if (rpc_read(conn, &error, sizeof(error)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    const char* return_value = cudaGetErrorName(error);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetErrorString(void *conn) {
    cudaError_t error;
    if (rpc_read(conn, &error, sizeof(error)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    const char* return_value = cudaGetErrorString(error);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetDeviceCount(void *conn) {
    int* count;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetDeviceCount(count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(count)) < 0)
        return -1;
}

int handle_cudaGetDeviceProperties_v2(void *conn) {
    struct cudaDeviceProp* prop;
    int device;
    if (rpc_read(conn, &prop, sizeof(prop)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetDeviceProperties_v2(prop, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(prop)) < 0)
        return -1;
}

int handle_cudaDeviceGetAttribute(void *conn) {
    int* value;
    enum cudaDeviceAttr attr;
    int device;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetAttribute(value, attr, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaDeviceGetDefaultMemPool(void *conn) {
    cudaMemPool_t* memPool;
    int device;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetDefaultMemPool(memPool, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
}

int handle_cudaDeviceSetMemPool(void *conn) {
    int device;
    cudaMemPool_t memPool;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceSetMemPool(device, memPool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceGetMemPool(void *conn) {
    cudaMemPool_t* memPool;
    int device;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetMemPool(memPool, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
}

int handle_cudaDeviceGetNvSciSyncAttributes(void *conn) {
    void* nvSciSyncAttrList;
    int device;
    int flags;
    if (rpc_read(conn, &nvSciSyncAttrList, sizeof(nvSciSyncAttrList)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nvSciSyncAttrList, sizeof(nvSciSyncAttrList)) < 0)
        return -1;
}

int handle_cudaDeviceGetP2PAttribute(void *conn) {
    int* value;
    enum cudaDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaChooseDevice(void *conn) {
    int* device;
    const struct cudaDeviceProp* prop;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &prop, sizeof(prop)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaChooseDevice(device, prop);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(prop)) < 0)
        return -1;
}

int handle_cudaInitDevice(void *conn) {
    int device;
    unsigned int deviceFlags;
    unsigned int flags;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &deviceFlags, sizeof(deviceFlags)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaInitDevice(device, deviceFlags, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaSetDevice(void *conn) {
    int device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaSetDevice(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetDevice(void *conn) {
    int* device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetDevice(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(device)) < 0)
        return -1;
}

int handle_cudaSetValidDevices(void *conn) {
    int* device_arr;
    int len;
    if (rpc_read(conn, &device_arr, sizeof(device_arr)) < 0)
        return -1;
    if (rpc_read(conn, &len, sizeof(len)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaSetValidDevices(device_arr, len);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device_arr, sizeof(device_arr)) < 0)
        return -1;
}

int handle_cudaSetDeviceFlags(void *conn) {
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaSetDeviceFlags(flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetDeviceFlags(void *conn) {
    unsigned int* flags;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetDeviceFlags(flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cudaStreamCreate(void *conn) {
    cudaStream_t* pStream;
    if (rpc_read(conn, &pStream, sizeof(pStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamCreate(pStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pStream, sizeof(pStream)) < 0)
        return -1;
}

int handle_cudaStreamCreateWithFlags(void *conn) {
    cudaStream_t* pStream;
    unsigned int flags;
    if (rpc_read(conn, &pStream, sizeof(pStream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamCreateWithFlags(pStream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pStream, sizeof(pStream)) < 0)
        return -1;
}

int handle_cudaStreamCreateWithPriority(void *conn) {
    cudaStream_t* pStream;
    unsigned int flags;
    int priority;
    if (rpc_read(conn, &pStream, sizeof(pStream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &priority, sizeof(priority)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamCreateWithPriority(pStream, flags, priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pStream, sizeof(pStream)) < 0)
        return -1;
}

int handle_cudaStreamGetPriority(void *conn) {
    cudaStream_t hStream;
    int* priority;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &priority, sizeof(priority)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamGetPriority(hStream, priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &priority, sizeof(priority)) < 0)
        return -1;
}

int handle_cudaStreamGetFlags(void *conn) {
    cudaStream_t hStream;
    unsigned int* flags;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamGetFlags(hStream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cudaStreamGetId(void *conn) {
    cudaStream_t hStream;
    unsigned long long* streamId;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &streamId, sizeof(streamId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamGetId(hStream, streamId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &streamId, sizeof(streamId)) < 0)
        return -1;
}

int handle_cudaCtxResetPersistingL2Cache(void *conn) {
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaCtxResetPersistingL2Cache();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaStreamCopyAttributes(void *conn) {
    cudaStream_t dst;
    cudaStream_t src;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaStreamGetAttribute(void *conn) {
    cudaStream_t hStream;
    cudaLaunchAttributeID attr;
    cudaLaunchAttributeValue* value_out;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamGetAttribute(hStream, attr, value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
}

int handle_cudaStreamSetAttribute(void *conn) {
    cudaStream_t hStream;
    cudaLaunchAttributeID attr;
    const cudaLaunchAttributeValue* value;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamSetAttribute(hStream, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaStreamDestroy(void *conn) {
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamDestroy(stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaStreamWaitEvent(void *conn) {
    cudaStream_t stream;
    cudaEvent_t event;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamWaitEvent(stream, event, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaStreamAddCallback(void *conn) {
    cudaStream_t stream;
    cudaStreamCallback_t callback;
    void* userData;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &callback, sizeof(callback)) < 0)
        return -1;
    if (rpc_read(conn, &userData, sizeof(userData)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamAddCallback(stream, callback, userData, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(userData)) < 0)
        return -1;
}

int handle_cudaStreamSynchronize(void *conn) {
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamSynchronize(stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaStreamQuery(void *conn) {
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamQuery(stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaStreamAttachMemAsync(void *conn) {
    cudaStream_t stream;
    void* devPtr;
    size_t length;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &length, sizeof(length)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamAttachMemAsync(stream, devPtr, length, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaStreamBeginCapture(void *conn) {
    cudaStream_t stream;
    enum cudaStreamCaptureMode mode;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamBeginCapture(stream, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaThreadExchangeStreamCaptureMode(void *conn) {
    enum cudaStreamCaptureMode* mode;
    if (rpc_read(conn, &mode, sizeof(mode)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaThreadExchangeStreamCaptureMode(mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(mode)) < 0)
        return -1;
}

int handle_cudaStreamEndCapture(void *conn) {
    cudaStream_t stream;
    cudaGraph_t* pGraph;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &pGraph, sizeof(pGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamEndCapture(stream, pGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraph, sizeof(pGraph)) < 0)
        return -1;
}

int handle_cudaStreamIsCapturing(void *conn) {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus* pCaptureStatus;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &pCaptureStatus, sizeof(pCaptureStatus)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamIsCapturing(stream, pCaptureStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCaptureStatus, sizeof(pCaptureStatus)) < 0)
        return -1;
}

int handle_cudaStreamGetCaptureInfo_v2(void *conn) {
    cudaStream_t stream;
    enum cudaStreamCaptureStatus* captureStatus_out;
    unsigned long long* id_out;
    cudaGraph_t* graph_out;
    const cudaGraphNode_t** dependencies_out;
    size_t* numDependencies_out;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &captureStatus_out, sizeof(captureStatus_out)) < 0)
        return -1;
    if (rpc_read(conn, &id_out, sizeof(id_out)) < 0)
        return -1;
    if (rpc_read(conn, &graph_out, sizeof(graph_out)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies_out, sizeof(dependencies_out)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies_out, sizeof(numDependencies_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &captureStatus_out, sizeof(captureStatus_out)) < 0)
        return -1;
    if (rpc_write(conn, &id_out, sizeof(id_out)) < 0)
        return -1;
    if (rpc_write(conn, &graph_out, sizeof(graph_out)) < 0)
        return -1;
    if (rpc_write(conn, &dependencies_out, sizeof(dependencies_out)) < 0)
        return -1;
    if (rpc_write(conn, &numDependencies_out, sizeof(numDependencies_out)) < 0)
        return -1;
}

int handle_cudaStreamUpdateCaptureDependencies(void *conn) {
    cudaStream_t stream;
    cudaGraphNode_t* dependencies;
    size_t numDependencies;
    unsigned int flags;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(dependencies)) < 0)
        return -1;
}

int handle_cudaEventCreate(void *conn) {
    cudaEvent_t* event;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventCreate(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event, sizeof(event)) < 0)
        return -1;
}

int handle_cudaEventCreateWithFlags(void *conn) {
    cudaEvent_t* event;
    unsigned int flags;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventCreateWithFlags(event, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event, sizeof(event)) < 0)
        return -1;
}

int handle_cudaEventRecord(void *conn) {
    cudaEvent_t event;
    cudaStream_t stream;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventRecord(event, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaEventRecordWithFlags(void *conn) {
    cudaEvent_t event;
    cudaStream_t stream;
    unsigned int flags;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventRecordWithFlags(event, stream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaEventQuery(void *conn) {
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventQuery(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaEventSynchronize(void *conn) {
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventSynchronize(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaEventDestroy(void *conn) {
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventDestroy(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaEventElapsedTime(void *conn) {
    float* ms;
    cudaEvent_t start;
    cudaEvent_t end;
    if (rpc_read(conn, &ms, sizeof(ms)) < 0)
        return -1;
    if (rpc_read(conn, &start, sizeof(start)) < 0)
        return -1;
    if (rpc_read(conn, &end, sizeof(end)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaEventElapsedTime(ms, start, end);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ms, sizeof(ms)) < 0)
        return -1;
}

int handle_cudaImportExternalMemory(void *conn) {
    cudaExternalMemory_t* extMem_out;
    const struct cudaExternalMemoryHandleDesc* memHandleDesc;
    if (rpc_read(conn, &extMem_out, sizeof(extMem_out)) < 0)
        return -1;
    if (rpc_read(conn, &memHandleDesc, sizeof(memHandleDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaImportExternalMemory(extMem_out, memHandleDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extMem_out, sizeof(extMem_out)) < 0)
        return -1;
    if (rpc_write(conn, &memHandleDesc, sizeof(memHandleDesc)) < 0)
        return -1;
}

int handle_cudaExternalMemoryGetMappedBuffer(void *conn) {
    void** devPtr;
    cudaExternalMemory_t extMem;
    const struct cudaExternalMemoryBufferDesc* bufferDesc;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &extMem, sizeof(extMem)) < 0)
        return -1;
    if (rpc_read(conn, &bufferDesc, sizeof(bufferDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_write(conn, &bufferDesc, sizeof(bufferDesc)) < 0)
        return -1;
}

int handle_cudaExternalMemoryGetMappedMipmappedArray(void *conn) {
    cudaMipmappedArray_t* mipmap;
    cudaExternalMemory_t extMem;
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc;
    if (rpc_read(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    if (rpc_read(conn, &extMem, sizeof(extMem)) < 0)
        return -1;
    if (rpc_read(conn, &mipmapDesc, sizeof(mipmapDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    if (rpc_write(conn, &mipmapDesc, sizeof(mipmapDesc)) < 0)
        return -1;
}

int handle_cudaDestroyExternalMemory(void *conn) {
    cudaExternalMemory_t extMem;
    if (rpc_read(conn, &extMem, sizeof(extMem)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDestroyExternalMemory(extMem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaImportExternalSemaphore(void *conn) {
    cudaExternalSemaphore_t* extSem_out;
    const struct cudaExternalSemaphoreHandleDesc* semHandleDesc;
    if (rpc_read(conn, &extSem_out, sizeof(extSem_out)) < 0)
        return -1;
    if (rpc_read(conn, &semHandleDesc, sizeof(semHandleDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaImportExternalSemaphore(extSem_out, semHandleDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extSem_out, sizeof(extSem_out)) < 0)
        return -1;
    if (rpc_write(conn, &semHandleDesc, sizeof(semHandleDesc)) < 0)
        return -1;
}

int handle_cudaSignalExternalSemaphoresAsync_v2(void *conn) {
    const cudaExternalSemaphore_t* extSemArray;
    const struct cudaExternalSemaphoreSignalParams* paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
    if (rpc_read(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_read(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
    if (rpc_read(conn, &numExtSems, sizeof(numExtSems)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_write(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
}

int handle_cudaWaitExternalSemaphoresAsync_v2(void *conn) {
    const cudaExternalSemaphore_t* extSemArray;
    const struct cudaExternalSemaphoreWaitParams* paramsArray;
    unsigned int numExtSems;
    cudaStream_t stream;
    if (rpc_read(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_read(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
    if (rpc_read(conn, &numExtSems, sizeof(numExtSems)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &extSemArray, sizeof(extSemArray)) < 0)
        return -1;
    if (rpc_write(conn, &paramsArray, sizeof(paramsArray)) < 0)
        return -1;
}

int handle_cudaDestroyExternalSemaphore(void *conn) {
    cudaExternalSemaphore_t extSem;
    if (rpc_read(conn, &extSem, sizeof(extSem)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDestroyExternalSemaphore(extSem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaLaunchKernel(void *conn) {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** args;
    size_t sharedMem;
    cudaStream_t stream;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &gridDim, sizeof(gridDim)) < 0)
        return -1;
    if (rpc_read(conn, &blockDim, sizeof(blockDim)) < 0)
        return -1;
    if (rpc_read(conn, &args, sizeof(args)) < 0)
        return -1;
    if (rpc_read(conn, &sharedMem, sizeof(sharedMem)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_write(conn, &args, sizeof(args)) < 0)
        return -1;
}

int handle_cudaLaunchKernelExC(void *conn) {
    const cudaLaunchConfig_t* config;
    const void* func;
    void** args;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &args, sizeof(args)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaLaunchKernelExC(config, func, args);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &config, sizeof(config)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_write(conn, &args, sizeof(args)) < 0)
        return -1;
}

int handle_cudaLaunchCooperativeKernel(void *conn) {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    void** args;
    size_t sharedMem;
    cudaStream_t stream;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &gridDim, sizeof(gridDim)) < 0)
        return -1;
    if (rpc_read(conn, &blockDim, sizeof(blockDim)) < 0)
        return -1;
    if (rpc_read(conn, &args, sizeof(args)) < 0)
        return -1;
    if (rpc_read(conn, &sharedMem, sizeof(sharedMem)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_write(conn, &args, sizeof(args)) < 0)
        return -1;
}

int handle_cudaLaunchCooperativeKernelMultiDevice(void *conn) {
    struct cudaLaunchParams* launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
    if (rpc_read(conn, &launchParamsList, sizeof(launchParamsList)) < 0)
        return -1;
    if (rpc_read(conn, &numDevices, sizeof(numDevices)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &launchParamsList, sizeof(launchParamsList)) < 0)
        return -1;
}

int handle_cudaFuncSetCacheConfig(void *conn) {
    const void* func;
    enum cudaFuncCache cacheConfig;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &cacheConfig, sizeof(cacheConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFuncSetCacheConfig(func, cacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaFuncSetSharedMemConfig(void *conn) {
    const void* func;
    enum cudaSharedMemConfig config;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &config, sizeof(config)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFuncSetSharedMemConfig(func, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaFuncGetAttributes(void *conn) {
    struct cudaFuncAttributes* attr;
    const void* func;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFuncGetAttributes(attr, func);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaFuncSetAttribute(void *conn) {
    const void* func;
    enum cudaFuncAttribute attr;
    int value;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFuncSetAttribute(func, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaSetDoubleForDevice(void *conn) {
    double* d;
    if (rpc_read(conn, &d, sizeof(d)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaSetDoubleForDevice(d);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &d, sizeof(d)) < 0)
        return -1;
}

int handle_cudaSetDoubleForHost(void *conn) {
    double* d;
    if (rpc_read(conn, &d, sizeof(d)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaSetDoubleForHost(d);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &d, sizeof(d)) < 0)
        return -1;
}

int handle_cudaLaunchHostFunc(void *conn) {
    cudaStream_t stream;
    cudaHostFn_t fn;
    void* userData;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    if (rpc_read(conn, &fn, sizeof(fn)) < 0)
        return -1;
    if (rpc_read(conn, &userData, sizeof(userData)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaLaunchHostFunc(stream, fn, userData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(userData)) < 0)
        return -1;
}

int handle_cudaOccupancyMaxActiveBlocksPerMultiprocessor(void *conn) {
    int* numBlocks;
    const void* func;
    int blockSize;
    size_t dynamicSMemSize;
    if (rpc_read(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &blockSize, sizeof(blockSize)) < 0)
        return -1;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaOccupancyAvailableDynamicSMemPerBlock(void *conn) {
    size_t* dynamicSmemSize;
    const void* func;
    int numBlocks;
    int blockSize;
    if (rpc_read(conn, &dynamicSmemSize, sizeof(dynamicSmemSize)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_read(conn, &blockSize, sizeof(blockSize)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dynamicSmemSize, sizeof(dynamicSmemSize)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *conn) {
    int* numBlocks;
    const void* func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
    if (rpc_read(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &blockSize, sizeof(blockSize)) < 0)
        return -1;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numBlocks, sizeof(numBlocks)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
}

int handle_cudaOccupancyMaxPotentialClusterSize(void *conn) {
    int* clusterSize;
    const void* func;
    const cudaLaunchConfig_t* launchConfig;
    if (rpc_read(conn, &clusterSize, sizeof(clusterSize)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &launchConfig, sizeof(launchConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaOccupancyMaxPotentialClusterSize(clusterSize, func, launchConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clusterSize, sizeof(clusterSize)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_write(conn, &launchConfig, sizeof(launchConfig)) < 0)
        return -1;
}

int handle_cudaOccupancyMaxActiveClusters(void *conn) {
    int* numClusters;
    const void* func;
    const cudaLaunchConfig_t* launchConfig;
    if (rpc_read(conn, &numClusters, sizeof(numClusters)) < 0)
        return -1;
    if (rpc_read(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_read(conn, &launchConfig, sizeof(launchConfig)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaOccupancyMaxActiveClusters(numClusters, func, launchConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numClusters, sizeof(numClusters)) < 0)
        return -1;
    if (rpc_write(conn, &func, sizeof(func)) < 0)
        return -1;
    if (rpc_write(conn, &launchConfig, sizeof(launchConfig)) < 0)
        return -1;
}

int handle_cudaMallocManaged(void *conn) {
    void** devPtr;
    size_t size;
    unsigned int flags;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocManaged(devPtr, size, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMalloc(void *conn) {
    void** devPtr;
    size_t size;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMalloc(devPtr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMallocHost(void *conn) {
    void** ptr;
    size_t size;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocHost(ptr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaMallocPitch(void *conn) {
    void** devPtr;
    size_t* pitch;
    size_t width;
    size_t height;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &pitch, sizeof(pitch)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocPitch(devPtr, pitch, width, height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_write(conn, &pitch, sizeof(pitch)) < 0)
        return -1;
}

int handle_cudaMallocArray(void *conn) {
    cudaArray_t* array;
    const struct cudaChannelFormatDesc* desc;
    size_t width;
    size_t height;
    unsigned int flags;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocArray(array, desc, width, height, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
}

int handle_cudaFree(void *conn) {
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFree(devPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaFreeHost(void *conn) {
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFreeHost(ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaFreeArray(void *conn) {
    cudaArray_t array;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFreeArray(array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaFreeMipmappedArray(void *conn) {
    cudaMipmappedArray_t mipmappedArray;
    if (rpc_read(conn, &mipmappedArray, sizeof(mipmappedArray)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFreeMipmappedArray(mipmappedArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaHostAlloc(void *conn) {
    void** pHost;
    size_t size;
    unsigned int flags;
    if (rpc_read(conn, &pHost, sizeof(pHost)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaHostAlloc(pHost, size, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHost, sizeof(pHost)) < 0)
        return -1;
}

int handle_cudaHostRegister(void *conn) {
    void* ptr;
    size_t size;
    unsigned int flags;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaHostRegister(ptr, size, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaHostUnregister(void *conn) {
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaHostUnregister(ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaHostGetDevicePointer(void *conn) {
    void** pDevice;
    void* pHost;
    unsigned int flags;
    if (rpc_read(conn, &pDevice, sizeof(pDevice)) < 0)
        return -1;
    if (rpc_read(conn, &pHost, sizeof(pHost)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaHostGetDevicePointer(pDevice, pHost, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDevice, sizeof(pDevice)) < 0)
        return -1;
    if (rpc_write(conn, &pHost, sizeof(pHost)) < 0)
        return -1;
}

int handle_cudaHostGetFlags(void *conn) {
    unsigned int* pFlags;
    void* pHost;
    if (rpc_read(conn, &pFlags, sizeof(pFlags)) < 0)
        return -1;
    if (rpc_read(conn, &pHost, sizeof(pHost)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaHostGetFlags(pFlags, pHost);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFlags, sizeof(pFlags)) < 0)
        return -1;
    if (rpc_write(conn, &pHost, sizeof(pHost)) < 0)
        return -1;
}

int handle_cudaMalloc3D(void *conn) {
    struct cudaPitchedPtr* pitchedDevPtr;
    struct cudaExtent extent;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(pitchedDevPtr)) < 0)
        return -1;
    if (rpc_read(conn, &extent, sizeof(extent)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMalloc3D(pitchedDevPtr, extent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pitchedDevPtr, sizeof(pitchedDevPtr)) < 0)
        return -1;
}

int handle_cudaMalloc3DArray(void *conn) {
    cudaArray_t* array;
    const struct cudaChannelFormatDesc* desc;
    struct cudaExtent extent;
    unsigned int flags;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &extent, sizeof(extent)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMalloc3DArray(array, desc, extent, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
}

int handle_cudaMallocMipmappedArray(void *conn) {
    cudaMipmappedArray_t* mipmappedArray;
    const struct cudaChannelFormatDesc* desc;
    struct cudaExtent extent;
    unsigned int numLevels;
    unsigned int flags;
    if (rpc_read(conn, &mipmappedArray, sizeof(mipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &extent, sizeof(extent)) < 0)
        return -1;
    if (rpc_read(conn, &numLevels, sizeof(numLevels)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mipmappedArray, sizeof(mipmappedArray)) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
}

int handle_cudaGetMipmappedArrayLevel(void *conn) {
    cudaArray_t* levelArray;
    cudaMipmappedArray_const_t mipmappedArray;
    unsigned int level;
    if (rpc_read(conn, &levelArray, sizeof(levelArray)) < 0)
        return -1;
    if (rpc_read(conn, &mipmappedArray, sizeof(mipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &level, sizeof(level)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &levelArray, sizeof(levelArray)) < 0)
        return -1;
}

int handle_cudaMemcpy3D(void *conn) {
    const struct cudaMemcpy3DParms* p;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy3D(p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cudaMemcpy3DPeer(void *conn) {
    const struct cudaMemcpy3DPeerParms* p;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy3DPeer(p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cudaMemcpy3DAsync(void *conn) {
    const struct cudaMemcpy3DParms* p;
    cudaStream_t stream;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy3DAsync(p, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cudaMemcpy3DPeerAsync(void *conn) {
    const struct cudaMemcpy3DPeerParms* p;
    cudaStream_t stream;
    if (rpc_read(conn, &p, sizeof(p)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy3DPeerAsync(p, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(p)) < 0)
        return -1;
}

int handle_cudaMemGetInfo(void *conn) {
    size_t* free;
    size_t* total;
    if (rpc_read(conn, &free, sizeof(free)) < 0)
        return -1;
    if (rpc_read(conn, &total, sizeof(total)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemGetInfo(free, total);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &free, sizeof(free)) < 0)
        return -1;
    if (rpc_write(conn, &total, sizeof(total)) < 0)
        return -1;
}

int handle_cudaArrayGetInfo(void *conn) {
    struct cudaChannelFormatDesc* desc;
    struct cudaExtent* extent;
    unsigned int* flags;
    cudaArray_t array;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &extent, sizeof(extent)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaArrayGetInfo(desc, extent, flags, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_write(conn, &extent, sizeof(extent)) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cudaArrayGetPlane(void *conn) {
    cudaArray_t* pPlaneArray;
    cudaArray_t hArray;
    unsigned int planeIdx;
    if (rpc_read(conn, &pPlaneArray, sizeof(pPlaneArray)) < 0)
        return -1;
    if (rpc_read(conn, &hArray, sizeof(hArray)) < 0)
        return -1;
    if (rpc_read(conn, &planeIdx, sizeof(planeIdx)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaArrayGetPlane(pPlaneArray, hArray, planeIdx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pPlaneArray, sizeof(pPlaneArray)) < 0)
        return -1;
}

int handle_cudaArrayGetMemoryRequirements(void *conn) {
    struct cudaArrayMemoryRequirements* memoryRequirements;
    cudaArray_t array;
    int device;
    if (rpc_read(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaArrayGetMemoryRequirements(memoryRequirements, array, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
}

int handle_cudaMipmappedArrayGetMemoryRequirements(void *conn) {
    struct cudaArrayMemoryRequirements* memoryRequirements;
    cudaMipmappedArray_t mipmap;
    int device;
    if (rpc_read(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
    if (rpc_read(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(memoryRequirements)) < 0)
        return -1;
}

int handle_cudaArrayGetSparseProperties(void *conn) {
    struct cudaArraySparseProperties* sparseProperties;
    cudaArray_t array;
    if (rpc_read(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaArrayGetSparseProperties(sparseProperties, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
}

int handle_cudaMipmappedArrayGetSparseProperties(void *conn) {
    struct cudaArraySparseProperties* sparseProperties;
    cudaMipmappedArray_t mipmap;
    if (rpc_read(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
    if (rpc_read(conn, &mipmap, sizeof(mipmap)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(sparseProperties)) < 0)
        return -1;
}

int handle_cudaMemcpy(void *conn) {
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy(dst, src, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpyPeer(void *conn) {
    void* dst;
    int dstDevice;
    const void* src;
    int srcDevice;
    size_t count;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpy2D(void *conn) {
    void* dst;
    size_t dpitch;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &dpitch, sizeof(dpitch)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &spitch, sizeof(spitch)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpy2DToArray(void *conn) {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &spitch, sizeof(spitch)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpy2DFromArray(void *conn) {
    void* dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &dpitch, sizeof(dpitch)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
}

int handle_cudaMemcpy2DArrayToArray(void *conn) {
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &wOffsetDst, sizeof(wOffsetDst)) < 0)
        return -1;
    if (rpc_read(conn, &hOffsetDst, sizeof(hOffsetDst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &wOffsetSrc, sizeof(wOffsetSrc)) < 0)
        return -1;
    if (rpc_read(conn, &hOffsetSrc, sizeof(hOffsetSrc)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaMemcpyToSymbol(void *conn) {
    const void* symbol;
    const void* src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyToSymbol(symbol, src, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpyFromSymbol(void *conn) {
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaMemcpyAsync(void *conn) {
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyAsync(dst, src, count, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpyPeerAsync(void *conn) {
    void* dst;
    int dstDevice;
    const void* src;
    int srcDevice;
    size_t count;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpy2DAsync(void *conn) {
    void* dst;
    size_t dpitch;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &dpitch, sizeof(dpitch)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &spitch, sizeof(spitch)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpy2DToArrayAsync(void *conn) {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t spitch;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &spitch, sizeof(spitch)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpy2DFromArrayAsync(void *conn) {
    void* dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &dpitch, sizeof(dpitch)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
}

int handle_cudaMemcpyToSymbolAsync(void *conn) {
    const void* symbol;
    const void* src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpyFromSymbolAsync(void *conn) {
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaMemset(void *conn) {
    void* devPtr;
    int value;
    size_t count;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemset(devPtr, value, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemset2D(void *conn) {
    void* devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &pitch, sizeof(pitch)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemset2D(devPtr, pitch, value, width, height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemset3D(void *conn) {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(pitchedDevPtr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &extent, sizeof(extent)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemset3D(pitchedDevPtr, value, extent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaMemsetAsync(void *conn) {
    void* devPtr;
    int value;
    size_t count;
    cudaStream_t stream;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemsetAsync(devPtr, value, count, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemset2DAsync(void *conn) {
    void* devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    cudaStream_t stream;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &pitch, sizeof(pitch)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &width, sizeof(width)) < 0)
        return -1;
    if (rpc_read(conn, &height, sizeof(height)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemset3DAsync(void *conn) {
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    cudaStream_t stream;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(pitchedDevPtr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    if (rpc_read(conn, &extent, sizeof(extent)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetSymbolAddress(void *conn) {
    void** devPtr;
    const void* symbol;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetSymbolAddress(devPtr, symbol);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaGetSymbolSize(void *conn) {
    size_t* size;
    const void* symbol;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetSymbolSize(size, symbol);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaMemPrefetchAsync(void *conn) {
    const void* devPtr;
    size_t count;
    int dstDevice;
    cudaStream_t stream;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemAdvise(void *conn) {
    const void* devPtr;
    size_t count;
    enum cudaMemoryAdvise advice;
    int device;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &advice, sizeof(advice)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemAdvise(devPtr, count, advice, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemRangeGetAttribute(void *conn) {
    void* data;
    size_t dataSize;
    enum cudaMemRangeAttribute attribute;
    const void* devPtr;
    size_t count;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &dataSize, sizeof(dataSize)) < 0)
        return -1;
    if (rpc_read(conn, &attribute, sizeof(attribute)) < 0)
        return -1;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemRangeGetAttributes(void *conn) {
    void** data;
    size_t* dataSizes;
    enum cudaMemRangeAttribute* attributes;
    size_t numAttributes;
    const void* devPtr;
    size_t count;
    if (rpc_read(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_read(conn, &dataSizes, sizeof(dataSizes)) < 0)
        return -1;
    if (rpc_read(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_read(conn, &numAttributes, sizeof(numAttributes)) < 0)
        return -1;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(data)) < 0)
        return -1;
    if (rpc_write(conn, &dataSizes, sizeof(dataSizes)) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemcpyToArray(void *conn) {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpyFromArray(void *conn) {
    void* dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
}

int handle_cudaMemcpyArrayToArray(void *conn) {
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &wOffsetDst, sizeof(wOffsetDst)) < 0)
        return -1;
    if (rpc_read(conn, &hOffsetDst, sizeof(hOffsetDst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &wOffsetSrc, sizeof(wOffsetSrc)) < 0)
        return -1;
    if (rpc_read(conn, &hOffsetSrc, sizeof(hOffsetSrc)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaMemcpyToArrayAsync(void *conn) {
    cudaArray_t dst;
    size_t wOffset;
    size_t hOffset;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaMemcpyFromArrayAsync(void *conn) {
    void* dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &wOffset, sizeof(wOffset)) < 0)
        return -1;
    if (rpc_read(conn, &hOffset, sizeof(hOffset)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
}

int handle_cudaMallocAsync(void *conn) {
    void** devPtr;
    size_t size;
    cudaStream_t hStream;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocAsync(devPtr, size, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaFreeAsync(void *conn) {
    void* devPtr;
    cudaStream_t hStream;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &hStream, sizeof(hStream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaFreeAsync(devPtr, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
}

int handle_cudaMemPoolTrimTo(void *conn) {
    cudaMemPool_t memPool;
    size_t minBytesToKeep;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &minBytesToKeep, sizeof(minBytesToKeep)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolTrimTo(memPool, minBytesToKeep);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaMemPoolSetAttribute(void *conn) {
    cudaMemPool_t memPool;
    enum cudaMemPoolAttr attr;
    void* value;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolSetAttribute(memPool, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaMemPoolGetAttribute(void *conn) {
    cudaMemPool_t memPool;
    enum cudaMemPoolAttr attr;
    void* value;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolGetAttribute(memPool, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaMemPoolSetAccess(void *conn) {
    cudaMemPool_t memPool;
    const struct cudaMemAccessDesc* descList;
    size_t count;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &descList, sizeof(descList)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolSetAccess(memPool, descList, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &descList, sizeof(descList)) < 0)
        return -1;
}

int handle_cudaMemPoolGetAccess(void *conn) {
    enum cudaMemAccessFlags* flags;
    cudaMemPool_t memPool;
    struct cudaMemLocation* location;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &location, sizeof(location)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolGetAccess(flags, memPool, location);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_write(conn, &location, sizeof(location)) < 0)
        return -1;
}

int handle_cudaMemPoolCreate(void *conn) {
    cudaMemPool_t* memPool;
    const struct cudaMemPoolProps* poolProps;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &poolProps, sizeof(poolProps)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolCreate(memPool, poolProps);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_write(conn, &poolProps, sizeof(poolProps)) < 0)
        return -1;
}

int handle_cudaMemPoolDestroy(void *conn) {
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolDestroy(memPool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaMallocFromPoolAsync(void *conn) {
    void** ptr;
    size_t size;
    cudaMemPool_t memPool;
    cudaStream_t stream;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMallocFromPoolAsync(ptr, size, memPool, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaMemPoolExportToShareableHandle(void *conn) {
    void* shareableHandle;
    cudaMemPool_t memPool;
    enum cudaMemAllocationHandleType handleType;
    unsigned int flags;
    if (rpc_read(conn, &shareableHandle, sizeof(shareableHandle)) < 0)
        return -1;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &handleType, sizeof(handleType)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &shareableHandle, sizeof(shareableHandle)) < 0)
        return -1;
}

int handle_cudaMemPoolImportFromShareableHandle(void *conn) {
    cudaMemPool_t* memPool;
    void* shareableHandle;
    enum cudaMemAllocationHandleType handleType;
    unsigned int flags;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &shareableHandle, sizeof(shareableHandle)) < 0)
        return -1;
    if (rpc_read(conn, &handleType, sizeof(handleType)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_write(conn, &shareableHandle, sizeof(shareableHandle)) < 0)
        return -1;
}

int handle_cudaMemPoolExportPointer(void *conn) {
    struct cudaMemPoolPtrExportData* exportData;
    void* ptr;
    if (rpc_read(conn, &exportData, sizeof(exportData)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolExportPointer(exportData, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &exportData, sizeof(exportData)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaMemPoolImportPointer(void *conn) {
    void** ptr;
    cudaMemPool_t memPool;
    struct cudaMemPoolPtrExportData* exportData;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &memPool, sizeof(memPool)) < 0)
        return -1;
    if (rpc_read(conn, &exportData, sizeof(exportData)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaMemPoolImportPointer(ptr, memPool, exportData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_write(conn, &exportData, sizeof(exportData)) < 0)
        return -1;
}

int handle_cudaPointerGetAttributes(void *conn) {
    struct cudaPointerAttributes* attributes;
    const void* ptr;
    if (rpc_read(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaPointerGetAttributes(attributes, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(attributes)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaDeviceCanAccessPeer(void *conn) {
    int* canAccessPeer;
    int device;
    int peerDevice;
    if (rpc_read(conn, &canAccessPeer, sizeof(canAccessPeer)) < 0)
        return -1;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &peerDevice, sizeof(peerDevice)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &canAccessPeer, sizeof(canAccessPeer)) < 0)
        return -1;
}

int handle_cudaDeviceEnablePeerAccess(void *conn) {
    int peerDevice;
    unsigned int flags;
    if (rpc_read(conn, &peerDevice, sizeof(peerDevice)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceEnablePeerAccess(peerDevice, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceDisablePeerAccess(void *conn) {
    int peerDevice;
    if (rpc_read(conn, &peerDevice, sizeof(peerDevice)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceDisablePeerAccess(peerDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphicsUnregisterResource(void *conn) {
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsUnregisterResource(resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphicsResourceSetMapFlags(void *conn) {
    cudaGraphicsResource_t resource;
    unsigned int flags;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsResourceSetMapFlags(resource, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphicsMapResources(void *conn) {
    int count;
    cudaGraphicsResource_t* resources;
    cudaStream_t stream;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &resources, sizeof(resources)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsMapResources(count, resources, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(resources)) < 0)
        return -1;
}

int handle_cudaGraphicsUnmapResources(void *conn) {
    int count;
    cudaGraphicsResource_t* resources;
    cudaStream_t stream;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &resources, sizeof(resources)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsUnmapResources(count, resources, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(resources)) < 0)
        return -1;
}

int handle_cudaGraphicsResourceGetMappedPointer(void *conn) {
    void** devPtr;
    size_t* size;
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_read(conn, &size, sizeof(size)) < 0)
        return -1;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(devPtr)) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(size)) < 0)
        return -1;
}

int handle_cudaGraphicsSubResourceGetMappedArray(void *conn) {
    cudaArray_t* array;
    cudaGraphicsResource_t resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    if (rpc_read(conn, &arrayIndex, sizeof(arrayIndex)) < 0)
        return -1;
    if (rpc_read(conn, &mipLevel, sizeof(mipLevel)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &array, sizeof(array)) < 0)
        return -1;
}

int handle_cudaGraphicsResourceGetMappedMipmappedArray(void *conn) {
    cudaMipmappedArray_t* mipmappedArray;
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &mipmappedArray, sizeof(mipmappedArray)) < 0)
        return -1;
    if (rpc_read(conn, &resource, sizeof(resource)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mipmappedArray, sizeof(mipmappedArray)) < 0)
        return -1;
}

int handle_cudaGetChannelDesc(void *conn) {
    struct cudaChannelFormatDesc* desc;
    cudaArray_const_t array;
    if (rpc_read(conn, &desc, sizeof(desc)) < 0)
        return -1;
    if (rpc_read(conn, &array, sizeof(array)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetChannelDesc(desc, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(desc)) < 0)
        return -1;
}

int handle_cudaCreateChannelDesc(void *conn) {
    int x;
    int y;
    int z;
    int w;
    enum cudaChannelFormatKind f;
    if (rpc_read(conn, &x, sizeof(x)) < 0)
        return -1;
    if (rpc_read(conn, &y, sizeof(y)) < 0)
        return -1;
    if (rpc_read(conn, &z, sizeof(z)) < 0)
        return -1;
    if (rpc_read(conn, &w, sizeof(w)) < 0)
        return -1;
    if (rpc_read(conn, &f, sizeof(f)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    struct cudaChannelFormatDesc return_value = cudaCreateChannelDesc(x, y, z, w, f);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaCreateTextureObject(void *conn) {
    cudaTextureObject_t* pTexObject;
    const struct cudaResourceDesc* pResDesc;
    const struct cudaTextureDesc* pTexDesc;
    const struct cudaResourceViewDesc* pResViewDesc;
    if (rpc_read(conn, &pTexObject, sizeof(pTexObject)) < 0)
        return -1;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_read(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
    if (rpc_read(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexObject, sizeof(pTexObject)) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_write(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
    if (rpc_write(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
}

int handle_cudaDestroyTextureObject(void *conn) {
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDestroyTextureObject(texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetTextureObjectResourceDesc(void *conn) {
    struct cudaResourceDesc* pResDesc;
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetTextureObjectResourceDesc(pResDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
}

int handle_cudaGetTextureObjectTextureDesc(void *conn) {
    struct cudaTextureDesc* pTexDesc;
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetTextureObjectTextureDesc(pTexDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexDesc, sizeof(pTexDesc)) < 0)
        return -1;
}

int handle_cudaGetTextureObjectResourceViewDesc(void *conn) {
    struct cudaResourceViewDesc* pResViewDesc;
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
    if (rpc_read(conn, &texObject, sizeof(texObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResViewDesc, sizeof(pResViewDesc)) < 0)
        return -1;
}

int handle_cudaCreateSurfaceObject(void *conn) {
    cudaSurfaceObject_t* pSurfObject;
    const struct cudaResourceDesc* pResDesc;
    if (rpc_read(conn, &pSurfObject, sizeof(pSurfObject)) < 0)
        return -1;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaCreateSurfaceObject(pSurfObject, pResDesc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSurfObject, sizeof(pSurfObject)) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
}

int handle_cudaDestroySurfaceObject(void *conn) {
    cudaSurfaceObject_t surfObject;
    if (rpc_read(conn, &surfObject, sizeof(surfObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDestroySurfaceObject(surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetSurfaceObjectResourceDesc(void *conn) {
    struct cudaResourceDesc* pResDesc;
    cudaSurfaceObject_t surfObject;
    if (rpc_read(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
    if (rpc_read(conn, &surfObject, sizeof(surfObject)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(pResDesc)) < 0)
        return -1;
}

int handle_cudaDriverGetVersion(void *conn) {
    int* driverVersion;
    if (rpc_read(conn, &driverVersion, sizeof(driverVersion)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDriverGetVersion(driverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &driverVersion, sizeof(driverVersion)) < 0)
        return -1;
}

int handle_cudaRuntimeGetVersion(void *conn) {
    int* runtimeVersion;
    if (rpc_read(conn, &runtimeVersion, sizeof(runtimeVersion)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaRuntimeGetVersion(runtimeVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &runtimeVersion, sizeof(runtimeVersion)) < 0)
        return -1;
}

int handle_cudaGraphCreate(void *conn) {
    cudaGraph_t* pGraph;
    unsigned int flags;
    if (rpc_read(conn, &pGraph, sizeof(pGraph)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphCreate(pGraph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraph, sizeof(pGraph)) < 0)
        return -1;
}

int handle_cudaGraphAddKernelNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const struct cudaKernelNodeParams* pNodeParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphKernelNodeGetParams(void *conn) {
    cudaGraphNode_t node;
    struct cudaKernelNodeParams* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphKernelNodeGetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphKernelNodeSetParams(void *conn) {
    cudaGraphNode_t node;
    const struct cudaKernelNodeParams* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphKernelNodeSetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphKernelNodeCopyAttributes(void *conn) {
    cudaGraphNode_t hSrc;
    cudaGraphNode_t hDst;
    if (rpc_read(conn, &hSrc, sizeof(hSrc)) < 0)
        return -1;
    if (rpc_read(conn, &hDst, sizeof(hDst)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphKernelNodeCopyAttributes(hSrc, hDst);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphKernelNodeGetAttribute(void *conn) {
    cudaGraphNode_t hNode;
    cudaLaunchAttributeID attr;
    cudaLaunchAttributeValue* value_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphKernelNodeGetAttribute(hNode, attr, value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(value_out)) < 0)
        return -1;
}

int handle_cudaGraphKernelNodeSetAttribute(void *conn) {
    cudaGraphNode_t hNode;
    cudaLaunchAttributeID attr;
    const cudaLaunchAttributeValue* value;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphKernelNodeSetAttribute(hNode, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaGraphAddMemcpyNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const struct cudaMemcpy3DParms* pCopyParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &pCopyParams, sizeof(pCopyParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &pCopyParams, sizeof(pCopyParams)) < 0)
        return -1;
}

int handle_cudaGraphAddMemcpyNodeToSymbol(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const void* symbol;
    const void* src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaGraphAddMemcpyNodeFromSymbol(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaGraphAddMemcpyNode1D(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaGraphMemcpyNodeGetParams(void *conn) {
    cudaGraphNode_t node;
    struct cudaMemcpy3DParms* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemcpyNodeGetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphMemcpyNodeSetParams(void *conn) {
    cudaGraphNode_t node;
    const struct cudaMemcpy3DParms* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemcpyNodeSetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphMemcpyNodeSetParamsToSymbol(void *conn) {
    cudaGraphNode_t node;
    const void* symbol;
    const void* src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaGraphMemcpyNodeSetParamsFromSymbol(void *conn) {
    cudaGraphNode_t node;
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaGraphMemcpyNodeSetParams1D(void *conn) {
    cudaGraphNode_t node;
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaGraphAddMemsetNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const struct cudaMemsetParams* pMemsetParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &pMemsetParams, sizeof(pMemsetParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &pMemsetParams, sizeof(pMemsetParams)) < 0)
        return -1;
}

int handle_cudaGraphMemsetNodeGetParams(void *conn) {
    cudaGraphNode_t node;
    struct cudaMemsetParams* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemsetNodeGetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphMemsetNodeSetParams(void *conn) {
    cudaGraphNode_t node;
    const struct cudaMemsetParams* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemsetNodeSetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphAddHostNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const struct cudaHostNodeParams* pNodeParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphHostNodeGetParams(void *conn) {
    cudaGraphNode_t node;
    struct cudaHostNodeParams* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphHostNodeGetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphHostNodeSetParams(void *conn) {
    cudaGraphNode_t node;
    const struct cudaHostNodeParams* pNodeParams;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphHostNodeSetParams(node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphAddChildGraphNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    cudaGraph_t childGraph;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &childGraph, sizeof(childGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
}

int handle_cudaGraphChildGraphNodeGetGraph(void *conn) {
    cudaGraphNode_t node;
    cudaGraph_t* pGraph;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pGraph, sizeof(pGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphChildGraphNodeGetGraph(node, pGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraph, sizeof(pGraph)) < 0)
        return -1;
}

int handle_cudaGraphAddEmptyNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
}

int handle_cudaGraphAddEventRecordNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    cudaEvent_t event;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
}

int handle_cudaGraphEventRecordNodeGetEvent(void *conn) {
    cudaGraphNode_t node;
    cudaEvent_t* event_out;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphEventRecordNodeGetEvent(node, event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
}

int handle_cudaGraphEventRecordNodeSetEvent(void *conn) {
    cudaGraphNode_t node;
    cudaEvent_t event;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphEventRecordNodeSetEvent(node, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphAddEventWaitNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    cudaEvent_t event;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
}

int handle_cudaGraphEventWaitNodeGetEvent(void *conn) {
    cudaGraphNode_t node;
    cudaEvent_t* event_out;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphEventWaitNodeGetEvent(node, event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(event_out)) < 0)
        return -1;
}

int handle_cudaGraphEventWaitNodeSetEvent(void *conn) {
    cudaGraphNode_t node;
    cudaEvent_t event;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphEventWaitNodeSetEvent(node, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphAddExternalSemaphoresSignalNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const struct cudaExternalSemaphoreSignalNodeParams* nodeParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *conn) {
    cudaGraphNode_t hNode;
    struct cudaExternalSemaphoreSignalNodeParams* params_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
}

int handle_cudaGraphExternalSemaphoresSignalNodeSetParams(void *conn) {
    cudaGraphNode_t hNode;
    const struct cudaExternalSemaphoreSignalNodeParams* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphAddExternalSemaphoresWaitNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    const struct cudaExternalSemaphoreWaitNodeParams* nodeParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *conn) {
    cudaGraphNode_t hNode;
    struct cudaExternalSemaphoreWaitNodeParams* params_out;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
}

int handle_cudaGraphExternalSemaphoresWaitNodeSetParams(void *conn) {
    cudaGraphNode_t hNode;
    const struct cudaExternalSemaphoreWaitNodeParams* nodeParams;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphAddMemAllocNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    struct cudaMemAllocNodeParams* nodeParams;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphMemAllocNodeGetParams(void *conn) {
    cudaGraphNode_t node;
    struct cudaMemAllocNodeParams* params_out;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemAllocNodeGetParams(node, params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(params_out)) < 0)
        return -1;
}

int handle_cudaGraphAddMemFreeNode(void *conn) {
    cudaGraphNode_t* pGraphNode;
    cudaGraph_t graph;
    const cudaGraphNode_t* pDependencies;
    size_t numDependencies;
    void* dptr;
    if (rpc_read(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphNode, sizeof(pGraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(dptr)) < 0)
        return -1;
}

int handle_cudaGraphMemFreeNodeGetParams(void *conn) {
    cudaGraphNode_t node;
    void* dptr_out;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &dptr_out, sizeof(dptr_out)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphMemFreeNodeGetParams(node, dptr_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr_out, sizeof(dptr_out)) < 0)
        return -1;
}

int handle_cudaDeviceGraphMemTrim(void *conn) {
    int device;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGraphMemTrim(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaDeviceGetGraphMemAttribute(void *conn) {
    int device;
    enum cudaGraphMemAttributeType attr;
    void* value;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceGetGraphMemAttribute(device, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaDeviceSetGraphMemAttribute(void *conn) {
    int device;
    enum cudaGraphMemAttributeType attr;
    void* value;
    if (rpc_read(conn, &device, sizeof(device)) < 0)
        return -1;
    if (rpc_read(conn, &attr, sizeof(attr)) < 0)
        return -1;
    if (rpc_read(conn, &value, sizeof(value)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaDeviceSetGraphMemAttribute(device, attr, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(value)) < 0)
        return -1;
}

int handle_cudaGraphClone(void *conn) {
    cudaGraph_t* pGraphClone;
    cudaGraph_t originalGraph;
    if (rpc_read(conn, &pGraphClone, sizeof(pGraphClone)) < 0)
        return -1;
    if (rpc_read(conn, &originalGraph, sizeof(originalGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphClone(pGraphClone, originalGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphClone, sizeof(pGraphClone)) < 0)
        return -1;
}

int handle_cudaGraphNodeFindInClone(void *conn) {
    cudaGraphNode_t* pNode;
    cudaGraphNode_t originalNode;
    cudaGraph_t clonedGraph;
    if (rpc_read(conn, &pNode, sizeof(pNode)) < 0)
        return -1;
    if (rpc_read(conn, &originalNode, sizeof(originalNode)) < 0)
        return -1;
    if (rpc_read(conn, &clonedGraph, sizeof(clonedGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNode, sizeof(pNode)) < 0)
        return -1;
}

int handle_cudaGraphNodeGetType(void *conn) {
    cudaGraphNode_t node;
    enum cudaGraphNodeType* pType;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pType, sizeof(pType)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphNodeGetType(node, pType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pType, sizeof(pType)) < 0)
        return -1;
}

int handle_cudaGraphGetNodes(void *conn) {
    cudaGraph_t graph;
    cudaGraphNode_t* nodes;
    size_t* numNodes;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &nodes, sizeof(nodes)) < 0)
        return -1;
    if (rpc_read(conn, &numNodes, sizeof(numNodes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphGetNodes(graph, nodes, numNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodes, sizeof(nodes)) < 0)
        return -1;
    if (rpc_write(conn, &numNodes, sizeof(numNodes)) < 0)
        return -1;
}

int handle_cudaGraphGetRootNodes(void *conn) {
    cudaGraph_t graph;
    cudaGraphNode_t* pRootNodes;
    size_t* pNumRootNodes;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &pRootNodes, sizeof(pRootNodes)) < 0)
        return -1;
    if (rpc_read(conn, &pNumRootNodes, sizeof(pNumRootNodes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pRootNodes, sizeof(pRootNodes)) < 0)
        return -1;
    if (rpc_write(conn, &pNumRootNodes, sizeof(pNumRootNodes)) < 0)
        return -1;
}

int handle_cudaGraphGetEdges(void *conn) {
    cudaGraph_t graph;
    cudaGraphNode_t* from;
    cudaGraphNode_t* to;
    size_t* numEdges;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_read(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_read(conn, &numEdges, sizeof(numEdges)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphGetEdges(graph, from, to, numEdges);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_write(conn, &numEdges, sizeof(numEdges)) < 0)
        return -1;
}

int handle_cudaGraphNodeGetDependencies(void *conn) {
    cudaGraphNode_t node;
    cudaGraphNode_t* pDependencies;
    size_t* pNumDependencies;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_read(conn, &pNumDependencies, sizeof(pNumDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(pDependencies)) < 0)
        return -1;
    if (rpc_write(conn, &pNumDependencies, sizeof(pNumDependencies)) < 0)
        return -1;
}

int handle_cudaGraphNodeGetDependentNodes(void *conn) {
    cudaGraphNode_t node;
    cudaGraphNode_t* pDependentNodes;
    size_t* pNumDependentNodes;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pDependentNodes, sizeof(pDependentNodes)) < 0)
        return -1;
    if (rpc_read(conn, &pNumDependentNodes, sizeof(pNumDependentNodes)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDependentNodes, sizeof(pDependentNodes)) < 0)
        return -1;
    if (rpc_write(conn, &pNumDependentNodes, sizeof(pNumDependentNodes)) < 0)
        return -1;
}

int handle_cudaGraphAddDependencies(void *conn) {
    cudaGraph_t graph;
    const cudaGraphNode_t* from;
    const cudaGraphNode_t* to;
    size_t numDependencies;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_read(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphAddDependencies(graph, from, to, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(to)) < 0)
        return -1;
}

int handle_cudaGraphRemoveDependencies(void *conn) {
    cudaGraph_t graph;
    const cudaGraphNode_t* from;
    const cudaGraphNode_t* to;
    size_t numDependencies;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_read(conn, &to, sizeof(to)) < 0)
        return -1;
    if (rpc_read(conn, &numDependencies, sizeof(numDependencies)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphRemoveDependencies(graph, from, to, numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(from)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(to)) < 0)
        return -1;
}

int handle_cudaGraphDestroyNode(void *conn) {
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphDestroyNode(node);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphInstantiate(void *conn) {
    cudaGraphExec_t* pGraphExec;
    cudaGraph_t graph;
    unsigned long long flags;
    if (rpc_read(conn, &pGraphExec, sizeof(pGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphInstantiate(pGraphExec, graph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphExec, sizeof(pGraphExec)) < 0)
        return -1;
}

int handle_cudaGraphInstantiateWithFlags(void *conn) {
    cudaGraphExec_t* pGraphExec;
    cudaGraph_t graph;
    unsigned long long flags;
    if (rpc_read(conn, &pGraphExec, sizeof(pGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphInstantiateWithFlags(pGraphExec, graph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphExec, sizeof(pGraphExec)) < 0)
        return -1;
}

int handle_cudaGraphInstantiateWithParams(void *conn) {
    cudaGraphExec_t* pGraphExec;
    cudaGraph_t graph;
    cudaGraphInstantiateParams* instantiateParams;
    if (rpc_read(conn, &pGraphExec, sizeof(pGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &instantiateParams, sizeof(instantiateParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphExec, sizeof(pGraphExec)) < 0)
        return -1;
    if (rpc_write(conn, &instantiateParams, sizeof(instantiateParams)) < 0)
        return -1;
}

int handle_cudaGraphExecGetFlags(void *conn) {
    cudaGraphExec_t graphExec;
    unsigned long long* flags;
    if (rpc_read(conn, &graphExec, sizeof(graphExec)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecGetFlags(graphExec, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(flags)) < 0)
        return -1;
}

int handle_cudaGraphExecKernelNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaKernelNodeParams* pNodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExecMemcpyNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaMemcpy3DParms* pNodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExecMemcpyNodeSetParamsToSymbol(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const void* symbol;
    const void* src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaGraphExecMemcpyNodeSetParamsFromSymbol(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    void* dst;
    const void* symbol;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
}

int handle_cudaGraphExecMemcpyNodeSetParams1D(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    void* dst;
    const void* src;
    size_t count;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_read(conn, &src, sizeof(src)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(dst)) < 0)
        return -1;
    if (rpc_write(conn, &src, sizeof(src)) < 0)
        return -1;
}

int handle_cudaGraphExecMemsetNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaMemsetParams* pNodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExecHostNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    const struct cudaHostNodeParams* pNodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(pNodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExecChildGraphNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    cudaGraph_t childGraph;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &node, sizeof(node)) < 0)
        return -1;
    if (rpc_read(conn, &childGraph, sizeof(childGraph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphExecEventRecordNodeSetEvent(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    cudaEvent_t event;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphExecEventWaitNodeSetEvent(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    cudaEvent_t event;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &event, sizeof(event)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphExecExternalSemaphoresSignalNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    const struct cudaExternalSemaphoreSignalNodeParams* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphExecExternalSemaphoresWaitNodeSetParams(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    const struct cudaExternalSemaphoreWaitNodeParams* nodeParams;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0)
        return -1;
}

int handle_cudaGraphNodeSetEnabled(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    unsigned int isEnabled;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphNodeGetEnabled(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    unsigned int* isEnabled;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hNode, sizeof(hNode)) < 0)
        return -1;
    if (rpc_read(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(isEnabled)) < 0)
        return -1;
}

int handle_cudaGraphExecUpdate(void *conn) {
    cudaGraphExec_t hGraphExec;
    cudaGraph_t hGraph;
    cudaGraphExecUpdateResultInfo* resultInfo;
    if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0)
        return -1;
    if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0)
        return -1;
    if (rpc_read(conn, &resultInfo, sizeof(resultInfo)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resultInfo, sizeof(resultInfo)) < 0)
        return -1;
}

int handle_cudaGraphUpload(void *conn) {
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    if (rpc_read(conn, &graphExec, sizeof(graphExec)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphUpload(graphExec, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphLaunch(void *conn) {
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    if (rpc_read(conn, &graphExec, sizeof(graphExec)) < 0)
        return -1;
    if (rpc_read(conn, &stream, sizeof(stream)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphLaunch(graphExec, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphExecDestroy(void *conn) {
    cudaGraphExec_t graphExec;
    if (rpc_read(conn, &graphExec, sizeof(graphExec)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphExecDestroy(graphExec);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphDestroy(void *conn) {
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphDestroy(graph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphDebugDotPrint(void *conn) {
    cudaGraph_t graph;
    const char* path;
    unsigned int flags;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &path, sizeof(path)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphDebugDotPrint(graph, path, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &path, sizeof(path)) < 0)
        return -1;
}

int handle_cudaUserObjectCreate(void *conn) {
    cudaUserObject_t* object_out;
    void* ptr;
    cudaHostFn_t destroy;
    unsigned int initialRefcount;
    unsigned int flags;
    if (rpc_read(conn, &object_out, sizeof(object_out)) < 0)
        return -1;
    if (rpc_read(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
    if (rpc_read(conn, &destroy, sizeof(destroy)) < 0)
        return -1;
    if (rpc_read(conn, &initialRefcount, sizeof(initialRefcount)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &object_out, sizeof(object_out)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(ptr)) < 0)
        return -1;
}

int handle_cudaUserObjectRetain(void *conn) {
    cudaUserObject_t object;
    unsigned int count;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaUserObjectRetain(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaUserObjectRelease(void *conn) {
    cudaUserObject_t object;
    unsigned int count;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaUserObjectRelease(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphRetainUserObject(void *conn) {
    cudaGraph_t graph;
    cudaUserObject_t object;
    unsigned int count;
    unsigned int flags;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphRetainUserObject(graph, object, count, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGraphReleaseUserObject(void *conn) {
    cudaGraph_t graph;
    cudaUserObject_t object;
    unsigned int count;
    if (rpc_read(conn, &graph, sizeof(graph)) < 0)
        return -1;
    if (rpc_read(conn, &object, sizeof(object)) < 0)
        return -1;
    if (rpc_read(conn, &count, sizeof(count)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGraphReleaseUserObject(graph, object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
}

int handle_cudaGetDriverEntryPoint(void *conn) {
    const char* symbol;
    void** funcPtr;
    unsigned long long flags;
    enum cudaDriverEntryPointQueryResult* driverStatus;
    if (rpc_read(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_read(conn, &funcPtr, sizeof(funcPtr)) < 0)
        return -1;
    if (rpc_read(conn, &flags, sizeof(flags)) < 0)
        return -1;
    if (rpc_read(conn, &driverStatus, sizeof(driverStatus)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &symbol, sizeof(symbol)) < 0)
        return -1;
    if (rpc_write(conn, &funcPtr, sizeof(funcPtr)) < 0)
        return -1;
    if (rpc_write(conn, &driverStatus, sizeof(driverStatus)) < 0)
        return -1;
}

int handle_cudaGetExportTable(void *conn) {
    const void** ppExportTable;
    const cudaUUID_t* pExportTableId;
    if (rpc_read(conn, &ppExportTable, sizeof(ppExportTable)) < 0)
        return -1;
    if (rpc_read(conn, &pExportTableId, sizeof(pExportTableId)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetExportTable(ppExportTable, pExportTableId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ppExportTable, sizeof(ppExportTable)) < 0)
        return -1;
    if (rpc_write(conn, &pExportTableId, sizeof(pExportTableId)) < 0)
        return -1;
}

int handle_cudaGetFuncBySymbol(void *conn) {
    cudaFunction_t* functionPtr;
    const void* symbolPtr;
    if (rpc_read(conn, &functionPtr, sizeof(functionPtr)) < 0)
        return -1;
    if (rpc_read(conn, &symbolPtr, sizeof(symbolPtr)) < 0)
        return -1;
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t return_value = cudaGetFuncBySymbol(functionPtr, symbolPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &functionPtr, sizeof(functionPtr)) < 0)
        return -1;
    if (rpc_write(conn, &symbolPtr, sizeof(symbolPtr)) < 0)
        return -1;
}

