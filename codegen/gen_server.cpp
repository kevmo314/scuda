#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);

int handle_nvmlInit_v2(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlInit_v2();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlInitWithFlags(void *conn)
{
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlInitWithFlags(flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlShutdown(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlShutdown();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetDriverVersion(void *conn)
{
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* version = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetDriverVersion(version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetNVMLVersion(void *conn)
{
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* version = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetNVMLVersion(version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetCudaDriverVersion(void *conn)
{
    int cudaDriverVersion;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetCudaDriverVersion(&cudaDriverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetCudaDriverVersion_v2(void *conn)
{
    int cudaDriverVersion;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetCudaDriverVersion_v2(&cudaDriverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetProcessName(void *conn)
{
    unsigned int pid;
    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* name = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetProcessName(pid, name, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, name, length * sizeof(name)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetCount(void *conn)
{
    unsigned int unitCount;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetCount(&unitCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &unitCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetHandleByIndex(void *conn)
{
    unsigned int index;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        return -1;
    nvmlUnit_t unit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetHandleByIndex(index, &unit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetUnitInfo(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlUnitInfo_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetUnitInfo(unit, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlUnitInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetLedState(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlLedState_t state;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetLedState(unit, &state);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &state, sizeof(nvmlLedState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetPsuInfo(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlPSUInfo_t psu;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetPsuInfo(unit, &psu);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &psu, sizeof(nvmlPSUInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetTemperature(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    unsigned int type;
    if (rpc_read(conn, &type, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int temp;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetTemperature(unit, type, &temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetFanSpeedInfo(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlUnitFanSpeeds_t fanSpeeds;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetDevices(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    unsigned int deviceCount;
    if (rpc_read(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlDevice_t* devices = (nvmlDevice_t*)malloc(deviceCount * sizeof(nvmlDevice_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetDevices(unit, &deviceCount, devices);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, devices, deviceCount * sizeof(devices)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetHicVersion(void *conn)
{
    unsigned int hwbcCount;
    if (rpc_read(conn, &hwbcCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlHwbcEntry_t* hwbcEntries = (nvmlHwbcEntry_t*)malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetHicVersion(&hwbcCount, hwbcEntries);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &hwbcCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, hwbcEntries, hwbcCount * sizeof(hwbcEntries)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCount_v2(void *conn)
{
    unsigned int deviceCount;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCount_v2(&deviceCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAttributes_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDeviceAttributes_t attributes;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAttributes_v2(device, &attributes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByIndex_v2(void *conn)
{
    unsigned int index;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        return -1;
    nvmlDevice_t device;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(index, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleBySerial(void *conn)
{
    std::size_t serial_len;
    if (rpc_read(conn, &serial_len, sizeof(serial_len)) < 0)
        return -1;
    char* serial = (char*)malloc(serial_len);
    if (rpc_read(conn, &serial, sizeof(char*)) < 0)
        return -1;
    nvmlDevice_t device;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleBySerial(serial, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByUUID(void *conn)
{
    std::size_t uuid_len;
    if (rpc_read(conn, &uuid_len, sizeof(uuid_len)) < 0)
        return -1;
    char* uuid = (char*)malloc(uuid_len);
    if (rpc_read(conn, &uuid, sizeof(char*)) < 0)
        return -1;
    nvmlDevice_t device;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByUUID(uuid, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(void *conn)
{
    std::size_t pciBusId_len;
    if (rpc_read(conn, &pciBusId_len, sizeof(pciBusId_len)) < 0)
        return -1;
    char* pciBusId = (char*)malloc(pciBusId_len);
    if (rpc_read(conn, &pciBusId, sizeof(char*)) < 0)
        return -1;
    nvmlDevice_t device;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetName(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* name = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetName(device, name, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, name, length * sizeof(name)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBrand(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlBrandType_t type;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBrand(device, &type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &type, sizeof(nvmlBrandType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetIndex(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int index;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetIndex(device, &index);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &index, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSerial(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    std::size_t serial_len;
    if (rpc_read(conn, &serial_len, sizeof(serial_len)) < 0)
        return -1;
    char* serial = (char*)malloc(serial_len);
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSerial(device, serial, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &serial_len, sizeof(serial_len)) < 0)
        return -1;
    if (rpc_write(conn, serial, serial_len) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryAffinity(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int nodeSetSize;
    if (rpc_read(conn, &nodeSetSize, sizeof(unsigned int)) < 0)
        return -1;
    unsigned long* nodeSet = (unsigned long*)malloc(nodeSetSize * sizeof(unsigned long));
    nvmlAffinityScope_t scope;
    if (rpc_read(conn, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, nodeSet, nodeSetSize * sizeof(nodeSet)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCpuAffinityWithinScope(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int cpuSetSize;
    if (rpc_read(conn, &cpuSetSize, sizeof(unsigned int)) < 0)
        return -1;
    unsigned long* cpuSet = (unsigned long*)malloc(cpuSetSize * sizeof(unsigned long));
    nvmlAffinityScope_t scope;
    if (rpc_read(conn, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, cpuSet, cpuSetSize * sizeof(cpuSet)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCpuAffinity(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int cpuSetSize;
    if (rpc_read(conn, &cpuSetSize, sizeof(unsigned int)) < 0)
        return -1;
    unsigned long* cpuSet = (unsigned long*)malloc(cpuSetSize * sizeof(unsigned long));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, cpuSet, cpuSetSize * sizeof(cpuSet)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetCpuAffinity(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetCpuAffinity(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceClearCpuAffinity(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearCpuAffinity(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTopologyCommonAncestor(void *conn)
{
    nvmlDevice_t device1;
    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDevice_t device2;
    if (rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuTopologyLevel_t pathInfo;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &pathInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTopologyNearestGpus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuTopologyLevel_t level;
    if (rpc_read(conn, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    nvmlDevice_t* deviceArray = (nvmlDevice_t*)malloc(count * sizeof(nvmlDevice_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, deviceArray, count * sizeof(deviceArray)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetTopologyGpuSet(void *conn)
{
    unsigned int cpuNumber;
    if (rpc_read(conn, &cpuNumber, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    nvmlDevice_t* deviceArray = (nvmlDevice_t*)malloc(count * sizeof(nvmlDevice_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetTopologyGpuSet(cpuNumber, &count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, deviceArray, count * sizeof(deviceArray)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetP2PStatus(void *conn)
{
    nvmlDevice_t device1;
    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDevice_t device2;
    if (rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuP2PCapsIndex_t p2pIndex;
    if (rpc_read(conn, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0)
        return -1;
    nvmlGpuP2PStatus_t p2pStatus;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, &p2pStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetUUID(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    std::size_t uuid_len;
    if (rpc_read(conn, &uuid_len, sizeof(uuid_len)) < 0)
        return -1;
    char* uuid = (char*)malloc(uuid_len);
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetUUID(device, uuid, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &uuid_len, sizeof(uuid_len)) < 0)
        return -1;
    if (rpc_write(conn, uuid, uuid_len) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetMdevUUID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    std::size_t mdevUuid_len;
    if (rpc_read(conn, &mdevUuid_len, sizeof(mdevUuid_len)) < 0)
        return -1;
    char* mdevUuid = (char*)malloc(mdevUuid_len);
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mdevUuid_len, sizeof(mdevUuid_len)) < 0)
        return -1;
    if (rpc_write(conn, mdevUuid, mdevUuid_len) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMinorNumber(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int minorNumber;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinorNumber(device, &minorNumber);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minorNumber, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBoardPartNumber(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* partNumber = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, partNumber, length * sizeof(partNumber)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetInforomVersion(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlInforomObject_t object;
    if (rpc_read(conn, &object, sizeof(nvmlInforomObject_t)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* version = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetInforomVersion(device, object, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetInforomImageVersion(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* version = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetInforomImageVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetInforomConfigurationChecksum(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int checksum;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetInforomConfigurationChecksum(device, &checksum);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &checksum, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceValidateInforom(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceValidateInforom(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDisplayMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t display;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDisplayMode(device, &display);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &display, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDisplayActive(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t isActive;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDisplayActive(device, &isActive);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPersistenceMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t mode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPersistenceMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPciInfo_v3(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPciInfo_t pci;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPciInfo_v3(device, &pci);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int maxLinkGen;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkGeneration(device, &maxLinkGen);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxLinkGen, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuMaxPcieLinkGeneration(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int maxLinkGenDevice;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &maxLinkGenDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxLinkGenDevice, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int maxLinkWidth;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkWidth(device, &maxLinkWidth);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int currLinkGen;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currLinkGen, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int currLinkWidth;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieThroughput(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPcieUtilCounter_t counter;
    if (rpc_read(conn, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0)
        return -1;
    unsigned int value;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieThroughput(device, counter, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieReplayCounter(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int value;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieReplayCounter(device, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetClockInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t type;
    if (rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0)
        return -1;
    unsigned int clock;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetClockInfo(device, type, &clock);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clock, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxClockInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t type;
    if (rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0)
        return -1;
    unsigned int clock;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxClockInfo(device, type, &clock);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clock, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetApplicationsClock(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t clockType;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    unsigned int clockMHz;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDefaultApplicationsClock(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t clockType;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    unsigned int clockMHz;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetApplicationsClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetApplicationsClocks(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetClock(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t clockType;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlClockId_t clockId;
    if (rpc_read(conn, &clockId, sizeof(nvmlClockId_t)) < 0)
        return -1;
    unsigned int clockMHz;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxCustomerBoostClock(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t clockType;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    unsigned int clockMHz;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedMemoryClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int* clocksMHz = (unsigned int*)malloc(count * sizeof(unsigned int));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedMemoryClocks(device, &count, clocksMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, clocksMHz, count * sizeof(clocksMHz)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int memoryClockMHz;
    if (rpc_read(conn, &memoryClockMHz, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int* clocksMHz = (unsigned int*)malloc(count * sizeof(unsigned int));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, clocksMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, clocksMHz, count * sizeof(clocksMHz)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t isEnabled;
    nvmlEnableState_t defaultIsEnabled;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    if (rpc_write(conn, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t enabled;
    if (rpc_read(conn, &enabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t enabled;
    if (rpc_read(conn, &enabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFanSpeed(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int speed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFanSpeed(device, &speed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &speed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFanSpeed_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int fan;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int speed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFanSpeed_v2(device, fan, &speed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &speed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTargetFanSpeed(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int fan;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int targetSpeed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTargetFanSpeed(device, fan, &targetSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &targetSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetDefaultFanSpeed_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int fan;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetDefaultFanSpeed_v2(device, fan);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMinMaxFanSpeed(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int minSpeed;
    unsigned int maxSpeed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minSpeed, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int fan;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        return -1;
    nvmlFanControlPolicy_t policy;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFanControlPolicy_v2(device, fan, &policy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetFanControlPolicy(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int fan;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        return -1;
    nvmlFanControlPolicy_t policy;
    if (rpc_read(conn, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetFanControlPolicy(device, fan, policy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNumFans(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int numFans;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNumFans(device, &numFans);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numFans, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTemperature(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlTemperatureSensors_t sensorType;
    if (rpc_read(conn, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0)
        return -1;
    unsigned int temp;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTemperature(device, sensorType, &temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTemperatureThreshold(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlTemperatureThresholds_t thresholdType;
    if (rpc_read(conn, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0)
        return -1;
    unsigned int temp;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, &temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetTemperatureThreshold(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlTemperatureThresholds_t thresholdType;
    if (rpc_read(conn, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0)
        return -1;
    int temp;
    if (rpc_read(conn, &temp, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetTemperatureThreshold(device, thresholdType, &temp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &temp, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetThermalSettings(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int sensorIndex;
    if (rpc_read(conn, &sensorIndex, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuThermalSettings_t pThermalSettings;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetThermalSettings(device, sensorIndex, &pThermalSettings);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPerformanceState(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPstates_t pState;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPerformanceState(device, &pState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCurrentClocksThrottleReasons(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long clocksThrottleReasons;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clocksThrottleReasons);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &clocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedClocksThrottleReasons(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long supportedClocksThrottleReasons;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedClocksThrottleReasons(device, &supportedClocksThrottleReasons);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerState(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPstates_t pState;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerState(device, &pState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t mode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementLimit(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int limit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimit(device, &limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &limit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementLimitConstraints(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int minLimit;
    unsigned int maxLimit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minLimit, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &maxLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int defaultLimit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &defaultLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerUsage(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int power;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &power, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTotalEnergyConsumption(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long energy;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &energy, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEnforcedPowerLimit(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int limit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEnforcedPowerLimit(device, &limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &limit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuOperationMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuOperationMode_t current;
    nvmlGpuOperationMode_t pending;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuOperationMode(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;
    if (rpc_write(conn, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlMemory_t memory;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memory, sizeof(nvmlMemory_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryInfo_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlMemory_v2_t memory;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryInfo_v2(device, &memory);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memory, sizeof(nvmlMemory_v2_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetComputeMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlComputeMode_t mode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetComputeMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCudaComputeCapability(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int major;
    int minor;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &major, sizeof(int)) < 0)
        return -1;
    if (rpc_write(conn, &minor, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEccMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t current;
    nvmlEnableState_t pending;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEccMode(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    if (rpc_write(conn, &pending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDefaultEccMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t defaultMode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDefaultEccMode(device, &defaultMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &defaultMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBoardId(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int boardId;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBoardId(device, &boardId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &boardId, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMultiGpuBoard(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int multiGpuBool;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMultiGpuBoard(device, &multiGpuBool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &multiGpuBool, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTotalEccErrors(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlMemoryErrorType_t errorType;
    if (rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0)
        return -1;
    nvmlEccCounterType_t counterType;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;
    unsigned long long eccCounts;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, &eccCounts);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eccCounts, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDetailedEccErrors(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlMemoryErrorType_t errorType;
    if (rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0)
        return -1;
    nvmlEccCounterType_t counterType;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;
    nvmlEccErrorCounts_t eccCounts;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, &eccCounts);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryErrorCounter(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlMemoryErrorType_t errorType;
    if (rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0)
        return -1;
    nvmlEccCounterType_t counterType;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;
    nvmlMemoryLocation_t locationType;
    if (rpc_read(conn, &locationType, sizeof(nvmlMemoryLocation_t)) < 0)
        return -1;
    unsigned long long count;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetUtilizationRates(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlUtilization_t utilization;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &utilization, sizeof(nvmlUtilization_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderUtilization(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int utilization;
    unsigned int samplingPeriodUs;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderUtilization(device, &utilization, &samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &utilization, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderCapacity(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEncoderType_t encoderQueryType;
    if (rpc_read(conn, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0)
        return -1;
    unsigned int encoderCapacity;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, &encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderStats(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &averageFps, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderSessions(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int sessionCount;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlEncoderSessionInfo_t* sessionInfos = (nvmlEncoderSessionInfo_t*)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfos, sessionCount * sizeof(sessionInfos)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDecoderUtilization(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int utilization;
    unsigned int samplingPeriodUs;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &utilization, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFBCStats(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlFBCStats_t fbcStats;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFBCStats(device, &fbcStats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFBCSessions(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int sessionCount;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlFBCSessionInfo_t* sessionInfo = (nvmlFBCSessionInfo_t*)malloc(sessionCount * sizeof(nvmlFBCSessionInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFBCSessions(device, &sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfo, sessionCount * sizeof(sessionInfo)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDriverModel(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDriverModel_t current;
    nvmlDriverModel_t pending;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDriverModel(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(nvmlDriverModel_t)) < 0)
        return -1;
    if (rpc_write(conn, &pending, sizeof(nvmlDriverModel_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVbiosVersion(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    std::size_t version_len;
    if (rpc_read(conn, &version_len, sizeof(version_len)) < 0)
        return -1;
    char* version = (char*)malloc(version_len);
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVbiosVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version_len, sizeof(version_len)) < 0)
        return -1;
    if (rpc_write(conn, version, version_len) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBridgeChipInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlBridgeChipHierarchy_t bridgeHierarchy;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int infoCount;
    if (rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessInfo_t* infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, infos, infoCount * sizeof(infos)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int infoCount;
    if (rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessInfo_t* infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, infos, infoCount * sizeof(infos)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int infoCount;
    if (rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessInfo_t* infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, infos, infoCount * sizeof(infos)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceOnSameBoard(void *conn)
{
    nvmlDevice_t device1;
    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDevice_t device2;
    if (rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int onSameBoard;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceOnSameBoard(device1, device2, &onSameBoard);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &onSameBoard, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAPIRestriction(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlRestrictedAPI_t apiType;
    if (rpc_read(conn, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0)
        return -1;
    nvmlEnableState_t isRestricted;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSamples(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlSamplingType_t type;
    if (rpc_read(conn, &type, sizeof(nvmlSamplingType_t)) < 0)
        return -1;
    unsigned long long lastSeenTimeStamp;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;
    nvmlValueType_t sampleValType;
    unsigned int sampleCount;
    if (rpc_read(conn, &sampleCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlSample_t* samples = (nvmlSample_t*)malloc(sampleCount * sizeof(nvmlSample_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0)
        return -1;
    if (rpc_write(conn, &sampleCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, samples, sampleCount * sizeof(samples)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlBAR1Memory_t bar1Memory;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetViolationStatus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPerfPolicyType_t perfPolicyType;
    if (rpc_read(conn, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0)
        return -1;
    nvmlViolationTime_t violTime;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetViolationStatus(device, perfPolicyType, &violTime);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &violTime, sizeof(nvmlViolationTime_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetIrqNum(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int irqNum;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetIrqNum(device, &irqNum);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &irqNum, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNumGpuCores(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int numCores;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNumGpuCores(device, &numCores);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numCores, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerSource(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPowerSource_t powerSource;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerSource(device, &powerSource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &powerSource, sizeof(nvmlPowerSource_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryBusWidth(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int busWidth;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryBusWidth(device, &busWidth);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &busWidth, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieLinkMaxSpeed(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int maxSpeed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieLinkMaxSpeed(device, &maxSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieSpeed(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int pcieSpeed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pcieSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAdaptiveClockInfoStatus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int adaptiveClockStatus;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &adaptiveClockStatus, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t mode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingStats(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int pid;
    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0)
        return -1;
    nvmlAccountingStats_t stats;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingStats(device, pid, &stats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingPids(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int* pids = (unsigned int*)malloc(count * sizeof(unsigned int));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingPids(device, &count, pids);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, pids, count * sizeof(pids)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingBufferSize(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int bufferSize;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingBufferSize(device, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPages(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPageRetirementCause_t cause;
    if (rpc_read(conn, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
        return -1;
    unsigned int pageCount;
    if (rpc_read(conn, &pageCount, sizeof(unsigned int)) < 0)
        return -1;
    unsigned long long* addresses = (unsigned long long*)malloc(pageCount * sizeof(unsigned long long));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, addresses);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pageCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, addresses, pageCount * sizeof(addresses)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPages_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlPageRetirementCause_t cause;
    if (rpc_read(conn, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
        return -1;
    unsigned int pageCount;
    if (rpc_read(conn, &pageCount, sizeof(unsigned int)) < 0)
        return -1;
    unsigned long long* addresses = (unsigned long long*)malloc(pageCount * sizeof(unsigned long long));
    unsigned long long* timestamps = (unsigned long long*)malloc(pageCount * sizeof(unsigned long long));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, addresses, timestamps);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pageCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, addresses, pageCount * sizeof(addresses)) < 0)
        return -1;
    if (rpc_write(conn, timestamps, pageCount * sizeof(timestamps)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t isPending;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPagesPendingStatus(device, &isPending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isPending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRemappedRows(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int corrRows;
    unsigned int uncRows;
    unsigned int isPending;
    unsigned int failureOccurred;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &corrRows, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &uncRows, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &isPending, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &failureOccurred, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRowRemapperHistogram(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlRowRemapperHistogramValues_t values;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRowRemapperHistogram(device, &values);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetArchitecture(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDeviceArchitecture_t arch;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetArchitecture(device, &arch);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitSetLedState(void *conn)
{
    nvmlUnit_t unit;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlLedColor_t color;
    if (rpc_read(conn, &color, sizeof(nvmlLedColor_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitSetLedState(unit, color);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetPersistenceMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t mode;
    if (rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetPersistenceMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetComputeMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlComputeMode_t mode;
    if (rpc_read(conn, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetComputeMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetEccMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t ecc;
    if (rpc_read(conn, &ecc, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetEccMode(device, ecc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceClearEccErrorCounts(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEccCounterType_t counterType;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearEccErrorCounts(device, counterType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetDriverModel(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDriverModel_t driverModel;
    if (rpc_read(conn, &driverModel, sizeof(nvmlDriverModel_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetDriverModel(device, driverModel, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetGpuLockedClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int minGpuClockMHz;
    if (rpc_read(conn, &minGpuClockMHz, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int maxGpuClockMHz;
    if (rpc_read(conn, &maxGpuClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetGpuLockedClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetGpuLockedClocks(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetMemoryLockedClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int minMemClockMHz;
    if (rpc_read(conn, &minMemClockMHz, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int maxMemClockMHz;
    if (rpc_read(conn, &maxMemClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetMemoryLockedClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetMemoryLockedClocks(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetApplicationsClocks(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int memClockMHz;
    if (rpc_read(conn, &memClockMHz, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int graphicsClockMHz;
    if (rpc_read(conn, &graphicsClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetClkMonStatus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClkMonStatus_t status;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetClkMonStatus(device, &status);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &status, sizeof(nvmlClkMonStatus_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetPowerManagementLimit(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int limit;
    if (rpc_read(conn, &limit, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetPowerManagementLimit(device, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetGpuOperationMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuOperationMode_t mode;
    if (rpc_read(conn, &mode, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetGpuOperationMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetAPIRestriction(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlRestrictedAPI_t apiType;
    if (rpc_read(conn, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0)
        return -1;
    nvmlEnableState_t isRestricted;
    if (rpc_read(conn, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetAPIRestriction(device, apiType, isRestricted);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetAccountingMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlEnableState_t mode;
    if (rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetAccountingMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceClearAccountingPids(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearAccountingPids(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkState(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    nvmlEnableState_t isActive;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkState(device, link, &isActive);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkVersion(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int version;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkVersion(device, link, &version);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkCapability(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    nvmlNvLinkCapability_t capability;
    if (rpc_read(conn, &capability, sizeof(nvmlNvLinkCapability_t)) < 0)
        return -1;
    unsigned int capResult;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkCapability(device, link, capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkRemotePciInfo_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    nvmlPciInfo_t pci;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, &pci);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkErrorCounter(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    nvmlNvLinkErrorCounter_t counter;
    if (rpc_read(conn, &counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0)
        return -1;
    unsigned long long counterValue;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, &counterValue);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &counterValue, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetNvLinkErrorCounters(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetNvLinkErrorCounters(device, link);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetNvLinkUtilizationControl(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int counter;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        return -1;
    nvmlNvLinkUtilizationControl_t control;
    if (rpc_read(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;
    unsigned int reset;
    if (rpc_read(conn, &reset, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, &control, reset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkUtilizationControl(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int counter;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        return -1;
    nvmlNvLinkUtilizationControl_t control;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, &control);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int counter;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        return -1;
    unsigned long long rxcounter;
    unsigned long long txcounter;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, &rxcounter, &txcounter);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &rxcounter, sizeof(unsigned long long)) < 0)
        return -1;
    if (rpc_write(conn, &txcounter, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceFreezeNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int counter;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        return -1;
    nvmlEnableState_t freeze;
    if (rpc_read(conn, &freeze, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int counter;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkRemoteDeviceType(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int link;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        return -1;
    nvmlIntNvLinkDeviceType_t pNvLinkDeviceType;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, &pNvLinkDeviceType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlEventSetCreate(void *conn)
{
    nvmlEventSet_t set;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetCreate(&set);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceRegisterEvents(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long eventTypes;
    if (rpc_read(conn, &eventTypes, sizeof(unsigned long long)) < 0)
        return -1;
    nvmlEventSet_t set;
    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceRegisterEvents(device, eventTypes, set);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedEventTypes(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long eventTypes;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedEventTypes(device, &eventTypes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eventTypes, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlEventSetWait_v2(void *conn)
{
    nvmlEventSet_t set;
    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;
    nvmlEventData_t data;
    unsigned int timeoutms;
    if (rpc_read(conn, &timeoutms, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetWait_v2(set, &data, timeoutms);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(nvmlEventData_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlEventSetFree(void *conn)
{
    nvmlEventSet_t set;
    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetFree(set);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceModifyDrainState(void *conn)
{
    nvmlPciInfo_t pciInfo;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;
    nvmlEnableState_t newState;
    if (rpc_read(conn, &newState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceModifyDrainState(&pciInfo, newState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceQueryDrainState(void *conn)
{
    nvmlPciInfo_t pciInfo;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;
    nvmlEnableState_t currentState;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceQueryDrainState(&pciInfo, &currentState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currentState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceRemoveGpu_v2(void *conn)
{
    nvmlPciInfo_t pciInfo;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;
    nvmlDetachGpuState_t gpuState;
    if (rpc_read(conn, &gpuState, sizeof(nvmlDetachGpuState_t)) < 0)
        return -1;
    nvmlPcieLinkState_t linkState;
    if (rpc_read(conn, &linkState, sizeof(nvmlPcieLinkState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceRemoveGpu_v2(&pciInfo, gpuState, linkState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceDiscoverGpus(void *conn)
{
    nvmlPciInfo_t pciInfo;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceDiscoverGpus(&pciInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFieldValues(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int valuesCount;
    if (rpc_read(conn, &valuesCount, sizeof(int)) < 0)
        return -1;
    nvmlFieldValue_t* values = (nvmlFieldValue_t*)malloc(valuesCount * sizeof(nvmlFieldValue_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFieldValues(device, valuesCount, values);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, values, valuesCount * sizeof(values)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceClearFieldValues(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int valuesCount;
    if (rpc_read(conn, &valuesCount, sizeof(int)) < 0)
        return -1;
    nvmlFieldValue_t* values = (nvmlFieldValue_t*)malloc(valuesCount * sizeof(nvmlFieldValue_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearFieldValues(device, valuesCount, values);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, values, valuesCount * sizeof(values)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVirtualizationMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuVirtualizationMode_t pVirtualMode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVirtualizationMode(device, &pVirtualMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHostVgpuMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlHostVgpuMode_t pHostVgpuMode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHostVgpuMode(device, &pHostVgpuMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetVirtualizationMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuVirtualizationMode_t virtualMode;
    if (rpc_read(conn, &virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetVirtualizationMode(device, virtualMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGridLicensableFeatures_v4(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGridLicensableFeatures_t pGridLicensableFeatures;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGridLicensableFeatures_v4(device, &pGridLicensableFeatures);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetProcessUtilization(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int processSamplesCount;
    if (rpc_read(conn, &processSamplesCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessUtilizationSample_t* utilization = (nvmlProcessUtilizationSample_t*)malloc(processSamplesCount * sizeof(nvmlProcessUtilizationSample_t));
    unsigned long long lastSeenTimeStamp;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetProcessUtilization(device, utilization, &processSamplesCount, lastSeenTimeStamp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &processSamplesCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, utilization, processSamplesCount * sizeof(utilization)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGspFirmwareVersion(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    char version;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGspFirmwareVersion(device, &version);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version, sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGspFirmwareMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int isEnabled;
    unsigned int defaultMode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &defaultMode, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuDriverCapabilities(void *conn)
{
    nvmlVgpuDriverCapability_t capability;
    if (rpc_read(conn, &capability, sizeof(nvmlVgpuDriverCapability_t)) < 0)
        return -1;
    unsigned int capResult;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuDriverCapabilities(capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuCapabilities(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDeviceVgpuCapability_t capability;
    if (rpc_read(conn, &capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0)
        return -1;
    unsigned int capResult;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuCapabilities(device, capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedVgpus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int vgpuCount;
    if (rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuTypeId_t* vgpuTypeIds = (nvmlVgpuTypeId_t*)malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(vgpuTypeIds)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCreatableVgpus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int vgpuCount;
    if (rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuTypeId_t* vgpuTypeIds = (nvmlVgpuTypeId_t*)malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCreatableVgpus(device, &vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(vgpuTypeIds)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetClass(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int size;
    char* vgpuTypeClass = (char*)malloc(size * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, &size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeClass, size * sizeof(vgpuTypeClass)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetName(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    char* vgpuTypeName = (char*)malloc(size * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, &size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeName, size * sizeof(vgpuTypeName)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int gpuInstanceProfileId;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, &gpuInstanceProfileId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstanceProfileId, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetDeviceID(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned long long deviceID;
    unsigned long long subsystemID;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, &deviceID, &subsystemID);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceID, sizeof(unsigned long long)) < 0)
        return -1;
    if (rpc_write(conn, &subsystemID, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetFramebufferSize(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned long long fbSize;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, &fbSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbSize, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetNumDisplayHeads(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int numDisplayHeads;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, &numDisplayHeads);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numDisplayHeads, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetResolution(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int displayIndex;
    if (rpc_read(conn, &displayIndex, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int xdim;
    unsigned int ydim;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, &xdim, &ydim);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &xdim, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &ydim, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetLicense(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    char* vgpuTypeLicenseString = (char*)malloc(size * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, vgpuTypeLicenseString, size * sizeof(vgpuTypeLicenseString)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetFrameRateLimit(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int frameRateLimit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, &frameRateLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetMaxInstances(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int vgpuInstanceCount;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, &vgpuInstanceCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuInstanceCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetMaxInstancesPerVm(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    unsigned int vgpuInstanceCountPerVm;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, &vgpuInstanceCountPerVm);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetActiveVgpus(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int vgpuCount;
    if (rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuInstance_t* vgpuInstances = (nvmlVgpuInstance_t*)malloc(vgpuCount * sizeof(nvmlVgpuInstance_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetActiveVgpus(device, &vgpuCount, vgpuInstances);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuInstances, vgpuCount * sizeof(vgpuInstances)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetVmID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    char* vmId = (char*)malloc(size * sizeof(char));
    nvmlVgpuVmIdType_t vmIdType;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, &vmIdType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, vmId, size * sizeof(vmId)) < 0)
        return -1;
    if (rpc_write(conn, &vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetUUID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    char* uuid = (char*)malloc(size * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, size * sizeof(uuid)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetVmDriverVersion(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* version = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(version)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFbUsage(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned long long fbUsage;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFbUsage(vgpuInstance, &fbUsage);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbUsage, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetLicenseStatus(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int licensed;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, &licensed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &licensed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetType(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    nvmlVgpuTypeId_t vgpuTypeId;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetType(vgpuInstance, &vgpuTypeId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFrameRateLimit(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int frameRateLimit;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, &frameRateLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEccMode(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    nvmlEnableState_t eccMode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEccMode(vgpuInstance, &eccMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &eccMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderCapacity(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int encoderCapacity;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, &encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceSetEncoderCapacity(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int encoderCapacity;
    if (rpc_read(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, &sessionCount, &averageFps, &averageLatency);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &averageFps, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderSessions(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int sessionCount;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlEncoderSessionInfo_t* sessionInfo = (nvmlEncoderSessionInfo_t*)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfo, sessionCount * sizeof(sessionInfo)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFBCStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    nvmlFBCStats_t fbcStats;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFBCStats(vgpuInstance, &fbcStats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFBCSessions(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int sessionCount;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlFBCSessionInfo_t* sessionInfo = (nvmlFBCSessionInfo_t*)malloc(sessionCount * sizeof(nvmlFBCSessionInfo_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, &sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, sessionInfo, sessionCount * sizeof(sessionInfo)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetGpuInstanceId(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int gpuInstanceId;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, &gpuInstanceId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstanceId, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetGpuPciId(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* vgpuPciId = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, &length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuPciId, length * sizeof(vgpuPciId)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetCapabilities(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;
    nvmlVgpuCapability_t capability;
    if (rpc_read(conn, &capability, sizeof(nvmlVgpuCapability_t)) < 0)
        return -1;
    unsigned int capResult;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetMetadata(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int bufferSize;
    if (rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuMetadata_t* vgpuMetadata = (nvmlVgpuMetadata_t*)malloc(bufferSize * sizeof(nvmlVgpuMetadata_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, vgpuMetadata, bufferSize * sizeof(vgpuMetadata)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuMetadata(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int bufferSize;
    if (rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuPgpuMetadata_t* pgpuMetadata = (nvmlVgpuPgpuMetadata_t*)malloc(bufferSize * sizeof(nvmlVgpuPgpuMetadata_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, pgpuMetadata, bufferSize * sizeof(pgpuMetadata)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuCompatibility(void *conn)
{
    nvmlVgpuMetadata_t vgpuMetadata;
    if (rpc_read(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0)
        return -1;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    nvmlVgpuPgpuCompatibility_t compatibilityInfo;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &compatibilityInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0)
        return -1;
    if (rpc_write(conn, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0)
        return -1;
    if (rpc_write(conn, &compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPgpuMetadataString(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int bufferSize;
    if (rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;
    char* pgpuMetadata = (char*)malloc(bufferSize * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, pgpuMetadata, bufferSize * sizeof(pgpuMetadata)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerLog(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlVgpuSchedulerLog_t pSchedulerLog;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuSchedulerLog(device, &pSchedulerLog);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerState(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlVgpuSchedulerGetState_t pSchedulerState;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuSchedulerState(device, &pSchedulerState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerCapabilities(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlVgpuSchedulerCapabilities_t pCapabilities;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuSchedulerCapabilities(device, &pCapabilities);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuVersion(void *conn)
{
    nvmlVgpuVersion_t supported;
    nvmlVgpuVersion_t current;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuVersion(&supported, &current);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &supported, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;
    if (rpc_write(conn, &current, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSetVgpuVersion(void *conn)
{
    nvmlVgpuVersion_t vgpuVersion;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSetVgpuVersion(&vgpuVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuUtilization(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long lastSeenTimeStamp;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;
    nvmlValueType_t sampleValType;
    if (rpc_read(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0)
        return -1;
    unsigned int vgpuInstanceSamplesCount;
    if (rpc_read(conn, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuInstanceUtilizationSample_t* utilizationSamples = (nvmlVgpuInstanceUtilizationSample_t*)malloc(vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, &sampleValType, &vgpuInstanceSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0)
        return -1;
    if (rpc_write(conn, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, utilizationSamples, vgpuInstanceSamplesCount * sizeof(utilizationSamples)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuProcessUtilization(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned long long lastSeenTimeStamp;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;
    unsigned int vgpuProcessSamplesCount;
    if (rpc_read(conn, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlVgpuProcessUtilizationSample_t* utilizationSamples = (nvmlVgpuProcessUtilizationSample_t*)malloc(vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, &vgpuProcessSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, utilizationSamples, vgpuProcessSamplesCount * sizeof(utilizationSamples)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingMode(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    nvmlEnableState_t mode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, &mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingPids(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int* pids = (unsigned int*)malloc(count * sizeof(unsigned int));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, &count, pids);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, pids, count * sizeof(pids)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int pid;
    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0)
        return -1;
    nvmlAccountingStats_t stats;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, &stats);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceClearAccountingPids(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceClearAccountingPids(vgpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetLicenseInfo_v2(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    nvmlVgpuLicenseInfo_t licenseInfo;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, &licenseInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetExcludedDeviceCount(void *conn)
{
    unsigned int deviceCount;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetExcludedDeviceCount(&deviceCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetExcludedDeviceInfoByIndex(void *conn)
{
    unsigned int index;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        return -1;
    nvmlExcludedDeviceInfo_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetExcludedDeviceInfoByIndex(index, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlExcludedDeviceInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetMigMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int mode;
    if (rpc_read(conn, &mode, sizeof(unsigned int)) < 0)
        return -1;
    nvmlReturn_t activationStatus;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetMigMode(device, mode, &activationStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &activationStatus, sizeof(nvmlReturn_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMigMode(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int currentMode;
    unsigned int pendingMode;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &currentMode, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &pendingMode, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profile;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuInstanceProfileInfo_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfoV(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profile;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuInstanceProfileInfo_v2_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstancePossiblePlacements_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuInstancePlacement_t* placements = (nvmlGpuInstancePlacement_t*)malloc(count * sizeof(nvmlGpuInstancePlacement_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, placements, count * sizeof(placements)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceRemainingCapacity(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceCreateGpuInstance(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuInstance_t gpuInstance;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCreateGpuInstance(device, profileId, &gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceCreateGpuInstanceWithPlacement(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    bool placement_null_check;
    if (rpc_read(conn, &placement_null_check, 1) < 0)
        return -1;
    nvmlGpuInstancePlacement_t* placement = nullptr;
    if (placement_null_check) {
        placement = (nvmlGpuInstancePlacement_t*)malloc(sizeof(nvmlGpuInstancePlacement_t));
        if (rpc_read(conn, placement, sizeof(nvmlGpuInstancePlacement_t)) < 0)
            return -1;
    }
    if (rpc_read(conn, &placement, sizeof(nvmlGpuInstancePlacement_t*)) < 0)
        return -1;
    nvmlGpuInstance_t gpuInstance;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement, &gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceDestroy(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceDestroy(gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstances(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuInstance_t* gpuInstances = (nvmlGpuInstance_t*)malloc(count * sizeof(nvmlGpuInstance_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, gpuInstances, count * sizeof(gpuInstances)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceById(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int id;
    if (rpc_read(conn, &id, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpuInstance_t gpuInstance;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceById(device, id, &gpuInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetInfo(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    nvmlGpuInstanceInfo_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetInfo(gpuInstance, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlGpuInstanceInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfo(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profile;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int engProfile;
    if (rpc_read(conn, &engProfile, sizeof(unsigned int)) < 0)
        return -1;
    nvmlComputeInstanceProfileInfo_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profile;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int engProfile;
    if (rpc_read(conn, &engProfile, sizeof(unsigned int)) < 0)
        return -1;
    nvmlComputeInstanceProfileInfo_v2_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile, engProfile, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    nvmlComputeInstancePlacement_t* placements = (nvmlComputeInstancePlacement_t*)malloc(count * sizeof(nvmlComputeInstancePlacement_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placements, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, placements, count * sizeof(placements)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceCreateComputeInstance(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    nvmlComputeInstance_t computeInstance;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, &computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceCreateComputeInstanceWithPlacement(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    bool placement_null_check;
    if (rpc_read(conn, &placement_null_check, 1) < 0)
        return -1;
    nvmlComputeInstancePlacement_t* placement = nullptr;
    if (placement_null_check) {
        placement = (nvmlComputeInstancePlacement_t*)malloc(sizeof(nvmlComputeInstancePlacement_t));
        if (rpc_read(conn, placement, sizeof(nvmlComputeInstancePlacement_t)) < 0)
            return -1;
    }
    if (rpc_read(conn, &placement, sizeof(nvmlComputeInstancePlacement_t*)) < 0)
        return -1;
    nvmlComputeInstance_t computeInstance;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, placement, &computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlComputeInstanceDestroy(void *conn)
{
    nvmlComputeInstance_t computeInstance;
    if (rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlComputeInstanceDestroy(computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstances(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int profileId;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    nvmlComputeInstance_t* computeInstances = (nvmlComputeInstance_t*)malloc(count * sizeof(nvmlComputeInstance_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, computeInstances, count * sizeof(computeInstances)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceById(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;
    unsigned int id;
    if (rpc_read(conn, &id, sizeof(unsigned int)) < 0)
        return -1;
    nvmlComputeInstance_t computeInstance;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, &computeInstance);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlComputeInstanceGetInfo_v2(void *conn)
{
    nvmlComputeInstance_t computeInstance;
    if (rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;
    nvmlComputeInstanceInfo_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlComputeInstanceGetInfo_v2(computeInstance, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlComputeInstanceInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceIsMigDeviceHandle(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int isMigDevice;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceIsMigDeviceHandle(device, &isMigDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isMigDevice, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceId(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int id;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceId(device, &id);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &id, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetComputeInstanceId(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int id;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetComputeInstanceId(device, &id);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &id, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxMigDeviceCount(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int count;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxMigDeviceCount(device, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMigDeviceHandleByIndex(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int index;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        return -1;
    nvmlDevice_t migDevice;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMigDeviceHandleByIndex(device, index, &migDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &migDevice, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDeviceHandleFromMigDeviceHandle(void *conn)
{
    nvmlDevice_t migDevice;
    if (rpc_read(conn, &migDevice, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlDevice_t device;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBusType(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlBusType_t type;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBusType(device, &type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &type, sizeof(nvmlBusType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDynamicPstatesInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetFanSpeed_v2(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int fan;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int speed;
    if (rpc_read(conn, &speed, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetFanSpeed_v2(device, fan, speed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpcClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int offset;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpcClkVfOffset(device, &offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &offset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetGpcClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int offset;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetGpcClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int offset;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemClkVfOffset(device, &offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &offset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetMemClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int offset;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetMemClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMinMaxClockOfPState(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlClockType_t type;
    if (rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlPstates_t pstate;
    if (rpc_read(conn, &pstate, sizeof(nvmlPstates_t)) < 0)
        return -1;
    unsigned int minClockMHz;
    unsigned int maxClockMHz;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minClockMHz, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &maxClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    nvmlPstates_t* pstates = (nvmlPstates_t*)malloc(size * sizeof(nvmlPstates_t));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedPerformanceStates(device, pstates, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, pstates, size * sizeof(pstates)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int minOffset;
    int maxOffset;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minOffset, sizeof(int)) < 0)
        return -1;
    if (rpc_write(conn, &maxOffset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemClkMinMaxVfOffset(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    int minOffset;
    int maxOffset;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &minOffset, sizeof(int)) < 0)
        return -1;
    if (rpc_write(conn, &maxOffset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuFabricInfo(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpuFabricInfo_t gpuFabricInfo;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmMetricsGet(void *conn)
{
    nvmlGpmMetricsGet_t metricsGet;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmMetricsGet(&metricsGet);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmSampleFree(void *conn)
{
    nvmlGpmSample_t gpmSample;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleFree(gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmSampleAlloc(void *conn)
{
    nvmlGpmSample_t gpmSample;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleAlloc(&gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmSampleGet(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpmSample_t gpmSample;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleGet(device, gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmMigSampleGet(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    unsigned int gpuInstanceId;
    if (rpc_read(conn, &gpuInstanceId, sizeof(unsigned int)) < 0)
        return -1;
    nvmlGpmSample_t gpmSample;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmQueryDeviceSupport(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlGpmSupport_t gpmSupport;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmQueryDeviceSupport(device, &gpmSupport);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &gpmSupport, sizeof(nvmlGpmSupport_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold(void *conn)
{
    nvmlDevice_t device;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlNvLinkPowerThres_t info;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, &info);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &info, sizeof(nvmlNvLinkPowerThres_t)) < 0)
        return -1;

    return result;
}

static RequestHandler opHandlers[] = {
    handle_nvmlInit_v2,
    handle_nvmlInitWithFlags,
    handle_nvmlShutdown,
    handle_nvmlSystemGetDriverVersion,
    handle_nvmlSystemGetNVMLVersion,
    handle_nvmlSystemGetCudaDriverVersion,
    handle_nvmlSystemGetCudaDriverVersion_v2,
    handle_nvmlSystemGetProcessName,
    handle_nvmlUnitGetCount,
    handle_nvmlUnitGetHandleByIndex,
    handle_nvmlUnitGetUnitInfo,
    handle_nvmlUnitGetLedState,
    handle_nvmlUnitGetPsuInfo,
    handle_nvmlUnitGetTemperature,
    handle_nvmlUnitGetFanSpeedInfo,
    handle_nvmlUnitGetDevices,
    handle_nvmlSystemGetHicVersion,
    handle_nvmlDeviceGetCount_v2,
    handle_nvmlDeviceGetAttributes_v2,
    handle_nvmlDeviceGetHandleByIndex_v2,
    handle_nvmlDeviceGetHandleBySerial,
    handle_nvmlDeviceGetHandleByUUID,
    handle_nvmlDeviceGetHandleByPciBusId_v2,
    handle_nvmlDeviceGetName,
    handle_nvmlDeviceGetBrand,
    handle_nvmlDeviceGetIndex,
    handle_nvmlDeviceGetSerial,
    handle_nvmlDeviceGetMemoryAffinity,
    handle_nvmlDeviceGetCpuAffinityWithinScope,
    handle_nvmlDeviceGetCpuAffinity,
    handle_nvmlDeviceSetCpuAffinity,
    handle_nvmlDeviceClearCpuAffinity,
    handle_nvmlDeviceGetTopologyCommonAncestor,
    handle_nvmlDeviceGetTopologyNearestGpus,
    handle_nvmlSystemGetTopologyGpuSet,
    handle_nvmlDeviceGetP2PStatus,
    handle_nvmlDeviceGetUUID,
    handle_nvmlVgpuInstanceGetMdevUUID,
    handle_nvmlDeviceGetMinorNumber,
    handle_nvmlDeviceGetBoardPartNumber,
    handle_nvmlDeviceGetInforomVersion,
    handle_nvmlDeviceGetInforomImageVersion,
    handle_nvmlDeviceGetInforomConfigurationChecksum,
    handle_nvmlDeviceValidateInforom,
    handle_nvmlDeviceGetDisplayMode,
    handle_nvmlDeviceGetDisplayActive,
    handle_nvmlDeviceGetPersistenceMode,
    handle_nvmlDeviceGetPciInfo_v3,
    handle_nvmlDeviceGetMaxPcieLinkGeneration,
    handle_nvmlDeviceGetGpuMaxPcieLinkGeneration,
    handle_nvmlDeviceGetMaxPcieLinkWidth,
    handle_nvmlDeviceGetCurrPcieLinkGeneration,
    handle_nvmlDeviceGetCurrPcieLinkWidth,
    handle_nvmlDeviceGetPcieThroughput,
    handle_nvmlDeviceGetPcieReplayCounter,
    handle_nvmlDeviceGetClockInfo,
    handle_nvmlDeviceGetMaxClockInfo,
    handle_nvmlDeviceGetApplicationsClock,
    handle_nvmlDeviceGetDefaultApplicationsClock,
    handle_nvmlDeviceResetApplicationsClocks,
    handle_nvmlDeviceGetClock,
    handle_nvmlDeviceGetMaxCustomerBoostClock,
    handle_nvmlDeviceGetSupportedMemoryClocks,
    handle_nvmlDeviceGetSupportedGraphicsClocks,
    handle_nvmlDeviceGetAutoBoostedClocksEnabled,
    handle_nvmlDeviceSetAutoBoostedClocksEnabled,
    handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled,
    handle_nvmlDeviceGetFanSpeed,
    handle_nvmlDeviceGetFanSpeed_v2,
    handle_nvmlDeviceGetTargetFanSpeed,
    handle_nvmlDeviceSetDefaultFanSpeed_v2,
    handle_nvmlDeviceGetMinMaxFanSpeed,
    handle_nvmlDeviceGetFanControlPolicy_v2,
    handle_nvmlDeviceSetFanControlPolicy,
    handle_nvmlDeviceGetNumFans,
    handle_nvmlDeviceGetTemperature,
    handle_nvmlDeviceGetTemperatureThreshold,
    handle_nvmlDeviceSetTemperatureThreshold,
    handle_nvmlDeviceGetThermalSettings,
    handle_nvmlDeviceGetPerformanceState,
    handle_nvmlDeviceGetCurrentClocksThrottleReasons,
    handle_nvmlDeviceGetSupportedClocksThrottleReasons,
    handle_nvmlDeviceGetPowerState,
    handle_nvmlDeviceGetPowerManagementMode,
    handle_nvmlDeviceGetPowerManagementLimit,
    handle_nvmlDeviceGetPowerManagementLimitConstraints,
    handle_nvmlDeviceGetPowerManagementDefaultLimit,
    handle_nvmlDeviceGetPowerUsage,
    handle_nvmlDeviceGetTotalEnergyConsumption,
    handle_nvmlDeviceGetEnforcedPowerLimit,
    handle_nvmlDeviceGetGpuOperationMode,
    handle_nvmlDeviceGetMemoryInfo,
    handle_nvmlDeviceGetMemoryInfo_v2,
    handle_nvmlDeviceGetComputeMode,
    handle_nvmlDeviceGetCudaComputeCapability,
    handle_nvmlDeviceGetEccMode,
    handle_nvmlDeviceGetDefaultEccMode,
    handle_nvmlDeviceGetBoardId,
    handle_nvmlDeviceGetMultiGpuBoard,
    handle_nvmlDeviceGetTotalEccErrors,
    handle_nvmlDeviceGetDetailedEccErrors,
    handle_nvmlDeviceGetMemoryErrorCounter,
    handle_nvmlDeviceGetUtilizationRates,
    handle_nvmlDeviceGetEncoderUtilization,
    handle_nvmlDeviceGetEncoderCapacity,
    handle_nvmlDeviceGetEncoderStats,
    handle_nvmlDeviceGetEncoderSessions,
    handle_nvmlDeviceGetDecoderUtilization,
    handle_nvmlDeviceGetFBCStats,
    handle_nvmlDeviceGetFBCSessions,
    handle_nvmlDeviceGetDriverModel,
    handle_nvmlDeviceGetVbiosVersion,
    handle_nvmlDeviceGetBridgeChipInfo,
    handle_nvmlDeviceGetComputeRunningProcesses_v3,
    handle_nvmlDeviceGetGraphicsRunningProcesses_v3,
    handle_nvmlDeviceGetMPSComputeRunningProcesses_v3,
    handle_nvmlDeviceOnSameBoard,
    handle_nvmlDeviceGetAPIRestriction,
    handle_nvmlDeviceGetSamples,
    handle_nvmlDeviceGetBAR1MemoryInfo,
    handle_nvmlDeviceGetViolationStatus,
    handle_nvmlDeviceGetIrqNum,
    handle_nvmlDeviceGetNumGpuCores,
    handle_nvmlDeviceGetPowerSource,
    handle_nvmlDeviceGetMemoryBusWidth,
    handle_nvmlDeviceGetPcieLinkMaxSpeed,
    handle_nvmlDeviceGetPcieSpeed,
    handle_nvmlDeviceGetAdaptiveClockInfoStatus,
    handle_nvmlDeviceGetAccountingMode,
    handle_nvmlDeviceGetAccountingStats,
    handle_nvmlDeviceGetAccountingPids,
    handle_nvmlDeviceGetAccountingBufferSize,
    handle_nvmlDeviceGetRetiredPages,
    handle_nvmlDeviceGetRetiredPages_v2,
    handle_nvmlDeviceGetRetiredPagesPendingStatus,
    handle_nvmlDeviceGetRemappedRows,
    handle_nvmlDeviceGetRowRemapperHistogram,
    handle_nvmlDeviceGetArchitecture,
    handle_nvmlUnitSetLedState,
    handle_nvmlDeviceSetPersistenceMode,
    handle_nvmlDeviceSetComputeMode,
    handle_nvmlDeviceSetEccMode,
    handle_nvmlDeviceClearEccErrorCounts,
    handle_nvmlDeviceSetDriverModel,
    handle_nvmlDeviceSetGpuLockedClocks,
    handle_nvmlDeviceResetGpuLockedClocks,
    handle_nvmlDeviceSetMemoryLockedClocks,
    handle_nvmlDeviceResetMemoryLockedClocks,
    handle_nvmlDeviceSetApplicationsClocks,
    handle_nvmlDeviceGetClkMonStatus,
    handle_nvmlDeviceSetPowerManagementLimit,
    handle_nvmlDeviceSetGpuOperationMode,
    handle_nvmlDeviceSetAPIRestriction,
    handle_nvmlDeviceSetAccountingMode,
    handle_nvmlDeviceClearAccountingPids,
    handle_nvmlDeviceGetNvLinkState,
    handle_nvmlDeviceGetNvLinkVersion,
    handle_nvmlDeviceGetNvLinkCapability,
    handle_nvmlDeviceGetNvLinkRemotePciInfo_v2,
    handle_nvmlDeviceGetNvLinkErrorCounter,
    handle_nvmlDeviceResetNvLinkErrorCounters,
    handle_nvmlDeviceSetNvLinkUtilizationControl,
    handle_nvmlDeviceGetNvLinkUtilizationControl,
    handle_nvmlDeviceGetNvLinkUtilizationCounter,
    handle_nvmlDeviceFreezeNvLinkUtilizationCounter,
    handle_nvmlDeviceResetNvLinkUtilizationCounter,
    handle_nvmlDeviceGetNvLinkRemoteDeviceType,
    handle_nvmlEventSetCreate,
    handle_nvmlDeviceRegisterEvents,
    handle_nvmlDeviceGetSupportedEventTypes,
    handle_nvmlEventSetWait_v2,
    handle_nvmlEventSetFree,
    handle_nvmlDeviceModifyDrainState,
    handle_nvmlDeviceQueryDrainState,
    handle_nvmlDeviceRemoveGpu_v2,
    handle_nvmlDeviceDiscoverGpus,
    handle_nvmlDeviceGetFieldValues,
    handle_nvmlDeviceClearFieldValues,
    handle_nvmlDeviceGetVirtualizationMode,
    handle_nvmlDeviceGetHostVgpuMode,
    handle_nvmlDeviceSetVirtualizationMode,
    handle_nvmlDeviceGetGridLicensableFeatures_v4,
    handle_nvmlDeviceGetProcessUtilization,
    handle_nvmlDeviceGetGspFirmwareVersion,
    handle_nvmlDeviceGetGspFirmwareMode,
    handle_nvmlGetVgpuDriverCapabilities,
    handle_nvmlDeviceGetVgpuCapabilities,
    handle_nvmlDeviceGetSupportedVgpus,
    handle_nvmlDeviceGetCreatableVgpus,
    handle_nvmlVgpuTypeGetClass,
    handle_nvmlVgpuTypeGetName,
    handle_nvmlVgpuTypeGetGpuInstanceProfileId,
    handle_nvmlVgpuTypeGetDeviceID,
    handle_nvmlVgpuTypeGetFramebufferSize,
    handle_nvmlVgpuTypeGetNumDisplayHeads,
    handle_nvmlVgpuTypeGetResolution,
    handle_nvmlVgpuTypeGetLicense,
    handle_nvmlVgpuTypeGetFrameRateLimit,
    handle_nvmlVgpuTypeGetMaxInstances,
    handle_nvmlVgpuTypeGetMaxInstancesPerVm,
    handle_nvmlDeviceGetActiveVgpus,
    handle_nvmlVgpuInstanceGetVmID,
    handle_nvmlVgpuInstanceGetUUID,
    handle_nvmlVgpuInstanceGetVmDriverVersion,
    handle_nvmlVgpuInstanceGetFbUsage,
    handle_nvmlVgpuInstanceGetLicenseStatus,
    handle_nvmlVgpuInstanceGetType,
    handle_nvmlVgpuInstanceGetFrameRateLimit,
    handle_nvmlVgpuInstanceGetEccMode,
    handle_nvmlVgpuInstanceGetEncoderCapacity,
    handle_nvmlVgpuInstanceSetEncoderCapacity,
    handle_nvmlVgpuInstanceGetEncoderStats,
    handle_nvmlVgpuInstanceGetEncoderSessions,
    handle_nvmlVgpuInstanceGetFBCStats,
    handle_nvmlVgpuInstanceGetFBCSessions,
    handle_nvmlVgpuInstanceGetGpuInstanceId,
    handle_nvmlVgpuInstanceGetGpuPciId,
    handle_nvmlVgpuTypeGetCapabilities,
    handle_nvmlVgpuInstanceGetMetadata,
    handle_nvmlDeviceGetVgpuMetadata,
    handle_nvmlGetVgpuCompatibility,
    handle_nvmlDeviceGetPgpuMetadataString,
    handle_nvmlDeviceGetVgpuSchedulerLog,
    handle_nvmlDeviceGetVgpuSchedulerState,
    handle_nvmlDeviceGetVgpuSchedulerCapabilities,
    handle_nvmlGetVgpuVersion,
    handle_nvmlSetVgpuVersion,
    handle_nvmlDeviceGetVgpuUtilization,
    handle_nvmlDeviceGetVgpuProcessUtilization,
    handle_nvmlVgpuInstanceGetAccountingMode,
    handle_nvmlVgpuInstanceGetAccountingPids,
    handle_nvmlVgpuInstanceGetAccountingStats,
    handle_nvmlVgpuInstanceClearAccountingPids,
    handle_nvmlVgpuInstanceGetLicenseInfo_v2,
    handle_nvmlGetExcludedDeviceCount,
    handle_nvmlGetExcludedDeviceInfoByIndex,
    handle_nvmlDeviceSetMigMode,
    handle_nvmlDeviceGetMigMode,
    handle_nvmlDeviceGetGpuInstanceProfileInfo,
    handle_nvmlDeviceGetGpuInstanceProfileInfoV,
    handle_nvmlDeviceGetGpuInstancePossiblePlacements_v2,
    handle_nvmlDeviceGetGpuInstanceRemainingCapacity,
    handle_nvmlDeviceCreateGpuInstance,
    handle_nvmlDeviceCreateGpuInstanceWithPlacement,
    handle_nvmlGpuInstanceDestroy,
    handle_nvmlDeviceGetGpuInstances,
    handle_nvmlDeviceGetGpuInstanceById,
    handle_nvmlGpuInstanceGetInfo,
    handle_nvmlGpuInstanceGetComputeInstanceProfileInfo,
    handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV,
    handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity,
    handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements,
    handle_nvmlGpuInstanceCreateComputeInstance,
    handle_nvmlGpuInstanceCreateComputeInstanceWithPlacement,
    handle_nvmlComputeInstanceDestroy,
    handle_nvmlGpuInstanceGetComputeInstances,
    handle_nvmlGpuInstanceGetComputeInstanceById,
    handle_nvmlComputeInstanceGetInfo_v2,
    handle_nvmlDeviceIsMigDeviceHandle,
    handle_nvmlDeviceGetGpuInstanceId,
    handle_nvmlDeviceGetComputeInstanceId,
    handle_nvmlDeviceGetMaxMigDeviceCount,
    handle_nvmlDeviceGetMigDeviceHandleByIndex,
    handle_nvmlDeviceGetDeviceHandleFromMigDeviceHandle,
    handle_nvmlDeviceGetBusType,
    handle_nvmlDeviceGetDynamicPstatesInfo,
    handle_nvmlDeviceSetFanSpeed_v2,
    handle_nvmlDeviceGetGpcClkVfOffset,
    handle_nvmlDeviceSetGpcClkVfOffset,
    handle_nvmlDeviceGetMemClkVfOffset,
    handle_nvmlDeviceSetMemClkVfOffset,
    handle_nvmlDeviceGetMinMaxClockOfPState,
    handle_nvmlDeviceGetSupportedPerformanceStates,
    handle_nvmlDeviceGetGpcClkMinMaxVfOffset,
    handle_nvmlDeviceGetMemClkMinMaxVfOffset,
    handle_nvmlDeviceGetGpuFabricInfo,
    handle_nvmlGpmMetricsGet,
    handle_nvmlGpmSampleFree,
    handle_nvmlGpmSampleAlloc,
    handle_nvmlGpmSampleGet,
    handle_nvmlGpmMigSampleGet,
    handle_nvmlGpmQueryDeviceSupport,
    handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold,
};

RequestHandler get_handler(const int op)
{
    return opHandlers[op];
}
