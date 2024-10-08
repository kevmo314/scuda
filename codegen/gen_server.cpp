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
    if (rpc_write(conn, version, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, version, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, name, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, devices, deviceCount * sizeof(nvmlDevice_t)) < 0)
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
    if (rpc_write(conn, hwbcEntries, hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
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
    const char* serial = (const char*)malloc(serial_len);
    if (rpc_read(conn, &serial, sizeof(const char*)) < 0)
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
    const char* uuid = (const char*)malloc(uuid_len);
    if (rpc_read(conn, &uuid, sizeof(const char*)) < 0)
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
    const char* pciBusId = (const char*)malloc(pciBusId_len);
    if (rpc_read(conn, &pciBusId, sizeof(const char*)) < 0)
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
    if (rpc_write(conn, name, length * sizeof(char)) < 0)
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
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* serial = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSerial(device, serial, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, serial, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, nodeSet, nodeSetSize * sizeof(unsigned long)) < 0)
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
    if (rpc_write(conn, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0)
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
    if (rpc_write(conn, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0)
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
    if (rpc_write(conn, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
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
    if (rpc_write(conn, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
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
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* uuid = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetUUID(device, uuid, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetMdevUUID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;
    unsigned int size;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;
    char* mdevUuid = (char*)malloc(size * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, mdevUuid, size * sizeof(char)) < 0)
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
    if (rpc_write(conn, partNumber, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, version, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, version, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, clocksMHz, count * sizeof(unsigned int)) < 0)
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
    if (rpc_write(conn, clocksMHz, count * sizeof(unsigned int)) < 0)
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
    if (rpc_write(conn, sessionInfos, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
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
    if (rpc_write(conn, sessionInfo, sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0)
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
    unsigned int length;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;
    char* version = (char*)malloc(length * sizeof(char));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVbiosVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, version, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
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
    if (rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
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
    if (rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
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
    if (rpc_write(conn, samples, sampleCount * sizeof(nvmlSample_t)) < 0)
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
    if (rpc_write(conn, pids, count * sizeof(unsigned int)) < 0)
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
    if (rpc_write(conn, addresses, pageCount * sizeof(unsigned long long)) < 0)
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
    if (rpc_write(conn, addresses, pageCount * sizeof(unsigned long long)) < 0)
        return -1;
    if (rpc_write(conn, timestamps, pageCount * sizeof(unsigned long long)) < 0)
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
    if (rpc_write(conn, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0)
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
    if (rpc_write(conn, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0)
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
    if (rpc_write(conn, utilization, processSamplesCount * sizeof(nvmlProcessUtilizationSample_t)) < 0)
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
    if (rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0)
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
    if (rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0)
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
    if (rpc_write(conn, vgpuTypeClass, size * sizeof(char)) < 0)
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
    if (rpc_write(conn, vgpuTypeName, size * sizeof(char)) < 0)
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
    if (rpc_write(conn, vgpuTypeLicenseString, size * sizeof(char)) < 0)
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
    if (rpc_write(conn, vgpuInstances, vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0)
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
    if (rpc_write(conn, vmId, size * sizeof(char)) < 0)
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
    if (rpc_write(conn, uuid, size * sizeof(char)) < 0)
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
    if (rpc_write(conn, version, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, sessionInfo, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
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
    if (rpc_write(conn, sessionInfo, sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0)
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
    if (rpc_write(conn, vgpuPciId, length * sizeof(char)) < 0)
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
    if (rpc_write(conn, vgpuMetadata, bufferSize * sizeof(nvmlVgpuMetadata_t)) < 0)
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
    if (rpc_write(conn, pgpuMetadata, bufferSize * sizeof(nvmlVgpuPgpuMetadata_t)) < 0)
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
    if (rpc_write(conn, pgpuMetadata, bufferSize * sizeof(char)) < 0)
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
    if (rpc_write(conn, utilizationSamples, vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0)
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
    if (rpc_write(conn, utilizationSamples, vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t)) < 0)
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
    if (rpc_write(conn, pids, count * sizeof(unsigned int)) < 0)
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
    if (rpc_write(conn, placements, count * sizeof(nvmlGpuInstancePlacement_t)) < 0)
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
    if (rpc_write(conn, gpuInstances, count * sizeof(nvmlGpuInstance_t)) < 0)
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
    if (rpc_write(conn, placements, count * sizeof(nvmlComputeInstancePlacement_t)) < 0)
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
    if (rpc_write(conn, computeInstances, count * sizeof(nvmlComputeInstance_t)) < 0)
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
    if (rpc_write(conn, pstates, size * sizeof(nvmlPstates_t)) < 0)
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

int handle_cuInit(void *conn)
{
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuInit(Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDriverGetVersion(void *conn)
{
    int driverVersion;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDriverGetVersion(&driverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &driverVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGet(void *conn)
{
    CUdevice device;
    int ordinal;
    if (rpc_read(conn, &ordinal, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGet(&device, ordinal);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(CUdevice)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetCount(void *conn)
{
    int count;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetCount(&count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetName(void *conn)
{
    int len;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        return -1;
    char* name = (char*)malloc(len * sizeof(char));
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetName(name, len, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, name, len * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetUuid(void *conn)
{
    CUuuid* uuid = (CUuuid*)malloc(16);
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetUuid(uuid, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, 16) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetUuid_v2(void *conn)
{
    CUuuid* uuid = (CUuuid*)malloc(16);
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetUuid_v2(uuid, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, uuid, 16) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetLuid(void *conn)
{
    std::size_t luid_len;
    if (rpc_read(conn, &luid_len, sizeof(luid_len)) < 0)
        return -1;
    char* luid = (char*)malloc(luid_len);
    unsigned int deviceNodeMask;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetLuid(luid, &deviceNodeMask, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &luid_len, sizeof(luid_len)) < 0)
        return -1;
    if (rpc_write(conn, luid, luid_len) < 0)
        return -1;
    if (rpc_write(conn, &deviceNodeMask, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceTotalMem_v2(void *conn)
{
    std::size_t bytes;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceTotalMem_v2(&bytes, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetTexture1DLinearMaxWidth(void *conn)
{
    std::size_t maxWidthInElements;
    CUarray_format format;
    if (rpc_read(conn, &format, sizeof(CUarray_format)) < 0)
        return -1;
    unsigned numChannels;
    if (rpc_read(conn, &numChannels, sizeof(unsigned)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetTexture1DLinearMaxWidth(&maxWidthInElements, format, numChannels, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &maxWidthInElements, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetAttribute(void *conn)
{
    int pi;
    CUdevice_attribute attrib;
    if (rpc_read(conn, &attrib, sizeof(CUdevice_attribute)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetAttribute(&pi, attrib, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceSetMemPool(void *conn)
{
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceSetMemPool(dev, pool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetMemPool(void *conn)
{
    CUmemoryPool pool;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetMemPool(&pool, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetDefaultMemPool(void *conn)
{
    CUmemoryPool pool_out;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetDefaultMemPool(&pool_out, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool_out, sizeof(CUmemoryPool)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetExecAffinitySupport(void *conn)
{
    int pi;
    CUexecAffinityType type;
    if (rpc_read(conn, &type, sizeof(CUexecAffinityType)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetExecAffinitySupport(&pi, type, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuFlushGPUDirectRDMAWrites(void *conn)
{
    CUflushGPUDirectRDMAWritesTarget target;
    if (rpc_read(conn, &target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0)
        return -1;
    CUflushGPUDirectRDMAWritesScope scope;
    if (rpc_read(conn, &scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFlushGPUDirectRDMAWrites(target, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetProperties(void *conn)
{
    CUdevprop prop;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetProperties(&prop, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(CUdevprop)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceComputeCapability(void *conn)
{
    int major;
    int minor;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceComputeCapability(&major, &minor, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &major, sizeof(int)) < 0)
        return -1;
    if (rpc_write(conn, &minor, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuDevicePrimaryCtxRetain(void *conn)
{
    CUcontext pctx;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDevicePrimaryCtxRetain(&pctx, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    return result;
}

int handle_cuDevicePrimaryCtxRelease_v2(void *conn)
{
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDevicePrimaryCtxRelease_v2(dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDevicePrimaryCtxSetFlags_v2(void *conn)
{
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDevicePrimaryCtxSetFlags_v2(dev, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDevicePrimaryCtxGetState(void *conn)
{
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;
    unsigned int flags;
    int active;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDevicePrimaryCtxGetState(dev, &flags, &active);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &active, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuDevicePrimaryCtxReset_v2(void *conn)
{
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDevicePrimaryCtxReset_v2(dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxCreate_v2(void *conn)
{
    CUcontext pctx;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxCreate_v2(&pctx, flags, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    return result;
}

int handle_cuCtxCreate_v3(void *conn)
{
    CUcontext pctx;
    int numParams;
    if (rpc_read(conn, &numParams, sizeof(int)) < 0)
        return -1;
    CUexecAffinityParam* paramsArray = (CUexecAffinityParam*)malloc(numParams * sizeof(CUexecAffinityParam));
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxCreate_v3(&pctx, paramsArray, numParams, flags, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;
    if (rpc_write(conn, paramsArray, numParams * sizeof(CUexecAffinityParam)) < 0)
        return -1;

    return result;
}

int handle_cuCtxDestroy_v2(void *conn)
{
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxDestroy_v2(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxPushCurrent_v2(void *conn)
{
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxPushCurrent_v2(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxPopCurrent_v2(void *conn)
{
    CUcontext pctx;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxPopCurrent_v2(&pctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    return result;
}

int handle_cuCtxSetCurrent(void *conn)
{
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxSetCurrent(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetCurrent(void *conn)
{
    CUcontext pctx;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetCurrent(&pctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetDevice(void *conn)
{
    CUdevice device;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetDevice(&device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(CUdevice)) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetFlags(void *conn)
{
    unsigned int flags;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetFlags(&flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetId(void *conn)
{
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        return -1;
    unsigned long long ctxId;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetId(ctx, &ctxId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ctxId, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_cuCtxSynchronize(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxSynchronize();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxSetLimit(void *conn)
{
    CUlimit limit;
    if (rpc_read(conn, &limit, sizeof(CUlimit)) < 0)
        return -1;
    std::size_t value;
    if (rpc_read(conn, &value, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetLimit(void *conn)
{
    std::size_t pvalue;
    CUlimit limit;
    if (rpc_read(conn, &limit, sizeof(CUlimit)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetLimit(&pvalue, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pvalue, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetCacheConfig(void *conn)
{
    CUfunc_cache pconfig;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetCacheConfig(&pconfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pconfig, sizeof(CUfunc_cache)) < 0)
        return -1;

    return result;
}

int handle_cuCtxSetCacheConfig(void *conn)
{
    CUfunc_cache config;
    if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxSetCacheConfig(config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetSharedMemConfig(void *conn)
{
    CUsharedconfig pConfig;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetSharedMemConfig(&pConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pConfig, sizeof(CUsharedconfig)) < 0)
        return -1;

    return result;
}

int handle_cuCtxSetSharedMemConfig(void *conn)
{
    CUsharedconfig config;
    if (rpc_read(conn, &config, sizeof(CUsharedconfig)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxSetSharedMemConfig(config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetApiVersion(void *conn)
{
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        return -1;
    unsigned int version;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetApiVersion(ctx, &version);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &version, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetStreamPriorityRange(void *conn)
{
    int leastPriority;
    int greatestPriority;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &leastPriority, sizeof(int)) < 0)
        return -1;
    if (rpc_write(conn, &greatestPriority, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuCtxResetPersistingL2Cache(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxResetPersistingL2Cache();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxGetExecAffinity(void *conn)
{
    CUexecAffinityParam pExecAffinity;
    CUexecAffinityType type;
    if (rpc_read(conn, &type, sizeof(CUexecAffinityType)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxGetExecAffinity(&pExecAffinity, type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pExecAffinity, sizeof(CUexecAffinityParam)) < 0)
        return -1;

    return result;
}

int handle_cuCtxAttach(void *conn)
{
    CUcontext pctx;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxAttach(&pctx, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    return result;
}

int handle_cuCtxDetach(void *conn)
{
    CUcontext ctx;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxDetach(ctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuModuleLoad(void *conn)
{
    CUmodule module;
    std::size_t fname_len;
    if (rpc_read(conn, &fname_len, sizeof(fname_len)) < 0)
        return -1;
    const char* fname = (const char*)malloc(fname_len);
    if (rpc_read(conn, &fname, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleLoad(&module, fname);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &module, sizeof(CUmodule)) < 0)
        return -1;

    return result;
}

int handle_cuModuleUnload(void *conn)
{
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleUnload(hmod);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuModuleGetLoadingMode(void *conn)
{
    CUmoduleLoadingMode mode;
    if (rpc_read(conn, &mode, sizeof(CUmoduleLoadingMode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleGetLoadingMode(&mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(CUmoduleLoadingMode)) < 0)
        return -1;

    return result;
}

int handle_cuModuleGetFunction(void *conn)
{
    CUfunction hfunc;
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleGetFunction(&hfunc, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;

    return result;
}

int handle_cuModuleGetGlobal_v2(void *conn)
{
    CUdeviceptr dptr;
    std::size_t bytes;
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleGetGlobal_v2(&dptr, &bytes, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuLinkCreate_v2(void *conn)
{
    unsigned int numOptions;
    if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0)
        return -1;
    CUjit_option options;
    if (rpc_read(conn, &options, sizeof(CUjit_option)) < 0)
        return -1;
    void* optionValues;
    if (rpc_read(conn, &optionValues, sizeof(void*)) < 0)
        return -1;
    CUlinkState stateOut;
    if (rpc_read(conn, &stateOut, sizeof(CUlinkState)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLinkCreate_v2(numOptions, &options, &optionValues, &stateOut);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &options, sizeof(CUjit_option)) < 0)
        return -1;
    if (rpc_write(conn, &optionValues, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &stateOut, sizeof(CUlinkState)) < 0)
        return -1;

    return result;
}

int handle_cuLinkAddData_v2(void *conn)
{
    CUlinkState state;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        return -1;
    CUjitInputType type;
    if (rpc_read(conn, &type, sizeof(CUjitInputType)) < 0)
        return -1;
    void* data;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;
    unsigned int numOptions;
    if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0)
        return -1;
    CUjit_option* options = (CUjit_option*)malloc(numOptions * sizeof(CUjit_option));
    if (rpc_read(conn, &options, sizeof(CUjit_option*)) < 0)
        return -1;
    void** optionValues = (void**)malloc(numOptions * sizeof(void*));
    if (rpc_read(conn, &optionValues, sizeof(void**)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLinkAddData_v2(state, type, &data, size, name, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuLinkAddFile_v2(void *conn)
{
    CUlinkState state;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        return -1;
    CUjitInputType type;
    if (rpc_read(conn, &type, sizeof(CUjitInputType)) < 0)
        return -1;
    std::size_t path_len;
    if (rpc_read(conn, &path_len, sizeof(path_len)) < 0)
        return -1;
    const char* path = (const char*)malloc(path_len);
    if (rpc_read(conn, &path, sizeof(const char*)) < 0)
        return -1;
    unsigned int numOptions;
    if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0)
        return -1;
    CUjit_option* options = (CUjit_option*)malloc(numOptions * sizeof(CUjit_option));
    if (rpc_read(conn, &options, sizeof(CUjit_option*)) < 0)
        return -1;
    void** optionValues = (void**)malloc(numOptions * sizeof(void*));
    if (rpc_read(conn, &optionValues, sizeof(void**)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuLinkComplete(void *conn)
{
    CUlinkState state;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        return -1;
    void* cubinOut;
    std::size_t sizeOut;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLinkComplete(state, &cubinOut, &sizeOut);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &cubinOut, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &sizeOut, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuLinkDestroy(void *conn)
{
    CUlinkState state;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLinkDestroy(state);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuModuleGetTexRef(void *conn)
{
    CUtexref pTexRef;
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleGetTexRef(&pTexRef, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexRef, sizeof(CUtexref)) < 0)
        return -1;

    return result;
}

int handle_cuModuleGetSurfRef(void *conn)
{
    CUsurfref pSurfRef;
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuModuleGetSurfRef(&pSurfRef, hmod, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pSurfRef, sizeof(CUsurfref)) < 0)
        return -1;

    return result;
}

int handle_cuLibraryLoadFromFile(void *conn)
{
    CUlibrary library;
    std::size_t fileName_len;
    if (rpc_read(conn, &fileName_len, sizeof(fileName_len)) < 0)
        return -1;
    const char* fileName = (const char*)malloc(fileName_len);
    if (rpc_read(conn, &fileName, sizeof(const char*)) < 0)
        return -1;
    unsigned int numJitOptions;
    if (rpc_read(conn, &numJitOptions, sizeof(unsigned int)) < 0)
        return -1;
    CUjit_option* jitOptions = (CUjit_option*)malloc(numJitOptions * sizeof(CUjit_option));
    if (rpc_read(conn, &jitOptions, sizeof(CUjit_option*)) < 0)
        return -1;
    void** jitOptionsValues = (void**)malloc(numJitOptions * sizeof(void*));
    if (rpc_read(conn, &jitOptionsValues, sizeof(void**)) < 0)
        return -1;
    unsigned int numLibraryOptions;
    if (rpc_read(conn, &numLibraryOptions, sizeof(unsigned int)) < 0)
        return -1;
    CUlibraryOption* libraryOptions = (CUlibraryOption*)malloc(numLibraryOptions * sizeof(CUlibraryOption));
    if (rpc_read(conn, &libraryOptions, sizeof(CUlibraryOption*)) < 0)
        return -1;
    void** libraryOptionValues = (void**)malloc(numLibraryOptions * sizeof(void*));
    if (rpc_read(conn, &libraryOptionValues, sizeof(void**)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryLoadFromFile(&library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;

    return result;
}

int handle_cuLibraryUnload(void *conn)
{
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryUnload(library);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuLibraryGetKernel(void *conn)
{
    CUkernel pKernel;
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryGetKernel(&pKernel, library, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pKernel, sizeof(CUkernel)) < 0)
        return -1;

    return result;
}

int handle_cuLibraryGetModule(void *conn)
{
    CUmodule pMod;
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryGetModule(&pMod, library);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pMod, sizeof(CUmodule)) < 0)
        return -1;

    return result;
}

int handle_cuKernelGetFunction(void *conn)
{
    CUfunction pFunc;
    CUkernel kernel;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuKernelGetFunction(&pFunc, kernel);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFunc, sizeof(CUfunction)) < 0)
        return -1;

    return result;
}

int handle_cuLibraryGetGlobal(void *conn)
{
    CUdeviceptr dptr;
    std::size_t bytes;
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryGetGlobal(&dptr, &bytes, library, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuLibraryGetManaged(void *conn)
{
    CUdeviceptr dptr;
    std::size_t bytes;
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;
    std::size_t name_len;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        return -1;
    const char* name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryGetManaged(&dptr, &bytes, library, name);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &bytes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuLibraryGetUnifiedFunction(void *conn)
{
    void* fptr;
    CUlibrary library;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        return -1;
    std::size_t symbol_len;
    if (rpc_read(conn, &symbol_len, sizeof(symbol_len)) < 0)
        return -1;
    const char* symbol = (const char*)malloc(symbol_len);
    if (rpc_read(conn, &symbol, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLibraryGetUnifiedFunction(&fptr, library, symbol);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &fptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuKernelGetAttribute(void *conn)
{
    int pi;
    if (rpc_read(conn, &pi, sizeof(int)) < 0)
        return -1;
    CUfunction_attribute attrib;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        return -1;
    CUkernel kernel;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuKernelGetAttribute(&pi, attrib, kernel, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuKernelSetAttribute(void *conn)
{
    CUfunction_attribute attrib;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        return -1;
    int val;
    if (rpc_read(conn, &val, sizeof(int)) < 0)
        return -1;
    CUkernel kernel;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuKernelSetAttribute(attrib, val, kernel, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuKernelSetCacheConfig(void *conn)
{
    CUkernel kernel;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        return -1;
    CUfunc_cache config;
    if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuKernelSetCacheConfig(kernel, config, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemGetInfo_v2(void *conn)
{
    std::size_t free;
    if (rpc_read(conn, &free, sizeof(size_t)) < 0)
        return -1;
    std::size_t total;
    if (rpc_read(conn, &total, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemGetInfo_v2(&free, &total);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &free, sizeof(size_t)) < 0)
        return -1;
    if (rpc_write(conn, &total, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuMemAlloc_v2(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAlloc_v2(&dptr, bytesize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuMemAllocPitch_v2(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t pPitch;
    if (rpc_read(conn, &pPitch, sizeof(size_t)) < 0)
        return -1;
    std::size_t WidthInBytes;
    if (rpc_read(conn, &WidthInBytes, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;
    unsigned int ElementSizeBytes;
    if (rpc_read(conn, &ElementSizeBytes, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAllocPitch_v2(&dptr, &pPitch, WidthInBytes, Height, ElementSizeBytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &pPitch, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuMemFree_v2(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemFree_v2(dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemGetAddressRange_v2(void *conn)
{
    CUdeviceptr pbase;
    if (rpc_read(conn, &pbase, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t psize;
    if (rpc_read(conn, &psize, sizeof(size_t)) < 0)
        return -1;
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemGetAddressRange_v2(&pbase, &psize, dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pbase, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &psize, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuMemAllocHost_v2(void *conn)
{
    void* pp;
    if (rpc_read(conn, &pp, sizeof(void*)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAllocHost_v2(&pp, bytesize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pp, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemFreeHost(void *conn)
{
    void* p;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemFreeHost(&p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemHostAlloc(void *conn)
{
    void* pp;
    if (rpc_read(conn, &pp, sizeof(void*)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemHostAlloc(&pp, bytesize, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pp, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemHostGetDevicePointer_v2(void *conn)
{
    CUdeviceptr pdptr;
    if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    void* p;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemHostGetDevicePointer_v2(&pdptr, &p, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemHostGetFlags(void *conn)
{
    unsigned int pFlags;
    if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0)
        return -1;
    void* p;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemHostGetFlags(&pFlags, &p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemAllocManaged(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAllocManaged(&dptr, bytesize, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetByPCIBusId(void *conn)
{
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;
    std::size_t pciBusId_len;
    if (rpc_read(conn, &pciBusId_len, sizeof(pciBusId_len)) < 0)
        return -1;
    const char* pciBusId = (const char*)malloc(pciBusId_len);
    if (rpc_read(conn, &pciBusId, sizeof(const char*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetByPCIBusId(&dev, pciBusId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetPCIBusId(void *conn)
{
    int len;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        return -1;
    char* pciBusId = (char*)malloc(len * sizeof(char));
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetPCIBusId(pciBusId, len, dev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, pciBusId, len * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_cuIpcGetEventHandle(void *conn)
{
    CUipcEventHandle pHandle;
    if (rpc_read(conn, &pHandle, sizeof(CUipcEventHandle)) < 0)
        return -1;
    CUevent event;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuIpcGetEventHandle(&pHandle, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(CUipcEventHandle)) < 0)
        return -1;

    return result;
}

int handle_cuIpcOpenEventHandle(void *conn)
{
    CUevent phEvent;
    if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0)
        return -1;
    CUipcEventHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUipcEventHandle)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuIpcOpenEventHandle(&phEvent, handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phEvent, sizeof(CUevent)) < 0)
        return -1;

    return result;
}

int handle_cuIpcGetMemHandle(void *conn)
{
    CUipcMemHandle pHandle;
    if (rpc_read(conn, &pHandle, sizeof(CUipcMemHandle)) < 0)
        return -1;
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuIpcGetMemHandle(&pHandle, dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHandle, sizeof(CUipcMemHandle)) < 0)
        return -1;

    return result;
}

int handle_cuIpcOpenMemHandle_v2(void *conn)
{
    CUdeviceptr pdptr;
    if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUipcMemHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUipcMemHandle)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuIpcOpenMemHandle_v2(&pdptr, handle, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuIpcCloseMemHandle(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuIpcCloseMemHandle(dptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemHostRegister_v2(void *conn)
{
    void* p;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemHostRegister_v2(&p, bytesize, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemHostUnregister(void *conn)
{
    void* p;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemHostUnregister(&p);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &p, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemcpy(void *conn)
{
    CUdeviceptr dst;
    if (rpc_read(conn, &dst, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUdeviceptr src;
    if (rpc_read(conn, &src, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpy(dst, src, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyPeer(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUcontext dstContext;
    if (rpc_read(conn, &dstContext, sizeof(CUcontext)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUcontext srcContext;
    if (rpc_read(conn, &srcContext, sizeof(CUcontext)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyDtoH_v2(void *conn)
{
    void* dstHost;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyDtoH_v2(&dstHost, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyDtoD_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyDtoA_v2(void *conn)
{
    CUarray dstArray;
    if (rpc_read(conn, &dstArray, sizeof(CUarray)) < 0)
        return -1;
    std::size_t dstOffset;
    if (rpc_read(conn, &dstOffset, sizeof(size_t)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyAtoD_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUarray srcArray;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        return -1;
    std::size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyAtoH_v2(void *conn)
{
    void* dstHost;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        return -1;
    CUarray srcArray;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        return -1;
    std::size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyAtoH_v2(&dstHost, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyAtoA_v2(void *conn)
{
    CUarray dstArray;
    if (rpc_read(conn, &dstArray, sizeof(CUarray)) < 0)
        return -1;
    std::size_t dstOffset;
    if (rpc_read(conn, &dstOffset, sizeof(size_t)) < 0)
        return -1;
    CUarray srcArray;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        return -1;
    std::size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyAsync(void *conn)
{
    CUdeviceptr dst;
    if (rpc_read(conn, &dst, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUdeviceptr src;
    if (rpc_read(conn, &src, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyAsync(dst, src, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyPeerAsync(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUcontext dstContext;
    if (rpc_read(conn, &dstContext, sizeof(CUcontext)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUcontext srcContext;
    if (rpc_read(conn, &srcContext, sizeof(CUcontext)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyDtoHAsync_v2(void *conn)
{
    void* dstHost;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyDtoHAsync_v2(&dstHost, srcDevice, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyDtoDAsync_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemcpyAtoHAsync_v2(void *conn)
{
    void* dstHost;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        return -1;
    CUarray srcArray;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        return -1;
    std::size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t ByteCount;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemcpyAtoHAsync_v2(&dstHost, srcArray, srcOffset, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dstHost, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD8_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned char uc;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        return -1;
    std::size_t N;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD8_v2(dstDevice, uc, N);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD16_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned short us;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        return -1;
    std::size_t N;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD16_v2(dstDevice, us, N);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD32_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned int ui;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        return -1;
    std::size_t N;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD32_v2(dstDevice, ui, N);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD2D8_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned char uc;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        return -1;
    std::size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD2D16_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned short us;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        return -1;
    std::size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD2D32_v2(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned int ui;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        return -1;
    std::size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD8Async(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned char uc;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        return -1;
    std::size_t N;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD8Async(dstDevice, uc, N, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD16Async(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned short us;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        return -1;
    std::size_t N;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD16Async(dstDevice, us, N, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD32Async(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned int ui;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        return -1;
    std::size_t N;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD32Async(dstDevice, ui, N, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD2D8Async(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned char uc;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        return -1;
    std::size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD2D16Async(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned short us;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        return -1;
    std::size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemsetD2D32Async(void *conn)
{
    CUdeviceptr dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned int ui;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        return -1;
    std::size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    std::size_t Height;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuArrayGetDescriptor_v2(void *conn)
{
    CUDA_ARRAY_DESCRIPTOR pArrayDescriptor;
    if (rpc_read(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0)
        return -1;
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuArrayGetDescriptor_v2(&pArrayDescriptor, hArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0)
        return -1;

    return result;
}

int handle_cuArrayGetSparseProperties(void *conn)
{
    CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties;
    if (rpc_read(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0)
        return -1;
    CUarray array;
    if (rpc_read(conn, &array, sizeof(CUarray)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuArrayGetSparseProperties(&sparseProperties, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0)
        return -1;

    return result;
}

int handle_cuMipmappedArrayGetSparseProperties(void *conn)
{
    CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties;
    if (rpc_read(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0)
        return -1;
    CUmipmappedArray mipmap;
    if (rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMipmappedArrayGetSparseProperties(&sparseProperties, mipmap);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0)
        return -1;

    return result;
}

int handle_cuArrayGetMemoryRequirements(void *conn)
{
    CUDA_ARRAY_MEMORY_REQUIREMENTS memoryRequirements;
    if (rpc_read(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0)
        return -1;
    CUarray array;
    if (rpc_read(conn, &array, sizeof(CUarray)) < 0)
        return -1;
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuArrayGetMemoryRequirements(&memoryRequirements, array, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0)
        return -1;

    return result;
}

int handle_cuMipmappedArrayGetMemoryRequirements(void *conn)
{
    CUDA_ARRAY_MEMORY_REQUIREMENTS memoryRequirements;
    if (rpc_read(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0)
        return -1;
    CUmipmappedArray mipmap;
    if (rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0)
        return -1;
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMipmappedArrayGetMemoryRequirements(&memoryRequirements, mipmap, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0)
        return -1;

    return result;
}

int handle_cuArrayGetPlane(void *conn)
{
    CUarray pPlaneArray;
    if (rpc_read(conn, &pPlaneArray, sizeof(CUarray)) < 0)
        return -1;
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        return -1;
    unsigned int planeIdx;
    if (rpc_read(conn, &planeIdx, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuArrayGetPlane(&pPlaneArray, hArray, planeIdx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pPlaneArray, sizeof(CUarray)) < 0)
        return -1;

    return result;
}

int handle_cuArrayDestroy(void *conn)
{
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuArrayDestroy(hArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuArray3DGetDescriptor_v2(void *conn)
{
    CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor;
    if (rpc_read(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0)
        return -1;
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuArray3DGetDescriptor_v2(&pArrayDescriptor, hArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0)
        return -1;

    return result;
}

int handle_cuMipmappedArrayGetLevel(void *conn)
{
    CUarray pLevelArray;
    if (rpc_read(conn, &pLevelArray, sizeof(CUarray)) < 0)
        return -1;
    CUmipmappedArray hMipmappedArray;
    if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;
    unsigned int level;
    if (rpc_read(conn, &level, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMipmappedArrayGetLevel(&pLevelArray, hMipmappedArray, level);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pLevelArray, sizeof(CUarray)) < 0)
        return -1;

    return result;
}

int handle_cuMipmappedArrayDestroy(void *conn)
{
    CUmipmappedArray hMipmappedArray;
    if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMipmappedArrayDestroy(hMipmappedArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemGetHandleForAddressRange(void *conn)
{
    void* handle;
    if (rpc_read(conn, &handle, sizeof(void*)) < 0)
        return -1;
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    CUmemRangeHandleType handleType;
    if (rpc_read(conn, &handleType, sizeof(CUmemRangeHandleType)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemGetHandleForAddressRange(&handle, dptr, size, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemAddressReserve(void *conn)
{
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    std::size_t alignment;
    if (rpc_read(conn, &alignment, sizeof(size_t)) < 0)
        return -1;
    CUdeviceptr addr;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAddressReserve(&ptr, size, alignment, addr, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuMemAddressFree(void *conn)
{
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAddressFree(ptr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemRelease(void *conn)
{
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemRelease(handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemMap(void *conn)
{
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    std::size_t offset;
    if (rpc_read(conn, &offset, sizeof(size_t)) < 0)
        return -1;
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemMap(ptr, size, offset, handle, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemMapArrayAsync(void *conn)
{
    CUarrayMapInfo mapInfoList;
    if (rpc_read(conn, &mapInfoList, sizeof(CUarrayMapInfo)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemMapArrayAsync(&mapInfoList, count, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mapInfoList, sizeof(CUarrayMapInfo)) < 0)
        return -1;

    return result;
}

int handle_cuMemUnmap(void *conn)
{
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemUnmap(ptr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemExportToShareableHandle(void *conn)
{
    void* shareableHandle;
    if (rpc_read(conn, &shareableHandle, sizeof(void*)) < 0)
        return -1;
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;
    CUmemAllocationHandleType handleType;
    if (rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemExportToShareableHandle(&shareableHandle, handle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &shareableHandle, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemImportFromShareableHandle(void *conn)
{
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;
    void* osHandle;
    if (rpc_read(conn, &osHandle, sizeof(void*)) < 0)
        return -1;
    CUmemAllocationHandleType shHandleType;
    if (rpc_read(conn, &shHandleType, sizeof(CUmemAllocationHandleType)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemImportFromShareableHandle(&handle, &osHandle, shHandleType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;
    if (rpc_write(conn, &osHandle, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemGetAllocationPropertiesFromHandle(void *conn)
{
    CUmemAllocationProp prop;
    if (rpc_read(conn, &prop, sizeof(CUmemAllocationProp)) < 0)
        return -1;
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemGetAllocationPropertiesFromHandle(&prop, handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(CUmemAllocationProp)) < 0)
        return -1;

    return result;
}

int handle_cuMemRetainAllocationHandle(void *conn)
{
    CUmemGenericAllocationHandle handle;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;
    void* addr;
    if (rpc_read(conn, &addr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemRetainAllocationHandle(&handle, &addr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        return -1;
    if (rpc_write(conn, &addr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemFreeAsync(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemFreeAsync(dptr, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemAllocAsync(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAllocAsync(&dptr, bytesize, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolTrimTo(void *conn)
{
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    std::size_t minBytesToKeep;
    if (rpc_read(conn, &minBytesToKeep, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolTrimTo(pool, minBytesToKeep);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolSetAttribute(void *conn)
{
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    CUmemPool_attribute attr;
    if (rpc_read(conn, &attr, sizeof(CUmemPool_attribute)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolSetAttribute(pool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolGetAttribute(void *conn)
{
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    CUmemPool_attribute attr;
    if (rpc_read(conn, &attr, sizeof(CUmemPool_attribute)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolGetAttribute(pool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolGetAccess(void *conn)
{
    CUmemAccess_flags flags;
    if (rpc_read(conn, &flags, sizeof(CUmemAccess_flags)) < 0)
        return -1;
    CUmemoryPool memPool;
    if (rpc_read(conn, &memPool, sizeof(CUmemoryPool)) < 0)
        return -1;
    CUmemLocation location;
    if (rpc_read(conn, &location, sizeof(CUmemLocation)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolGetAccess(&flags, memPool, &location);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(CUmemAccess_flags)) < 0)
        return -1;
    if (rpc_write(conn, &location, sizeof(CUmemLocation)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolDestroy(void *conn)
{
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolDestroy(pool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemAllocFromPoolAsync(void *conn)
{
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t bytesize;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        return -1;
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAllocFromPoolAsync(&dptr, bytesize, pool, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolExportToShareableHandle(void *conn)
{
    void* handle_out;
    if (rpc_read(conn, &handle_out, sizeof(void*)) < 0)
        return -1;
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    CUmemAllocationHandleType handleType;
    if (rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolExportToShareableHandle(&handle_out, pool, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle_out, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolImportFromShareableHandle(void *conn)
{
    CUmemoryPool pool_out;
    if (rpc_read(conn, &pool_out, sizeof(CUmemoryPool)) < 0)
        return -1;
    void* handle;
    if (rpc_read(conn, &handle, sizeof(void*)) < 0)
        return -1;
    CUmemAllocationHandleType handleType;
    if (rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolImportFromShareableHandle(&pool_out, &handle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pool_out, sizeof(CUmemoryPool)) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolExportPointer(void *conn)
{
    CUmemPoolPtrExportData shareData_out;
    if (rpc_read(conn, &shareData_out, sizeof(CUmemPoolPtrExportData)) < 0)
        return -1;
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolExportPointer(&shareData_out, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &shareData_out, sizeof(CUmemPoolPtrExportData)) < 0)
        return -1;

    return result;
}

int handle_cuMemPoolImportPointer(void *conn)
{
    CUdeviceptr ptr_out;
    if (rpc_read(conn, &ptr_out, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    CUmemPoolPtrExportData shareData;
    if (rpc_read(conn, &shareData, sizeof(CUmemPoolPtrExportData)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr_out, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &shareData, sizeof(CUmemPoolPtrExportData)) < 0)
        return -1;

    return result;
}

int handle_cuPointerGetAttribute(void *conn)
{
    void* data;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        return -1;
    CUpointer_attribute attribute;
    if (rpc_read(conn, &attribute, sizeof(CUpointer_attribute)) < 0)
        return -1;
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuPointerGetAttribute(&data, attribute, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemPrefetchAsync(void *conn)
{
    CUdeviceptr devPtr;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    CUdevice dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdevice)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemAdvise(void *conn)
{
    CUdeviceptr devPtr;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    CUmem_advise advice;
    if (rpc_read(conn, &advice, sizeof(CUmem_advise)) < 0)
        return -1;
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemAdvise(devPtr, count, advice, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuMemRangeGetAttribute(void *conn)
{
    void* data;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        return -1;
    std::size_t dataSize;
    if (rpc_read(conn, &dataSize, sizeof(size_t)) < 0)
        return -1;
    CUmem_range_attribute attribute;
    if (rpc_read(conn, &attribute, sizeof(CUmem_range_attribute)) < 0)
        return -1;
    CUdeviceptr devPtr;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemRangeGetAttribute(&data, dataSize, attribute, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuMemRangeGetAttributes(void *conn)
{
    void* data;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        return -1;
    std::size_t dataSizes;
    if (rpc_read(conn, &dataSizes, sizeof(size_t)) < 0)
        return -1;
    CUmem_range_attribute attributes;
    if (rpc_read(conn, &attributes, sizeof(CUmem_range_attribute)) < 0)
        return -1;
    std::size_t numAttributes;
    if (rpc_read(conn, &numAttributes, sizeof(size_t)) < 0)
        return -1;
    CUdeviceptr devPtr;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuMemRangeGetAttributes(&data, &dataSizes, &attributes, numAttributes, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &dataSizes, sizeof(size_t)) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(CUmem_range_attribute)) < 0)
        return -1;

    return result;
}

int handle_cuPointerGetAttributes(void *conn)
{
    unsigned int numAttributes;
    if (rpc_read(conn, &numAttributes, sizeof(unsigned int)) < 0)
        return -1;
    CUpointer_attribute attributes;
    if (rpc_read(conn, &attributes, sizeof(CUpointer_attribute)) < 0)
        return -1;
    void* data;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        return -1;
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuPointerGetAttributes(numAttributes, &attributes, &data, ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &attributes, sizeof(CUpointer_attribute)) < 0)
        return -1;
    if (rpc_write(conn, &data, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuStreamCreate(void *conn)
{
    CUstream phStream;
    if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamCreate(&phStream, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phStream, sizeof(CUstream)) < 0)
        return -1;

    return result;
}

int handle_cuStreamCreateWithPriority(void *conn)
{
    CUstream phStream;
    if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;
    int priority;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamCreateWithPriority(&phStream, flags, priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phStream, sizeof(CUstream)) < 0)
        return -1;

    return result;
}

int handle_cuStreamGetPriority(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    int priority;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamGetPriority(hStream, &priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &priority, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuStreamGetFlags(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamGetFlags(hStream, &flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cuStreamGetId(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    unsigned long long streamId;
    if (rpc_read(conn, &streamId, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamGetId(hStream, &streamId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &streamId, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_cuStreamGetCtx(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUcontext pctx;
    if (rpc_read(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamGetCtx(hStream, &pctx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pctx, sizeof(CUcontext)) < 0)
        return -1;

    return result;
}

int handle_cuStreamWaitEvent(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamWaitEvent(hStream, hEvent, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamAddCallback(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUstreamCallback callback;
    if (rpc_read(conn, &callback, sizeof(CUstreamCallback)) < 0)
        return -1;
    void* userData;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamAddCallback(hStream, callback, &userData, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuStreamBeginCapture_v2(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUstreamCaptureMode mode;
    if (rpc_read(conn, &mode, sizeof(CUstreamCaptureMode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamBeginCapture_v2(hStream, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuThreadExchangeStreamCaptureMode(void *conn)
{
    CUstreamCaptureMode mode;
    if (rpc_read(conn, &mode, sizeof(CUstreamCaptureMode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuThreadExchangeStreamCaptureMode(&mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0)
        return -1;

    return result;
}

int handle_cuStreamEndCapture(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUgraph phGraph;
    if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamEndCapture(hStream, &phGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0)
        return -1;

    return result;
}

int handle_cuStreamIsCapturing(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUstreamCaptureStatus captureStatus;
    if (rpc_read(conn, &captureStatus, sizeof(CUstreamCaptureStatus)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamIsCapturing(hStream, &captureStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &captureStatus, sizeof(CUstreamCaptureStatus)) < 0)
        return -1;

    return result;
}

int handle_cuStreamUpdateCaptureDependencies(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUgraphNode dependencies;
    if (rpc_read(conn, &dependencies, sizeof(CUgraphNode)) < 0)
        return -1;
    std::size_t numDependencies;
    if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamUpdateCaptureDependencies(hStream, &dependencies, numDependencies, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(CUgraphNode)) < 0)
        return -1;

    return result;
}

int handle_cuStreamAttachMemAsync(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t length;
    if (rpc_read(conn, &length, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamAttachMemAsync(hStream, dptr, length, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamQuery(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamQuery(hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamSynchronize(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamSynchronize(hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamDestroy_v2(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamDestroy_v2(hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamCopyAttributes(void *conn)
{
    CUstream dst;
    if (rpc_read(conn, &dst, sizeof(CUstream)) < 0)
        return -1;
    CUstream src;
    if (rpc_read(conn, &src, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamGetAttribute(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUstreamAttrID attr;
    if (rpc_read(conn, &attr, sizeof(CUstreamAttrID)) < 0)
        return -1;
    CUstreamAttrValue value_out;
    if (rpc_read(conn, &value_out, sizeof(CUstreamAttrValue)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamGetAttribute(hStream, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(CUstreamAttrValue)) < 0)
        return -1;

    return result;
}

int handle_cuEventCreate(void *conn)
{
    CUevent phEvent;
    if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventCreate(&phEvent, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phEvent, sizeof(CUevent)) < 0)
        return -1;

    return result;
}

int handle_cuEventRecord(void *conn)
{
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventRecord(hEvent, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuEventRecordWithFlags(void *conn)
{
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventRecordWithFlags(hEvent, hStream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuEventQuery(void *conn)
{
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventQuery(hEvent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuEventSynchronize(void *conn)
{
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventSynchronize(hEvent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuEventDestroy_v2(void *conn)
{
    CUevent hEvent;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventDestroy_v2(hEvent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuEventElapsedTime(void *conn)
{
    float pMilliseconds;
    if (rpc_read(conn, &pMilliseconds, sizeof(float)) < 0)
        return -1;
    CUevent hStart;
    if (rpc_read(conn, &hStart, sizeof(CUevent)) < 0)
        return -1;
    CUevent hEnd;
    if (rpc_read(conn, &hEnd, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuEventElapsedTime(&pMilliseconds, hStart, hEnd);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pMilliseconds, sizeof(float)) < 0)
        return -1;

    return result;
}

int handle_cuDestroyExternalMemory(void *conn)
{
    CUexternalMemory extMem;
    if (rpc_read(conn, &extMem, sizeof(CUexternalMemory)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDestroyExternalMemory(extMem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDestroyExternalSemaphore(void *conn)
{
    CUexternalSemaphore extSem;
    if (rpc_read(conn, &extSem, sizeof(CUexternalSemaphore)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDestroyExternalSemaphore(extSem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamWaitValue32_v2(void *conn)
{
    CUstream stream;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        return -1;
    CUdeviceptr addr;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        return -1;
    cuuint32_t value;
    if (rpc_read(conn, &value, sizeof(cuuint32_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamWaitValue32_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamWaitValue64_v2(void *conn)
{
    CUstream stream;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        return -1;
    CUdeviceptr addr;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        return -1;
    cuuint64_t value;
    if (rpc_read(conn, &value, sizeof(cuuint64_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamWaitValue64_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamWriteValue32_v2(void *conn)
{
    CUstream stream;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        return -1;
    CUdeviceptr addr;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        return -1;
    cuuint32_t value;
    if (rpc_read(conn, &value, sizeof(cuuint32_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamWriteValue32_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamWriteValue64_v2(void *conn)
{
    CUstream stream;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        return -1;
    CUdeviceptr addr;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        return -1;
    cuuint64_t value;
    if (rpc_read(conn, &value, sizeof(cuuint64_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamWriteValue64_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuStreamBatchMemOp_v2(void *conn)
{
    CUstream stream;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    CUstreamBatchMemOpParams paramArray;
    if (rpc_read(conn, &paramArray, sizeof(CUstreamBatchMemOpParams)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuStreamBatchMemOp_v2(stream, count, &paramArray, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &paramArray, sizeof(CUstreamBatchMemOpParams)) < 0)
        return -1;

    return result;
}

int handle_cuFuncGetAttribute(void *conn)
{
    int pi;
    if (rpc_read(conn, &pi, sizeof(int)) < 0)
        return -1;
    CUfunction_attribute attrib;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        return -1;
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncGetAttribute(&pi, attrib, hfunc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pi, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuFuncSetAttribute(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    CUfunction_attribute attrib;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncSetAttribute(hfunc, attrib, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuFuncSetCacheConfig(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    CUfunc_cache config;
    if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncSetCacheConfig(hfunc, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuFuncSetSharedMemConfig(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    CUsharedconfig config;
    if (rpc_read(conn, &config, sizeof(CUsharedconfig)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncSetSharedMemConfig(hfunc, config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuFuncGetModule(void *conn)
{
    CUmodule hmod;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncGetModule(&hmod, hfunc);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &hmod, sizeof(CUmodule)) < 0)
        return -1;

    return result;
}

int handle_cuLaunchKernel(void *conn)
{
    CUfunction f;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        return -1;
    unsigned int gridDimX;
    if (rpc_read(conn, &gridDimX, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int gridDimY;
    if (rpc_read(conn, &gridDimY, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int gridDimZ;
    if (rpc_read(conn, &gridDimZ, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int blockDimX;
    if (rpc_read(conn, &blockDimX, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int blockDimY;
    if (rpc_read(conn, &blockDimY, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int blockDimZ;
    if (rpc_read(conn, &blockDimZ, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int sharedMemBytes;
    if (rpc_read(conn, &sharedMemBytes, sizeof(unsigned int)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    void* kernelParams;
    if (rpc_read(conn, &kernelParams, sizeof(void*)) < 0)
        return -1;
    void* extra;
    if (rpc_read(conn, &extra, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams, &extra);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuLaunchCooperativeKernel(void *conn)
{
    CUfunction f;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        return -1;
    unsigned int gridDimX;
    if (rpc_read(conn, &gridDimX, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int gridDimY;
    if (rpc_read(conn, &gridDimY, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int gridDimZ;
    if (rpc_read(conn, &gridDimZ, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int blockDimX;
    if (rpc_read(conn, &blockDimX, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int blockDimY;
    if (rpc_read(conn, &blockDimY, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int blockDimZ;
    if (rpc_read(conn, &blockDimZ, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int sharedMemBytes;
    if (rpc_read(conn, &sharedMemBytes, sizeof(unsigned int)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    void* kernelParams;
    if (rpc_read(conn, &kernelParams, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &kernelParams, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuLaunchCooperativeKernelMultiDevice(void *conn)
{
    CUDA_LAUNCH_PARAMS launchParamsList;
    if (rpc_read(conn, &launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0)
        return -1;
    unsigned int numDevices;
    if (rpc_read(conn, &numDevices, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunchCooperativeKernelMultiDevice(&launchParamsList, numDevices, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuLaunchHostFunc(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUhostFn fn;
    if (rpc_read(conn, &fn, sizeof(CUhostFn)) < 0)
        return -1;
    void* userData;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunchHostFunc(hStream, fn, &userData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuFuncSetBlockShape(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    int x;
    if (rpc_read(conn, &x, sizeof(int)) < 0)
        return -1;
    int y;
    if (rpc_read(conn, &y, sizeof(int)) < 0)
        return -1;
    int z;
    if (rpc_read(conn, &z, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncSetBlockShape(hfunc, x, y, z);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuFuncSetSharedSize(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    unsigned int bytes;
    if (rpc_read(conn, &bytes, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuFuncSetSharedSize(hfunc, bytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuParamSetSize(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    unsigned int numbytes;
    if (rpc_read(conn, &numbytes, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuParamSetSize(hfunc, numbytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuParamSeti(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    int offset;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;
    unsigned int value;
    if (rpc_read(conn, &value, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuParamSeti(hfunc, offset, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuParamSetf(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    int offset;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;
    float value;
    if (rpc_read(conn, &value, sizeof(float)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuParamSetf(hfunc, offset, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuParamSetv(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    int offset;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    unsigned int numbytes;
    if (rpc_read(conn, &numbytes, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuParamSetv(hfunc, offset, &ptr, numbytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuLaunch(void *conn)
{
    CUfunction f;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunch(f);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuLaunchGrid(void *conn)
{
    CUfunction f;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        return -1;
    int grid_width;
    if (rpc_read(conn, &grid_width, sizeof(int)) < 0)
        return -1;
    int grid_height;
    if (rpc_read(conn, &grid_height, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunchGrid(f, grid_width, grid_height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuLaunchGridAsync(void *conn)
{
    CUfunction f;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        return -1;
    int grid_width;
    if (rpc_read(conn, &grid_width, sizeof(int)) < 0)
        return -1;
    int grid_height;
    if (rpc_read(conn, &grid_height, sizeof(int)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuLaunchGridAsync(f, grid_width, grid_height, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuParamSetTexRef(void *conn)
{
    CUfunction hfunc;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        return -1;
    int texunit;
    if (rpc_read(conn, &texunit, sizeof(int)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuParamSetTexRef(hfunc, texunit, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphCreate(void *conn)
{
    CUgraph phGraph;
    if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphCreate(&phGraph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0)
        return -1;

    return result;
}

int handle_cuGraphKernelNodeGetParams_v2(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphKernelNodeGetParams_v2(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphMemcpyNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_MEMCPY3D nodeParams;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_MEMCPY3D)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphMemcpyNodeGetParams(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(CUDA_MEMCPY3D)) < 0)
        return -1;

    return result;
}

int handle_cuGraphMemsetNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_MEMSET_NODE_PARAMS nodeParams;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphMemsetNodeGetParams(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphHostNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_HOST_NODE_PARAMS nodeParams;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphHostNodeGetParams(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphChildGraphNodeGetGraph(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraph phGraph;
    if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphChildGraphNodeGetGraph(hNode, &phGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0)
        return -1;

    return result;
}

int handle_cuGraphEventRecordNodeGetEvent(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUevent event_out;
    if (rpc_read(conn, &event_out, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphEventRecordNodeGetEvent(hNode, &event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(CUevent)) < 0)
        return -1;

    return result;
}

int handle_cuGraphEventRecordNodeSetEvent(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUevent event;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphEventRecordNodeSetEvent(hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphEventWaitNodeGetEvent(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUevent event_out;
    if (rpc_read(conn, &event_out, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphEventWaitNodeGetEvent(hNode, &event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(CUevent)) < 0)
        return -1;

    return result;
}

int handle_cuGraphEventWaitNodeSetEvent(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUevent event;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphEventWaitNodeSetEvent(hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphExternalSemaphoresSignalNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS params_out;
    if (rpc_read(conn, &params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExternalSemaphoresSignalNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphExternalSemaphoresWaitNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_EXT_SEM_WAIT_NODE_PARAMS params_out;
    if (rpc_read(conn, &params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExternalSemaphoresWaitNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphBatchMemOpNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_BATCH_MEM_OP_NODE_PARAMS nodeParams_out;
    if (rpc_read(conn, &nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphBatchMemOpNodeGetParams(hNode, &nodeParams_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphMemAllocNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUDA_MEM_ALLOC_NODE_PARAMS params_out;
    if (rpc_read(conn, &params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphMemAllocNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphMemFreeNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUdeviceptr dptr_out;
    if (rpc_read(conn, &dptr_out, sizeof(CUdeviceptr)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr_out, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGraphMemTrim(void *conn)
{
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGraphMemTrim(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetGraphMemAttribute(void *conn)
{
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        return -1;
    CUgraphMem_attribute attr;
    if (rpc_read(conn, &attr, sizeof(CUgraphMem_attribute)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceSetGraphMemAttribute(void *conn)
{
    CUdevice device;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        return -1;
    CUgraphMem_attribute attr;
    if (rpc_read(conn, &attr, sizeof(CUgraphMem_attribute)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceSetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuGraphClone(void *conn)
{
    CUgraph phGraphClone;
    if (rpc_read(conn, &phGraphClone, sizeof(CUgraph)) < 0)
        return -1;
    CUgraph originalGraph;
    if (rpc_read(conn, &originalGraph, sizeof(CUgraph)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphClone(&phGraphClone, originalGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphClone, sizeof(CUgraph)) < 0)
        return -1;

    return result;
}

int handle_cuGraphNodeFindInClone(void *conn)
{
    CUgraphNode phNode;
    if (rpc_read(conn, &phNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraphNode hOriginalNode;
    if (rpc_read(conn, &hOriginalNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraph hClonedGraph;
    if (rpc_read(conn, &hClonedGraph, sizeof(CUgraph)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphNodeFindInClone(&phNode, hOriginalNode, hClonedGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phNode, sizeof(CUgraphNode)) < 0)
        return -1;

    return result;
}

int handle_cuGraphNodeGetType(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraphNodeType type;
    if (rpc_read(conn, &type, sizeof(CUgraphNodeType)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphNodeGetType(hNode, &type);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &type, sizeof(CUgraphNodeType)) < 0)
        return -1;

    return result;
}

int handle_cuGraphGetNodes(void *conn)
{
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;
    CUgraphNode nodes;
    if (rpc_read(conn, &nodes, sizeof(CUgraphNode)) < 0)
        return -1;
    std::size_t numNodes;
    if (rpc_read(conn, &numNodes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphGetNodes(hGraph, &nodes, &numNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodes, sizeof(CUgraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &numNodes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphGetRootNodes(void *conn)
{
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;
    CUgraphNode rootNodes;
    if (rpc_read(conn, &rootNodes, sizeof(CUgraphNode)) < 0)
        return -1;
    std::size_t numRootNodes;
    if (rpc_read(conn, &numRootNodes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphGetRootNodes(hGraph, &rootNodes, &numRootNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &rootNodes, sizeof(CUgraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &numRootNodes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphGetEdges(void *conn)
{
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;
    CUgraphNode from;
    if (rpc_read(conn, &from, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraphNode to;
    if (rpc_read(conn, &to, sizeof(CUgraphNode)) < 0)
        return -1;
    std::size_t numEdges;
    if (rpc_read(conn, &numEdges, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphGetEdges(hGraph, &from, &to, &numEdges);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(CUgraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(CUgraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &numEdges, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphNodeGetDependencies(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraphNode dependencies;
    if (rpc_read(conn, &dependencies, sizeof(CUgraphNode)) < 0)
        return -1;
    std::size_t numDependencies;
    if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphNodeGetDependencies(hNode, &dependencies, &numDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependencies, sizeof(CUgraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &numDependencies, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphNodeGetDependentNodes(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraphNode dependentNodes;
    if (rpc_read(conn, &dependentNodes, sizeof(CUgraphNode)) < 0)
        return -1;
    std::size_t numDependentNodes;
    if (rpc_read(conn, &numDependentNodes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphNodeGetDependentNodes(hNode, &dependentNodes, &numDependentNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dependentNodes, sizeof(CUgraphNode)) < 0)
        return -1;
    if (rpc_write(conn, &numDependentNodes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphDestroyNode(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphDestroyNode(hNode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphInstantiateWithFlags(void *conn)
{
    CUgraphExec phGraphExec;
    if (rpc_read(conn, &phGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphInstantiateWithFlags(&phGraphExec, hGraph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;

    return result;
}

int handle_cuGraphInstantiateWithParams(void *conn)
{
    CUgraphExec phGraphExec;
    if (rpc_read(conn, &phGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;
    CUDA_GRAPH_INSTANTIATE_PARAMS instantiateParams;
    if (rpc_read(conn, &instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphInstantiateWithParams(&phGraphExec, hGraph, &instantiateParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    if (rpc_write(conn, &instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0)
        return -1;

    return result;
}

int handle_cuGraphExecGetFlags(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    cuuint64_t flags;
    if (rpc_read(conn, &flags, sizeof(cuuint64_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExecGetFlags(hGraphExec, &flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(cuuint64_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphExecChildGraphNodeSetParams(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraph childGraph;
    if (rpc_read(conn, &childGraph, sizeof(CUgraph)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphExecEventRecordNodeSetEvent(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUevent event;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphExecEventWaitNodeSetEvent(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUevent event;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphNodeSetEnabled(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    unsigned int isEnabled;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphNodeGetEnabled(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    unsigned int isEnabled;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphNodeGetEnabled(hGraphExec, hNode, &isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cuGraphUpload(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphUpload(hGraphExec, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphLaunch(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphLaunch(hGraphExec, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphExecDestroy(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExecDestroy(hGraphExec);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphDestroy(void *conn)
{
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphDestroy(hGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphExecUpdate_v2(void *conn)
{
    CUgraphExec hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        return -1;
    CUgraph hGraph;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        return -1;
    CUgraphExecUpdateResultInfo resultInfo;
    if (rpc_read(conn, &resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphExecUpdate_v2(hGraphExec, hGraph, &resultInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0)
        return -1;

    return result;
}

int handle_cuGraphKernelNodeCopyAttributes(void *conn)
{
    CUgraphNode dst;
    if (rpc_read(conn, &dst, sizeof(CUgraphNode)) < 0)
        return -1;
    CUgraphNode src;
    if (rpc_read(conn, &src, sizeof(CUgraphNode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphKernelNodeCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphKernelNodeGetAttribute(void *conn)
{
    CUgraphNode hNode;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        return -1;
    CUkernelNodeAttrID attr;
    if (rpc_read(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0)
        return -1;
    CUkernelNodeAttrValue value_out;
    if (rpc_read(conn, &value_out, sizeof(CUkernelNodeAttrValue)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphKernelNodeGetAttribute(hNode, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(CUkernelNodeAttrValue)) < 0)
        return -1;

    return result;
}

int handle_cuUserObjectCreate(void *conn)
{
    CUuserObject object_out;
    if (rpc_read(conn, &object_out, sizeof(CUuserObject)) < 0)
        return -1;
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    CUhostFn destroy;
    if (rpc_read(conn, &destroy, sizeof(CUhostFn)) < 0)
        return -1;
    unsigned int initialRefcount;
    if (rpc_read(conn, &initialRefcount, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuUserObjectCreate(&object_out, &ptr, destroy, initialRefcount, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &object_out, sizeof(CUuserObject)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuUserObjectRetain(void *conn)
{
    CUuserObject object;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuUserObjectRetain(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuUserObjectRelease(void *conn)
{
    CUuserObject object;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuUserObjectRelease(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphRetainUserObject(void *conn)
{
    CUgraph graph;
    if (rpc_read(conn, &graph, sizeof(CUgraph)) < 0)
        return -1;
    CUuserObject object;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphRetainUserObject(graph, object, count, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphReleaseUserObject(void *conn)
{
    CUgraph graph;
    if (rpc_read(conn, &graph, sizeof(CUgraph)) < 0)
        return -1;
    CUuserObject object;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphReleaseUserObject(graph, object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(void *conn)
{
    int numBlocks;
    if (rpc_read(conn, &numBlocks, sizeof(int)) < 0)
        return -1;
    CUfunction func;
    if (rpc_read(conn, &func, sizeof(CUfunction)) < 0)
        return -1;
    int blockSize;
    if (rpc_read(conn, &blockSize, sizeof(int)) < 0)
        return -1;
    std::size_t dynamicSMemSize;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func, blockSize, dynamicSMemSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numBlocks, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *conn)
{
    int numBlocks;
    if (rpc_read(conn, &numBlocks, sizeof(int)) < 0)
        return -1;
    CUfunction func;
    if (rpc_read(conn, &func, sizeof(CUfunction)) < 0)
        return -1;
    int blockSize;
    if (rpc_read(conn, &blockSize, sizeof(int)) < 0)
        return -1;
    std::size_t dynamicSMemSize;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, func, blockSize, dynamicSMemSize, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &numBlocks, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuOccupancyAvailableDynamicSMemPerBlock(void *conn)
{
    std::size_t dynamicSmemSize;
    if (rpc_read(conn, &dynamicSmemSize, sizeof(size_t)) < 0)
        return -1;
    CUfunction func;
    if (rpc_read(conn, &func, sizeof(CUfunction)) < 0)
        return -1;
    int numBlocks;
    if (rpc_read(conn, &numBlocks, sizeof(int)) < 0)
        return -1;
    int blockSize;
    if (rpc_read(conn, &blockSize, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, func, numBlocks, blockSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dynamicSmemSize, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetArray(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetArray(hTexRef, hArray, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetMipmappedArray(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    CUmipmappedArray hMipmappedArray;
    if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetAddress_v2(void *conn)
{
    std::size_t ByteOffset;
    if (rpc_read(conn, &ByteOffset, sizeof(size_t)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t bytes;
    if (rpc_read(conn, &bytes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetAddress_v2(&ByteOffset, hTexRef, dptr, bytes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ByteOffset, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetFormat(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    CUarray_format fmt;
    if (rpc_read(conn, &fmt, sizeof(CUarray_format)) < 0)
        return -1;
    int NumPackedComponents;
    if (rpc_read(conn, &NumPackedComponents, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetAddressMode(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    int dim;
    if (rpc_read(conn, &dim, sizeof(int)) < 0)
        return -1;
    CUaddress_mode am;
    if (rpc_read(conn, &am, sizeof(CUaddress_mode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetAddressMode(hTexRef, dim, am);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetFilterMode(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    CUfilter_mode fm;
    if (rpc_read(conn, &fm, sizeof(CUfilter_mode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetFilterMode(hTexRef, fm);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetMipmapFilterMode(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    CUfilter_mode fm;
    if (rpc_read(conn, &fm, sizeof(CUfilter_mode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetMipmapFilterMode(hTexRef, fm);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetMipmapLevelBias(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    float bias;
    if (rpc_read(conn, &bias, sizeof(float)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetMipmapLevelBias(hTexRef, bias);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetMipmapLevelClamp(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    float minMipmapLevelClamp;
    if (rpc_read(conn, &minMipmapLevelClamp, sizeof(float)) < 0)
        return -1;
    float maxMipmapLevelClamp;
    if (rpc_read(conn, &maxMipmapLevelClamp, sizeof(float)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetMaxAnisotropy(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    unsigned int maxAniso;
    if (rpc_read(conn, &maxAniso, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetMaxAnisotropy(hTexRef, maxAniso);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetBorderColor(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    float pBorderColor;
    if (rpc_read(conn, &pBorderColor, sizeof(float)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetBorderColor(hTexRef, &pBorderColor);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pBorderColor, sizeof(float)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefSetFlags(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefSetFlags(hTexRef, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetAddress_v2(void *conn)
{
    CUdeviceptr pdptr;
    if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetAddress_v2(&pdptr, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetArray(void *conn)
{
    CUarray phArray;
    if (rpc_read(conn, &phArray, sizeof(CUarray)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetArray(&phArray, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phArray, sizeof(CUarray)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetMipmappedArray(void *conn)
{
    CUmipmappedArray phMipmappedArray;
    if (rpc_read(conn, &phMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetMipmappedArray(&phMipmappedArray, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetAddressMode(void *conn)
{
    CUaddress_mode pam;
    if (rpc_read(conn, &pam, sizeof(CUaddress_mode)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;
    int dim;
    if (rpc_read(conn, &dim, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetAddressMode(&pam, hTexRef, dim);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pam, sizeof(CUaddress_mode)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetFilterMode(void *conn)
{
    CUfilter_mode pfm;
    if (rpc_read(conn, &pfm, sizeof(CUfilter_mode)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetFilterMode(&pfm, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pfm, sizeof(CUfilter_mode)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetFormat(void *conn)
{
    CUarray_format pFormat;
    if (rpc_read(conn, &pFormat, sizeof(CUarray_format)) < 0)
        return -1;
    int pNumChannels;
    if (rpc_read(conn, &pNumChannels, sizeof(int)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetFormat(&pFormat, &pNumChannels, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFormat, sizeof(CUarray_format)) < 0)
        return -1;
    if (rpc_write(conn, &pNumChannels, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetMipmapFilterMode(void *conn)
{
    CUfilter_mode pfm;
    if (rpc_read(conn, &pfm, sizeof(CUfilter_mode)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetMipmapFilterMode(&pfm, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pfm, sizeof(CUfilter_mode)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetMipmapLevelBias(void *conn)
{
    float pbias;
    if (rpc_read(conn, &pbias, sizeof(float)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetMipmapLevelBias(&pbias, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pbias, sizeof(float)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetMipmapLevelClamp(void *conn)
{
    float pminMipmapLevelClamp;
    if (rpc_read(conn, &pminMipmapLevelClamp, sizeof(float)) < 0)
        return -1;
    float pmaxMipmapLevelClamp;
    if (rpc_read(conn, &pmaxMipmapLevelClamp, sizeof(float)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetMipmapLevelClamp(&pminMipmapLevelClamp, &pmaxMipmapLevelClamp, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pminMipmapLevelClamp, sizeof(float)) < 0)
        return -1;
    if (rpc_write(conn, &pmaxMipmapLevelClamp, sizeof(float)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetMaxAnisotropy(void *conn)
{
    int pmaxAniso;
    if (rpc_read(conn, &pmaxAniso, sizeof(int)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetMaxAnisotropy(&pmaxAniso, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pmaxAniso, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetBorderColor(void *conn)
{
    float pBorderColor;
    if (rpc_read(conn, &pBorderColor, sizeof(float)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetBorderColor(&pBorderColor, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pBorderColor, sizeof(float)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefGetFlags(void *conn)
{
    unsigned int pFlags;
    if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0)
        return -1;
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefGetFlags(&pFlags, hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefCreate(void *conn)
{
    CUtexref pTexRef;
    if (rpc_read(conn, &pTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefCreate(&pTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexRef, sizeof(CUtexref)) < 0)
        return -1;

    return result;
}

int handle_cuTexRefDestroy(void *conn)
{
    CUtexref hTexRef;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexRefDestroy(hTexRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuSurfRefSetArray(void *conn)
{
    CUsurfref hSurfRef;
    if (rpc_read(conn, &hSurfRef, sizeof(CUsurfref)) < 0)
        return -1;
    CUarray hArray;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuSurfRefSetArray(hSurfRef, hArray, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuSurfRefGetArray(void *conn)
{
    CUarray phArray;
    if (rpc_read(conn, &phArray, sizeof(CUarray)) < 0)
        return -1;
    CUsurfref hSurfRef;
    if (rpc_read(conn, &hSurfRef, sizeof(CUsurfref)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuSurfRefGetArray(&phArray, hSurfRef);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &phArray, sizeof(CUarray)) < 0)
        return -1;

    return result;
}

int handle_cuTexObjectDestroy(void *conn)
{
    CUtexObject texObject;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexObjectDestroy(texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuTexObjectGetResourceDesc(void *conn)
{
    CUDA_RESOURCE_DESC pResDesc;
    if (rpc_read(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0)
        return -1;
    CUtexObject texObject;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexObjectGetResourceDesc(&pResDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0)
        return -1;

    return result;
}

int handle_cuTexObjectGetTextureDesc(void *conn)
{
    CUDA_TEXTURE_DESC pTexDesc;
    if (rpc_read(conn, &pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0)
        return -1;
    CUtexObject texObject;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexObjectGetTextureDesc(&pTexDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0)
        return -1;

    return result;
}

int handle_cuTexObjectGetResourceViewDesc(void *conn)
{
    CUDA_RESOURCE_VIEW_DESC pResViewDesc;
    if (rpc_read(conn, &pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0)
        return -1;
    CUtexObject texObject;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTexObjectGetResourceViewDesc(&pResViewDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0)
        return -1;

    return result;
}

int handle_cuSurfObjectDestroy(void *conn)
{
    CUsurfObject surfObject;
    if (rpc_read(conn, &surfObject, sizeof(CUsurfObject)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuSurfObjectDestroy(surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuSurfObjectGetResourceDesc(void *conn)
{
    CUDA_RESOURCE_DESC pResDesc;
    if (rpc_read(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0)
        return -1;
    CUsurfObject surfObject;
    if (rpc_read(conn, &surfObject, sizeof(CUsurfObject)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuSurfObjectGetResourceDesc(&pResDesc, surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0)
        return -1;

    return result;
}

int handle_cuTensorMapReplaceAddress(void *conn)
{
    CUtensorMap tensorMap;
    if (rpc_read(conn, &tensorMap, sizeof(CUtensorMap)) < 0)
        return -1;
    void* globalAddress;
    if (rpc_read(conn, &globalAddress, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuTensorMapReplaceAddress(&tensorMap, &globalAddress);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &tensorMap, sizeof(CUtensorMap)) < 0)
        return -1;
    if (rpc_write(conn, &globalAddress, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cuDeviceCanAccessPeer(void *conn)
{
    int canAccessPeer;
    if (rpc_read(conn, &canAccessPeer, sizeof(int)) < 0)
        return -1;
    CUdevice dev;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        return -1;
    CUdevice peerDev;
    if (rpc_read(conn, &peerDev, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceCanAccessPeer(&canAccessPeer, dev, peerDev);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &canAccessPeer, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuCtxEnablePeerAccess(void *conn)
{
    CUcontext peerContext;
    if (rpc_read(conn, &peerContext, sizeof(CUcontext)) < 0)
        return -1;
    unsigned int Flags;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxEnablePeerAccess(peerContext, Flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuCtxDisablePeerAccess(void *conn)
{
    CUcontext peerContext;
    if (rpc_read(conn, &peerContext, sizeof(CUcontext)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuCtxDisablePeerAccess(peerContext);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuDeviceGetP2PAttribute(void *conn)
{
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    CUdevice_P2PAttribute attrib;
    if (rpc_read(conn, &attrib, sizeof(CUdevice_P2PAttribute)) < 0)
        return -1;
    CUdevice srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdevice)) < 0)
        return -1;
    CUdevice dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(CUdevice)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuDeviceGetP2PAttribute(&value, attrib, srcDevice, dstDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsUnregisterResource(void *conn)
{
    CUgraphicsResource resource;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsUnregisterResource(resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsSubResourceGetMappedArray(void *conn)
{
    CUarray pArray;
    if (rpc_read(conn, &pArray, sizeof(CUarray)) < 0)
        return -1;
    CUgraphicsResource resource;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        return -1;
    unsigned int arrayIndex;
    if (rpc_read(conn, &arrayIndex, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int mipLevel;
    if (rpc_read(conn, &mipLevel, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsSubResourceGetMappedArray(&pArray, resource, arrayIndex, mipLevel);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pArray, sizeof(CUarray)) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsResourceGetMappedMipmappedArray(void *conn)
{
    CUmipmappedArray pMipmappedArray;
    if (rpc_read(conn, &pMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;
    CUgraphicsResource resource;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsResourceGetMappedMipmappedArray(&pMipmappedArray, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsResourceGetMappedPointer_v2(void *conn)
{
    CUdeviceptr pDevPtr;
    if (rpc_read(conn, &pDevPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    std::size_t pSize;
    if (rpc_read(conn, &pSize, sizeof(size_t)) < 0)
        return -1;
    CUgraphicsResource resource;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsResourceGetMappedPointer_v2(&pDevPtr, &pSize, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDevPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    if (rpc_write(conn, &pSize, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsResourceSetMapFlags_v2(void *conn)
{
    CUgraphicsResource resource;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsResourceSetMapFlags_v2(resource, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsMapResources(void *conn)
{
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    CUgraphicsResource resources;
    if (rpc_read(conn, &resources, sizeof(CUgraphicsResource)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsMapResources(count, &resources, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(CUgraphicsResource)) < 0)
        return -1;

    return result;
}

int handle_cuGraphicsUnmapResources(void *conn)
{
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    CUgraphicsResource resources;
    if (rpc_read(conn, &resources, sizeof(CUgraphicsResource)) < 0)
        return -1;
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    CUresult result = cuGraphicsUnmapResources(count, &resources, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(CUgraphicsResource)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceReset(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceReset();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceSynchronize(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceSynchronize();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceSetLimit(void *conn)
{
    enum cudaLimit limit;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        return -1;
    std::size_t value;
    if (rpc_read(conn, &value, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetLimit(void *conn)
{
    std::size_t pValue;
    if (rpc_read(conn, &pValue, sizeof(size_t)) < 0)
        return -1;
    enum cudaLimit limit;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetLimit(&pValue, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pValue, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetCacheConfig(void *conn)
{
    enum cudaFuncCache pCacheConfig;
    if (rpc_read(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetCacheConfig(&pCacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetStreamPriorityRange(void *conn)
{
    int leastPriority;
    if (rpc_read(conn, &leastPriority, sizeof(int)) < 0)
        return -1;
    int greatestPriority;
    if (rpc_read(conn, &greatestPriority, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &leastPriority, sizeof(int)) < 0)
        return -1;
    if (rpc_write(conn, &greatestPriority, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceSetCacheConfig(void *conn)
{
    enum cudaFuncCache cacheConfig;
    if (rpc_read(conn, &cacheConfig, sizeof(enum cudaFuncCache)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceSetCacheConfig(cacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetSharedMemConfig(void *conn)
{
    enum cudaSharedMemConfig pConfig;
    if (rpc_read(conn, &pConfig, sizeof(enum cudaSharedMemConfig)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetSharedMemConfig(&pConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pConfig, sizeof(enum cudaSharedMemConfig)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceSetSharedMemConfig(void *conn)
{
    enum cudaSharedMemConfig config;
    if (rpc_read(conn, &config, sizeof(enum cudaSharedMemConfig)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceSetSharedMemConfig(config);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetPCIBusId(void *conn)
{
    char pciBusId;
    if (rpc_read(conn, &pciBusId, sizeof(char)) < 0)
        return -1;
    int len;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetPCIBusId(&pciBusId, len, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pciBusId, sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_cudaIpcGetEventHandle(void *conn)
{
    cudaIpcEventHandle_t handle;
    if (rpc_read(conn, &handle, sizeof(cudaIpcEventHandle_t)) < 0)
        return -1;
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaIpcGetEventHandle(&handle, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(cudaIpcEventHandle_t)) < 0)
        return -1;

    return result;
}

int handle_cudaIpcOpenEventHandle(void *conn)
{
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;
    cudaIpcEventHandle_t handle;
    if (rpc_read(conn, &handle, sizeof(cudaIpcEventHandle_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaIpcOpenEventHandle(&event, handle);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    return result;
}

int handle_cudaIpcGetMemHandle(void *conn)
{
    cudaIpcMemHandle_t handle;
    if (rpc_read(conn, &handle, sizeof(cudaIpcMemHandle_t)) < 0)
        return -1;
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaIpcGetMemHandle(&handle, &devPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &handle, sizeof(cudaIpcMemHandle_t)) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaIpcOpenMemHandle(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    cudaIpcMemHandle_t handle;
    if (rpc_read(conn, &handle, sizeof(cudaIpcMemHandle_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaIpcOpenMemHandle(&devPtr, handle, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaIpcCloseMemHandle(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaIpcCloseMemHandle(&devPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceFlushGPUDirectRDMAWrites(void *conn)
{
    enum cudaFlushGPUDirectRDMAWritesTarget target;
    if (rpc_read(conn, &target, sizeof(enum cudaFlushGPUDirectRDMAWritesTarget)) < 0)
        return -1;
    enum cudaFlushGPUDirectRDMAWritesScope scope;
    if (rpc_read(conn, &scope, sizeof(enum cudaFlushGPUDirectRDMAWritesScope)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceFlushGPUDirectRDMAWrites(target, scope);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaThreadExit(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadExit();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaThreadSynchronize(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadSynchronize();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaThreadSetLimit(void *conn)
{
    enum cudaLimit limit;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        return -1;
    std::size_t value;
    if (rpc_read(conn, &value, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaThreadGetLimit(void *conn)
{
    std::size_t pValue;
    if (rpc_read(conn, &pValue, sizeof(size_t)) < 0)
        return -1;
    enum cudaLimit limit;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadGetLimit(&pValue, limit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pValue, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaThreadGetCacheConfig(void *conn)
{
    enum cudaFuncCache pCacheConfig;
    if (rpc_read(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadGetCacheConfig(&pCacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0)
        return -1;

    return result;
}

int handle_cudaThreadSetCacheConfig(void *conn)
{
    enum cudaFuncCache cacheConfig;
    if (rpc_read(conn, &cacheConfig, sizeof(enum cudaFuncCache)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadSetCacheConfig(cacheConfig);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGetLastError(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetLastError();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaPeekAtLastError(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaPeekAtLastError();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGetDeviceCount(void *conn)
{
    int count;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetDeviceCount(&count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &count, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaGetDeviceProperties_v2(void *conn)
{
    struct cudaDeviceProp prop;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetDeviceProperties_v2(&prop, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &prop, sizeof(struct cudaDeviceProp)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetAttribute(void *conn)
{
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    enum cudaDeviceAttr attr;
    if (rpc_read(conn, &attr, sizeof(enum cudaDeviceAttr)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetAttribute(&value, attr, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetDefaultMemPool(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetDefaultMemPool(&memPool, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceSetMemPool(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceSetMemPool(device, memPool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetMemPool(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetMemPool(&memPool, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetNvSciSyncAttributes(void *conn)
{
    void* nvSciSyncAttrList;
    if (rpc_read(conn, &nvSciSyncAttrList, sizeof(void*)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;
    int flags;
    if (rpc_read(conn, &flags, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetNvSciSyncAttributes(&nvSciSyncAttrList, device, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nvSciSyncAttrList, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetP2PAttribute(void *conn)
{
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    enum cudaDeviceP2PAttr attr;
    if (rpc_read(conn, &attr, sizeof(enum cudaDeviceP2PAttr)) < 0)
        return -1;
    int srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(int)) < 0)
        return -1;
    int dstDevice;
    if (rpc_read(conn, &dstDevice, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetP2PAttribute(&value, attr, srcDevice, dstDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaInitDevice(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;
    unsigned int deviceFlags;
    if (rpc_read(conn, &deviceFlags, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaInitDevice(device, deviceFlags, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaSetDevice(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaSetDevice(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGetDevice(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetDevice(&device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaSetValidDevices(void *conn)
{
    int device_arr;
    if (rpc_read(conn, &device_arr, sizeof(int)) < 0)
        return -1;
    int len;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaSetValidDevices(&device_arr, len);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &device_arr, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaSetDeviceFlags(void *conn)
{
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaSetDeviceFlags(flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGetDeviceFlags(void *conn)
{
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetDeviceFlags(&flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamCreate(void *conn)
{
    cudaStream_t pStream;
    if (rpc_read(conn, &pStream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamCreate(&pStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pStream, sizeof(cudaStream_t)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamCreateWithFlags(void *conn)
{
    cudaStream_t pStream;
    if (rpc_read(conn, &pStream, sizeof(cudaStream_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamCreateWithFlags(&pStream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pStream, sizeof(cudaStream_t)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamCreateWithPriority(void *conn)
{
    cudaStream_t pStream;
    if (rpc_read(conn, &pStream, sizeof(cudaStream_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;
    int priority;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamCreateWithPriority(&pStream, flags, priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pStream, sizeof(cudaStream_t)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamGetPriority(void *conn)
{
    cudaStream_t hStream;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        return -1;
    int priority;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamGetPriority(hStream, &priority);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &priority, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamGetFlags(void *conn)
{
    cudaStream_t hStream;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamGetFlags(hStream, &flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamGetId(void *conn)
{
    cudaStream_t hStream;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        return -1;
    unsigned long long streamId;
    if (rpc_read(conn, &streamId, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamGetId(hStream, &streamId);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &streamId, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_cudaCtxResetPersistingL2Cache(void *conn)
{

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaCtxResetPersistingL2Cache();

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaStreamCopyAttributes(void *conn)
{
    cudaStream_t dst;
    if (rpc_read(conn, &dst, sizeof(cudaStream_t)) < 0)
        return -1;
    cudaStream_t src;
    if (rpc_read(conn, &src, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaStreamGetAttribute(void *conn)
{
    cudaStream_t hStream;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        return -1;
    cudaLaunchAttributeID attr;
    if (rpc_read(conn, &attr, sizeof(cudaLaunchAttributeID)) < 0)
        return -1;
    cudaLaunchAttributeValue value_out;
    if (rpc_read(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamGetAttribute(hStream, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamDestroy(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamDestroy(stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaStreamWaitEvent(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamWaitEvent(stream, event, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaStreamAddCallback(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    cudaStreamCallback_t callback;
    if (rpc_read(conn, &callback, sizeof(cudaStreamCallback_t)) < 0)
        return -1;
    void* userData;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamAddCallback(stream, callback, &userData, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamSynchronize(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamSynchronize(stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaStreamQuery(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamQuery(stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaStreamAttachMemAsync(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t length;
    if (rpc_read(conn, &length, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamAttachMemAsync(stream, &devPtr, length, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamBeginCapture(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    enum cudaStreamCaptureMode mode;
    if (rpc_read(conn, &mode, sizeof(enum cudaStreamCaptureMode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamBeginCapture(stream, mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaThreadExchangeStreamCaptureMode(void *conn)
{
    enum cudaStreamCaptureMode mode;
    if (rpc_read(conn, &mode, sizeof(enum cudaStreamCaptureMode)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaThreadExchangeStreamCaptureMode(&mode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mode, sizeof(enum cudaStreamCaptureMode)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamEndCapture(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    cudaGraph_t pGraph;
    if (rpc_read(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamEndCapture(stream, &pGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamIsCapturing(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    enum cudaStreamCaptureStatus pCaptureStatus;
    if (rpc_read(conn, &pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamIsCapturing(stream, &pCaptureStatus);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamGetCaptureInfo_v2(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    enum cudaStreamCaptureStatus captureStatus_out;
    unsigned long long id_out;
    cudaGraph_t graph_out;
    std::size_t numDependencies_out;
    const cudaGraphNode_t** dependencies_out = (const cudaGraphNode_t**)malloc(numDependencies_out * sizeof(const cudaGraphNode_t*));

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamGetCaptureInfo_v2(stream, &captureStatus_out, &id_out, &graph_out, dependencies_out, &numDependencies_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &captureStatus_out, sizeof(enum cudaStreamCaptureStatus)) < 0)
        return -1;
    if (rpc_write(conn, &id_out, sizeof(unsigned long long)) < 0)
        return -1;
    if (rpc_write(conn, &graph_out, sizeof(cudaGraph_t)) < 0)
        return -1;
    if (rpc_write(conn, &numDependencies_out, sizeof(size_t)) < 0)
        return -1;
    if (rpc_write(conn, dependencies_out, numDependencies_out * sizeof(const cudaGraphNode_t*)) < 0)
        return -1;

    return result;
}

int handle_cudaStreamUpdateCaptureDependencies(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    std::size_t numDependencies;
    if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0)
        return -1;
    cudaGraphNode_t* dependencies = (cudaGraphNode_t*)malloc(numDependencies * sizeof(cudaGraphNode_t));
    if (rpc_read(conn, &dependencies, sizeof(cudaGraphNode_t*)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaEventCreate(void *conn)
{
    cudaEvent_t event;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventCreate(&event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    return result;
}

int handle_cudaEventCreateWithFlags(void *conn)
{
    cudaEvent_t event;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventCreateWithFlags(&event, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    return result;
}

int handle_cudaEventRecord(void *conn)
{
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventRecord(event, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaEventRecordWithFlags(void *conn)
{
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventRecordWithFlags(event, stream, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaEventQuery(void *conn)
{
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventQuery(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaEventSynchronize(void *conn)
{
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventSynchronize(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaEventDestroy(void *conn)
{
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventDestroy(event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaEventElapsedTime(void *conn)
{
    float ms;
    cudaEvent_t start;
    if (rpc_read(conn, &start, sizeof(cudaEvent_t)) < 0)
        return -1;
    cudaEvent_t end;
    if (rpc_read(conn, &end, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaEventElapsedTime(&ms, start, end);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ms, sizeof(float)) < 0)
        return -1;

    return result;
}

int handle_cudaDestroyExternalMemory(void *conn)
{
    cudaExternalMemory_t extMem;
    if (rpc_read(conn, &extMem, sizeof(cudaExternalMemory_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDestroyExternalMemory(extMem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDestroyExternalSemaphore(void *conn)
{
    cudaExternalSemaphore_t extSem;
    if (rpc_read(conn, &extSem, sizeof(cudaExternalSemaphore_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDestroyExternalSemaphore(extSem);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaLaunchCooperativeKernelMultiDevice(void *conn)
{
    struct cudaLaunchParams launchParamsList;
    if (rpc_read(conn, &launchParamsList, sizeof(struct cudaLaunchParams)) < 0)
        return -1;
    unsigned int numDevices;
    if (rpc_read(conn, &numDevices, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaLaunchCooperativeKernelMultiDevice(&launchParamsList, numDevices, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &launchParamsList, sizeof(struct cudaLaunchParams)) < 0)
        return -1;

    return result;
}

int handle_cudaSetDoubleForDevice(void *conn)
{
    double d;
    if (rpc_read(conn, &d, sizeof(double)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaSetDoubleForDevice(&d);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &d, sizeof(double)) < 0)
        return -1;

    return result;
}

int handle_cudaSetDoubleForHost(void *conn)
{
    double d;
    if (rpc_read(conn, &d, sizeof(double)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaSetDoubleForHost(&d);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &d, sizeof(double)) < 0)
        return -1;

    return result;
}

int handle_cudaLaunchHostFunc(void *conn)
{
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;
    cudaHostFn_t fn;
    if (rpc_read(conn, &fn, sizeof(cudaHostFn_t)) < 0)
        return -1;
    void* userData;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaLaunchHostFunc(stream, fn, &userData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &userData, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMallocManaged(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMallocManaged(&devPtr, size, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMalloc(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMalloc(&devPtr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMallocHost(void *conn)
{
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMallocHost(&ptr, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMallocPitch(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t pitch;
    if (rpc_read(conn, &pitch, sizeof(size_t)) < 0)
        return -1;
    std::size_t width;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        return -1;
    std::size_t height;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMallocPitch(&devPtr, &pitch, width, height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &pitch, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaFree(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaFree(&devPtr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaFreeHost(void *conn)
{
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaFreeHost(&ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaFreeArray(void *conn)
{
    cudaArray_t array;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaFreeArray(array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaFreeMipmappedArray(void *conn)
{
    cudaMipmappedArray_t mipmappedArray;
    if (rpc_read(conn, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaFreeMipmappedArray(mipmappedArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaHostAlloc(void *conn)
{
    void* pHost;
    if (rpc_read(conn, &pHost, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaHostAlloc(&pHost, size, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pHost, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaHostRegister(void *conn)
{
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaHostRegister(&ptr, size, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaHostUnregister(void *conn)
{
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaHostUnregister(&ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaHostGetDevicePointer(void *conn)
{
    void* pDevice;
    if (rpc_read(conn, &pDevice, sizeof(void*)) < 0)
        return -1;
    void* pHost;
    if (rpc_read(conn, &pHost, sizeof(void*)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaHostGetDevicePointer(&pDevice, &pHost, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDevice, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &pHost, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaHostGetFlags(void *conn)
{
    unsigned int pFlags;
    if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0)
        return -1;
    void* pHost;
    if (rpc_read(conn, &pHost, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaHostGetFlags(&pFlags, &pHost);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0)
        return -1;
    if (rpc_write(conn, &pHost, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMalloc3D(void *conn)
{
    struct cudaPitchedPtr pitchedDevPtr;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        return -1;
    struct cudaExtent extent;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMalloc3D(&pitchedDevPtr, extent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        return -1;

    return result;
}

int handle_cudaGetMipmappedArrayLevel(void *conn)
{
    cudaArray_t levelArray;
    if (rpc_read(conn, &levelArray, sizeof(cudaArray_t)) < 0)
        return -1;
    cudaMipmappedArray_const_t mipmappedArray;
    if (rpc_read(conn, &mipmappedArray, sizeof(cudaMipmappedArray_const_t)) < 0)
        return -1;
    unsigned int level;
    if (rpc_read(conn, &level, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, level);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &levelArray, sizeof(cudaArray_t)) < 0)
        return -1;

    return result;
}

int handle_cudaMemGetInfo(void *conn)
{
    std::size_t free;
    if (rpc_read(conn, &free, sizeof(size_t)) < 0)
        return -1;
    std::size_t total;
    if (rpc_read(conn, &total, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemGetInfo(&free, &total);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &free, sizeof(size_t)) < 0)
        return -1;
    if (rpc_write(conn, &total, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaArrayGetInfo(void *conn)
{
    struct cudaChannelFormatDesc desc;
    if (rpc_read(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0)
        return -1;
    struct cudaExtent extent;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;
    cudaArray_t array;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaArrayGetInfo(&desc, &extent, &flags, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0)
        return -1;
    if (rpc_write(conn, &extent, sizeof(struct cudaExtent)) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cudaArrayGetPlane(void *conn)
{
    cudaArray_t pPlaneArray;
    if (rpc_read(conn, &pPlaneArray, sizeof(cudaArray_t)) < 0)
        return -1;
    cudaArray_t hArray;
    if (rpc_read(conn, &hArray, sizeof(cudaArray_t)) < 0)
        return -1;
    unsigned int planeIdx;
    if (rpc_read(conn, &planeIdx, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaArrayGetPlane(&pPlaneArray, hArray, planeIdx);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pPlaneArray, sizeof(cudaArray_t)) < 0)
        return -1;

    return result;
}

int handle_cudaArrayGetMemoryRequirements(void *conn)
{
    struct cudaArrayMemoryRequirements memoryRequirements;
    if (rpc_read(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0)
        return -1;
    cudaArray_t array;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaArrayGetMemoryRequirements(&memoryRequirements, array, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0)
        return -1;

    return result;
}

int handle_cudaMipmappedArrayGetMemoryRequirements(void *conn)
{
    struct cudaArrayMemoryRequirements memoryRequirements;
    if (rpc_read(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0)
        return -1;
    cudaMipmappedArray_t mipmap;
    if (rpc_read(conn, &mipmap, sizeof(cudaMipmappedArray_t)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMipmappedArrayGetMemoryRequirements(&memoryRequirements, mipmap, device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0)
        return -1;

    return result;
}

int handle_cudaArrayGetSparseProperties(void *conn)
{
    struct cudaArraySparseProperties sparseProperties;
    if (rpc_read(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0)
        return -1;
    cudaArray_t array;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaArrayGetSparseProperties(&sparseProperties, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0)
        return -1;

    return result;
}

int handle_cudaMipmappedArrayGetSparseProperties(void *conn)
{
    struct cudaArraySparseProperties sparseProperties;
    if (rpc_read(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0)
        return -1;
    cudaMipmappedArray_t mipmap;
    if (rpc_read(conn, &mipmap, sizeof(cudaMipmappedArray_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMipmappedArrayGetSparseProperties(&sparseProperties, mipmap);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpy2DFromArray(void *conn)
{
    void* dst;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        return -1;
    std::size_t dpitch;
    if (rpc_read(conn, &dpitch, sizeof(size_t)) < 0)
        return -1;
    cudaArray_const_t src;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        return -1;
    std::size_t wOffset;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffset;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t width;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        return -1;
    std::size_t height;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpy2DFromArray(&dst, dpitch, src, wOffset, hOffset, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpy2DArrayToArray(void *conn)
{
    cudaArray_t dst;
    if (rpc_read(conn, &dst, sizeof(cudaArray_t)) < 0)
        return -1;
    std::size_t wOffsetDst;
    if (rpc_read(conn, &wOffsetDst, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffsetDst;
    if (rpc_read(conn, &hOffsetDst, sizeof(size_t)) < 0)
        return -1;
    cudaArray_const_t src;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        return -1;
    std::size_t wOffsetSrc;
    if (rpc_read(conn, &wOffsetSrc, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffsetSrc;
    if (rpc_read(conn, &hOffsetSrc, sizeof(size_t)) < 0)
        return -1;
    std::size_t width;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        return -1;
    std::size_t height;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpyAsync(void *conn)
{
    void* dst;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    void* src = (void*)malloc(count);
    if (rpc_read(conn, &src, sizeof(void*)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpyAsync(&dst, &src, count, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpy2DFromArrayAsync(void *conn)
{
    void* dst;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        return -1;
    std::size_t dpitch;
    if (rpc_read(conn, &dpitch, sizeof(size_t)) < 0)
        return -1;
    cudaArray_const_t src;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        return -1;
    std::size_t wOffset;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffset;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t width;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        return -1;
    std::size_t height;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpy2DFromArrayAsync(&dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemset(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemset(&devPtr, value, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemset2D(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t pitch;
    if (rpc_read(conn, &pitch, sizeof(size_t)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    std::size_t width;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        return -1;
    std::size_t height;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemset2D(&devPtr, pitch, value, width, height);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemset3D(void *conn)
{
    struct cudaPitchedPtr pitchedDevPtr;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    struct cudaExtent extent;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemset3D(pitchedDevPtr, value, extent);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMemsetAsync(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemsetAsync(&devPtr, value, count, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemset2DAsync(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t pitch;
    if (rpc_read(conn, &pitch, sizeof(size_t)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    std::size_t width;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        return -1;
    std::size_t height;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemset2DAsync(&devPtr, pitch, value, width, height, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemset3DAsync(void *conn)
{
    struct cudaPitchedPtr pitchedDevPtr;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        return -1;
    int value;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        return -1;
    struct cudaExtent extent;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpyFromArray(void *conn)
{
    void* dst;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        return -1;
    cudaArray_const_t src;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        return -1;
    std::size_t wOffset;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffset;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpyFromArray(&dst, src, wOffset, hOffset, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpyArrayToArray(void *conn)
{
    cudaArray_t dst;
    if (rpc_read(conn, &dst, sizeof(cudaArray_t)) < 0)
        return -1;
    std::size_t wOffsetDst;
    if (rpc_read(conn, &wOffsetDst, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffsetDst;
    if (rpc_read(conn, &hOffsetDst, sizeof(size_t)) < 0)
        return -1;
    cudaArray_const_t src;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        return -1;
    std::size_t wOffsetSrc;
    if (rpc_read(conn, &wOffsetSrc, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffsetSrc;
    if (rpc_read(conn, &hOffsetSrc, sizeof(size_t)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMemcpyFromArrayAsync(void *conn)
{
    void* dst;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        return -1;
    cudaArray_const_t src;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        return -1;
    std::size_t wOffset;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t hOffset;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        return -1;
    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;
    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemcpyFromArrayAsync(&dst, src, wOffset, hOffset, count, kind, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dst, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMallocAsync(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    cudaStream_t hStream;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMallocAsync(&devPtr, size, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaFreeAsync(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    cudaStream_t hStream;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaFreeAsync(&devPtr, hStream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolTrimTo(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    std::size_t minBytesToKeep;
    if (rpc_read(conn, &minBytesToKeep, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolTrimTo(memPool, minBytesToKeep);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolSetAttribute(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    enum cudaMemPoolAttr attr;
    if (rpc_read(conn, &attr, sizeof(enum cudaMemPoolAttr)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolSetAttribute(memPool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolGetAttribute(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    enum cudaMemPoolAttr attr;
    if (rpc_read(conn, &attr, sizeof(enum cudaMemPoolAttr)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolGetAttribute(memPool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolGetAccess(void *conn)
{
    enum cudaMemAccessFlags flags;
    if (rpc_read(conn, &flags, sizeof(enum cudaMemAccessFlags)) < 0)
        return -1;
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    struct cudaMemLocation location;
    if (rpc_read(conn, &location, sizeof(struct cudaMemLocation)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolGetAccess(&flags, memPool, &location);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(enum cudaMemAccessFlags)) < 0)
        return -1;
    if (rpc_write(conn, &location, sizeof(struct cudaMemLocation)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolDestroy(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolDestroy(memPool);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaMallocFromPoolAsync(void *conn)
{
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMallocFromPoolAsync(&ptr, size, memPool, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolExportToShareableHandle(void *conn)
{
    void* shareableHandle;
    if (rpc_read(conn, &shareableHandle, sizeof(void*)) < 0)
        return -1;
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    enum cudaMemAllocationHandleType handleType;
    if (rpc_read(conn, &handleType, sizeof(enum cudaMemAllocationHandleType)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolExportToShareableHandle(&shareableHandle, memPool, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &shareableHandle, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolImportFromShareableHandle(void *conn)
{
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    void* shareableHandle;
    if (rpc_read(conn, &shareableHandle, sizeof(void*)) < 0)
        return -1;
    enum cudaMemAllocationHandleType handleType;
    if (rpc_read(conn, &handleType, sizeof(enum cudaMemAllocationHandleType)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolImportFromShareableHandle(&memPool, &shareableHandle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    if (rpc_write(conn, &shareableHandle, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolExportPointer(void *conn)
{
    struct cudaMemPoolPtrExportData exportData;
    if (rpc_read(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0)
        return -1;
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolExportPointer(&exportData, &ptr);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaMemPoolImportPointer(void *conn)
{
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    cudaMemPool_t memPool;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        return -1;
    struct cudaMemPoolPtrExportData exportData;
    if (rpc_read(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaMemPoolImportPointer(&ptr, memPool, &exportData);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceCanAccessPeer(void *conn)
{
    int canAccessPeer;
    if (rpc_read(conn, &canAccessPeer, sizeof(int)) < 0)
        return -1;
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;
    int peerDevice;
    if (rpc_read(conn, &peerDevice, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &canAccessPeer, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceEnablePeerAccess(void *conn)
{
    int peerDevice;
    if (rpc_read(conn, &peerDevice, sizeof(int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceEnablePeerAccess(peerDevice, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceDisablePeerAccess(void *conn)
{
    int peerDevice;
    if (rpc_read(conn, &peerDevice, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceDisablePeerAccess(peerDevice);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsUnregisterResource(void *conn)
{
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsUnregisterResource(resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsResourceSetMapFlags(void *conn)
{
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsResourceSetMapFlags(resource, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsMapResources(void *conn)
{
    int count;
    if (rpc_read(conn, &count, sizeof(int)) < 0)
        return -1;
    cudaGraphicsResource_t resources;
    if (rpc_read(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsMapResources(count, &resources, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsUnmapResources(void *conn)
{
    int count;
    if (rpc_read(conn, &count, sizeof(int)) < 0)
        return -1;
    cudaGraphicsResource_t resources;
    if (rpc_read(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsUnmapResources(count, &resources, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsResourceGetMappedPointer(void *conn)
{
    void* devPtr;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    std::size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsResourceGetMappedPointer(&devPtr, &size, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &devPtr, sizeof(void*)) < 0)
        return -1;
    if (rpc_write(conn, &size, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsSubResourceGetMappedArray(void *conn)
{
    cudaArray_t array;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        return -1;
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;
    unsigned int arrayIndex;
    if (rpc_read(conn, &arrayIndex, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int mipLevel;
    if (rpc_read(conn, &mipLevel, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsSubResourceGetMappedArray(&array, resource, arrayIndex, mipLevel);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &array, sizeof(cudaArray_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphicsResourceGetMappedMipmappedArray(void *conn)
{
    cudaMipmappedArray_t mipmappedArray;
    if (rpc_read(conn, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0)
        return -1;
    cudaGraphicsResource_t resource;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphicsResourceGetMappedMipmappedArray(&mipmappedArray, resource);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGetChannelDesc(void *conn)
{
    struct cudaChannelFormatDesc desc;
    if (rpc_read(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0)
        return -1;
    cudaArray_const_t array;
    if (rpc_read(conn, &array, sizeof(cudaArray_const_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetChannelDesc(&desc, array);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0)
        return -1;

    return result;
}

int handle_cudaDestroyTextureObject(void *conn)
{
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDestroyTextureObject(texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGetTextureObjectResourceDesc(void *conn)
{
    struct cudaResourceDesc pResDesc;
    if (rpc_read(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0)
        return -1;
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetTextureObjectResourceDesc(&pResDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0)
        return -1;

    return result;
}

int handle_cudaGetTextureObjectTextureDesc(void *conn)
{
    struct cudaTextureDesc pTexDesc;
    if (rpc_read(conn, &pTexDesc, sizeof(struct cudaTextureDesc)) < 0)
        return -1;
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetTextureObjectTextureDesc(&pTexDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pTexDesc, sizeof(struct cudaTextureDesc)) < 0)
        return -1;

    return result;
}

int handle_cudaGetTextureObjectResourceViewDesc(void *conn)
{
    struct cudaResourceViewDesc pResViewDesc;
    if (rpc_read(conn, &pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0)
        return -1;
    cudaTextureObject_t texObject;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetTextureObjectResourceViewDesc(&pResViewDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0)
        return -1;

    return result;
}

int handle_cudaDestroySurfaceObject(void *conn)
{
    cudaSurfaceObject_t surfObject;
    if (rpc_read(conn, &surfObject, sizeof(cudaSurfaceObject_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDestroySurfaceObject(surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGetSurfaceObjectResourceDesc(void *conn)
{
    struct cudaResourceDesc pResDesc;
    if (rpc_read(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0)
        return -1;
    cudaSurfaceObject_t surfObject;
    if (rpc_read(conn, &surfObject, sizeof(cudaSurfaceObject_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGetSurfaceObjectResourceDesc(&pResDesc, surfObject);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0)
        return -1;

    return result;
}

int handle_cudaDriverGetVersion(void *conn)
{
    int driverVersion;
    if (rpc_read(conn, &driverVersion, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDriverGetVersion(&driverVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &driverVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaRuntimeGetVersion(void *conn)
{
    int runtimeVersion;
    if (rpc_read(conn, &runtimeVersion, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaRuntimeGetVersion(&runtimeVersion);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &runtimeVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphCreate(void *conn)
{
    cudaGraph_t pGraph;
    if (rpc_read(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphCreate(&pGraph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphKernelNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaKernelNodeParams pNodeParams;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphKernelNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphKernelNodeCopyAttributes(void *conn)
{
    cudaGraphNode_t hSrc;
    if (rpc_read(conn, &hSrc, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraphNode_t hDst;
    if (rpc_read(conn, &hDst, sizeof(cudaGraphNode_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphKernelNodeCopyAttributes(hSrc, hDst);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphKernelNodeGetAttribute(void *conn)
{
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaLaunchAttributeID attr;
    if (rpc_read(conn, &attr, sizeof(cudaLaunchAttributeID)) < 0)
        return -1;
    cudaLaunchAttributeValue value_out;
    if (rpc_read(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphKernelNodeGetAttribute(hNode, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphMemcpyNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaMemcpy3DParms pNodeParams;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphMemcpyNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphMemsetNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaMemsetParams pNodeParams;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaMemsetParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphMemsetNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(struct cudaMemsetParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphHostNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaHostNodeParams pNodeParams;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaHostNodeParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphHostNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNodeParams, sizeof(struct cudaHostNodeParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphChildGraphNodeGetGraph(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraph_t pGraph;
    if (rpc_read(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphChildGraphNodeGetGraph(node, &pGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphEventRecordNodeGetEvent(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaEvent_t event_out;
    if (rpc_read(conn, &event_out, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphEventRecordNodeGetEvent(node, &event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(cudaEvent_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphEventRecordNodeSetEvent(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphEventRecordNodeSetEvent(node, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphEventWaitNodeGetEvent(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaEvent_t event_out;
    if (rpc_read(conn, &event_out, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphEventWaitNodeGetEvent(node, &event_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &event_out, sizeof(cudaEvent_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphEventWaitNodeSetEvent(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphEventWaitNodeSetEvent(node, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *conn)
{
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaExternalSemaphoreSignalNodeParams params_out;
    if (rpc_read(conn, &params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *conn)
{
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaExternalSemaphoreWaitNodeParams params_out;
    if (rpc_read(conn, &params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphMemAllocNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    struct cudaMemAllocNodeParams params_out;
    if (rpc_read(conn, &params_out, sizeof(struct cudaMemAllocNodeParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphMemAllocNodeGetParams(node, &params_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &params_out, sizeof(struct cudaMemAllocNodeParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphMemFreeNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    void* dptr_out;
    if (rpc_read(conn, &dptr_out, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphMemFreeNodeGetParams(node, &dptr_out);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &dptr_out, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGraphMemTrim(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGraphMemTrim(device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceGetGraphMemAttribute(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;
    enum cudaGraphMemAttributeType attr;
    if (rpc_read(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceGetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaDeviceSetGraphMemAttribute(void *conn)
{
    int device;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        return -1;
    enum cudaGraphMemAttributeType attr;
    if (rpc_read(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0)
        return -1;
    void* value;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaDeviceSetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &value, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphClone(void *conn)
{
    cudaGraph_t pGraphClone;
    if (rpc_read(conn, &pGraphClone, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaGraph_t originalGraph;
    if (rpc_read(conn, &originalGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphClone(&pGraphClone, originalGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphClone, sizeof(cudaGraph_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphNodeFindInClone(void *conn)
{
    cudaGraphNode_t pNode;
    if (rpc_read(conn, &pNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraphNode_t originalNode;
    if (rpc_read(conn, &originalNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraph_t clonedGraph;
    if (rpc_read(conn, &clonedGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphNodeFindInClone(&pNode, originalNode, clonedGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphNodeGetType(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    enum cudaGraphNodeType pType;
    if (rpc_read(conn, &pType, sizeof(enum cudaGraphNodeType)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphNodeGetType(node, &pType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pType, sizeof(enum cudaGraphNodeType)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphGetNodes(void *conn)
{
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaGraphNode_t nodes;
    if (rpc_read(conn, &nodes, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    std::size_t numNodes;
    if (rpc_read(conn, &numNodes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphGetNodes(graph, &nodes, &numNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &nodes, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    if (rpc_write(conn, &numNodes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphGetRootNodes(void *conn)
{
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaGraphNode_t pRootNodes;
    if (rpc_read(conn, &pRootNodes, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    std::size_t pNumRootNodes;
    if (rpc_read(conn, &pNumRootNodes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphGetRootNodes(graph, &pRootNodes, &pNumRootNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pRootNodes, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    if (rpc_write(conn, &pNumRootNodes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphGetEdges(void *conn)
{
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaGraphNode_t from;
    if (rpc_read(conn, &from, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraphNode_t to;
    if (rpc_read(conn, &to, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    std::size_t numEdges;
    if (rpc_read(conn, &numEdges, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphGetEdges(graph, &from, &to, &numEdges);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &from, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    if (rpc_write(conn, &to, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    if (rpc_write(conn, &numEdges, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphNodeGetDependencies(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraphNode_t pDependencies;
    if (rpc_read(conn, &pDependencies, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    std::size_t pNumDependencies;
    if (rpc_read(conn, &pNumDependencies, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphNodeGetDependencies(node, &pDependencies, &pNumDependencies);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDependencies, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    if (rpc_write(conn, &pNumDependencies, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphNodeGetDependentNodes(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraphNode_t pDependentNodes;
    if (rpc_read(conn, &pDependentNodes, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    std::size_t pNumDependentNodes;
    if (rpc_read(conn, &pNumDependentNodes, sizeof(size_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphNodeGetDependentNodes(node, &pDependentNodes, &pNumDependentNodes);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pDependentNodes, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    if (rpc_write(conn, &pNumDependentNodes, sizeof(size_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphDestroyNode(void *conn)
{
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphDestroyNode(node);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphInstantiate(void *conn)
{
    cudaGraphExec_t pGraphExec;
    if (rpc_read(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphInstantiate(&pGraphExec, graph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphInstantiateWithFlags(void *conn)
{
    cudaGraphExec_t pGraphExec;
    if (rpc_read(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphInstantiateWithFlags(&pGraphExec, graph, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphInstantiateWithParams(void *conn)
{
    cudaGraphExec_t pGraphExec;
    if (rpc_read(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaGraphInstantiateParams instantiateParams;
    if (rpc_read(conn, &instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphInstantiateWithParams(&pGraphExec, graph, &instantiateParams);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    if (rpc_write(conn, &instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExecGetFlags(void *conn)
{
    cudaGraphExec_t graphExec;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    unsigned long long flags;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExecGetFlags(graphExec, &flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &flags, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExecChildGraphNodeSetParams(void *conn)
{
    cudaGraphExec_t hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraphNode_t node;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaGraph_t childGraph;
    if (rpc_read(conn, &childGraph, sizeof(cudaGraph_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExecEventRecordNodeSetEvent(void *conn)
{
    cudaGraphExec_t hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExecEventWaitNodeSetEvent(void *conn)
{
    cudaGraphExec_t hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    cudaEvent_t event;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphNodeSetEnabled(void *conn)
{
    cudaGraphExec_t hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    unsigned int isEnabled;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphNodeGetEnabled(void *conn)
{
    cudaGraphExec_t hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraphNode_t hNode;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        return -1;
    unsigned int isEnabled;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphNodeGetEnabled(hGraphExec, hNode, &isEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExecUpdate(void *conn)
{
    cudaGraphExec_t hGraphExec;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaGraph_t hGraph;
    if (rpc_read(conn, &hGraph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaGraphExecUpdateResultInfo resultInfo;
    if (rpc_read(conn, &resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0)
        return -1;

    return result;
}

int handle_cudaGraphUpload(void *conn)
{
    cudaGraphExec_t graphExec;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphUpload(graphExec, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphLaunch(void *conn)
{
    cudaGraphExec_t graphExec;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;
    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphLaunch(graphExec, stream);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphExecDestroy(void *conn)
{
    cudaGraphExec_t graphExec;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphExecDestroy(graphExec);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphDestroy(void *conn)
{
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphDestroy(graph);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaUserObjectCreate(void *conn)
{
    cudaUserObject_t object_out;
    if (rpc_read(conn, &object_out, sizeof(cudaUserObject_t)) < 0)
        return -1;
    void* ptr;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        return -1;
    cudaHostFn_t destroy;
    if (rpc_read(conn, &destroy, sizeof(cudaHostFn_t)) < 0)
        return -1;
    unsigned int initialRefcount;
    if (rpc_read(conn, &initialRefcount, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaUserObjectCreate(&object_out, &ptr, destroy, initialRefcount, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;
    if (rpc_write(conn, &object_out, sizeof(cudaUserObject_t)) < 0)
        return -1;
    if (rpc_write(conn, &ptr, sizeof(void*)) < 0)
        return -1;

    return result;
}

int handle_cudaUserObjectRetain(void *conn)
{
    cudaUserObject_t object;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaUserObjectRetain(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaUserObjectRelease(void *conn)
{
    cudaUserObject_t object;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaUserObjectRelease(object, count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphRetainUserObject(void *conn)
{
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaUserObject_t object;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;
    unsigned int flags;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphRetainUserObject(graph, object, count, flags);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_cudaGraphReleaseUserObject(void *conn)
{
    cudaGraph_t graph;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        return -1;
    cudaUserObject_t object;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        return -1;
    unsigned int count;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = cudaGraphReleaseUserObject(graph, object, count);

    if (rpc_start_response(conn, request_id) < 0)
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
    handle_nvmlGpuInstanceDestroy,
    handle_nvmlDeviceGetGpuInstances,
    handle_nvmlDeviceGetGpuInstanceById,
    handle_nvmlGpuInstanceGetInfo,
    handle_nvmlGpuInstanceGetComputeInstanceProfileInfo,
    handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV,
    handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity,
    handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements,
    handle_nvmlGpuInstanceCreateComputeInstance,
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
    handle_cuInit,
    handle_cuDriverGetVersion,
    handle_cuDeviceGet,
    handle_cuDeviceGetCount,
    handle_cuDeviceGetName,
    handle_cuDeviceGetUuid,
    handle_cuDeviceGetUuid_v2,
    handle_cuDeviceGetLuid,
    handle_cuDeviceTotalMem_v2,
    handle_cuDeviceGetTexture1DLinearMaxWidth,
    handle_cuDeviceGetAttribute,
    handle_cuDeviceSetMemPool,
    handle_cuDeviceGetMemPool,
    handle_cuDeviceGetDefaultMemPool,
    handle_cuDeviceGetExecAffinitySupport,
    handle_cuFlushGPUDirectRDMAWrites,
    handle_cuDeviceGetProperties,
    handle_cuDeviceComputeCapability,
    handle_cuDevicePrimaryCtxRetain,
    handle_cuDevicePrimaryCtxRelease_v2,
    handle_cuDevicePrimaryCtxSetFlags_v2,
    handle_cuDevicePrimaryCtxGetState,
    handle_cuDevicePrimaryCtxReset_v2,
    handle_cuCtxCreate_v2,
    handle_cuCtxCreate_v3,
    handle_cuCtxDestroy_v2,
    handle_cuCtxPushCurrent_v2,
    handle_cuCtxPopCurrent_v2,
    handle_cuCtxSetCurrent,
    handle_cuCtxGetCurrent,
    handle_cuCtxGetDevice,
    handle_cuCtxGetFlags,
    handle_cuCtxGetId,
    handle_cuCtxSynchronize,
    handle_cuCtxSetLimit,
    handle_cuCtxGetLimit,
    handle_cuCtxGetCacheConfig,
    handle_cuCtxSetCacheConfig,
    handle_cuCtxGetSharedMemConfig,
    handle_cuCtxSetSharedMemConfig,
    handle_cuCtxGetApiVersion,
    handle_cuCtxGetStreamPriorityRange,
    handle_cuCtxResetPersistingL2Cache,
    handle_cuCtxGetExecAffinity,
    handle_cuCtxAttach,
    handle_cuCtxDetach,
    handle_cuModuleLoad,
    handle_cuModuleUnload,
    handle_cuModuleGetLoadingMode,
    handle_cuModuleGetFunction,
    handle_cuModuleGetGlobal_v2,
    handle_cuLinkCreate_v2,
    handle_cuLinkAddData_v2,
    handle_cuLinkAddFile_v2,
    handle_cuLinkComplete,
    handle_cuLinkDestroy,
    handle_cuModuleGetTexRef,
    handle_cuModuleGetSurfRef,
    handle_cuLibraryLoadFromFile,
    handle_cuLibraryUnload,
    handle_cuLibraryGetKernel,
    handle_cuLibraryGetModule,
    handle_cuKernelGetFunction,
    handle_cuLibraryGetGlobal,
    handle_cuLibraryGetManaged,
    handle_cuLibraryGetUnifiedFunction,
    handle_cuKernelGetAttribute,
    handle_cuKernelSetAttribute,
    handle_cuKernelSetCacheConfig,
    handle_cuMemGetInfo_v2,
    handle_cuMemAlloc_v2,
    handle_cuMemAllocPitch_v2,
    handle_cuMemFree_v2,
    handle_cuMemGetAddressRange_v2,
    handle_cuMemAllocHost_v2,
    handle_cuMemFreeHost,
    handle_cuMemHostAlloc,
    handle_cuMemHostGetDevicePointer_v2,
    handle_cuMemHostGetFlags,
    handle_cuMemAllocManaged,
    handle_cuDeviceGetByPCIBusId,
    handle_cuDeviceGetPCIBusId,
    handle_cuIpcGetEventHandle,
    handle_cuIpcOpenEventHandle,
    handle_cuIpcGetMemHandle,
    handle_cuIpcOpenMemHandle_v2,
    handle_cuIpcCloseMemHandle,
    handle_cuMemHostRegister_v2,
    handle_cuMemHostUnregister,
    handle_cuMemcpy,
    handle_cuMemcpyPeer,
    handle_cuMemcpyDtoH_v2,
    handle_cuMemcpyDtoD_v2,
    handle_cuMemcpyDtoA_v2,
    handle_cuMemcpyAtoD_v2,
    handle_cuMemcpyAtoH_v2,
    handle_cuMemcpyAtoA_v2,
    handle_cuMemcpyAsync,
    handle_cuMemcpyPeerAsync,
    handle_cuMemcpyDtoHAsync_v2,
    handle_cuMemcpyDtoDAsync_v2,
    handle_cuMemcpyAtoHAsync_v2,
    handle_cuMemsetD8_v2,
    handle_cuMemsetD16_v2,
    handle_cuMemsetD32_v2,
    handle_cuMemsetD2D8_v2,
    handle_cuMemsetD2D16_v2,
    handle_cuMemsetD2D32_v2,
    handle_cuMemsetD8Async,
    handle_cuMemsetD16Async,
    handle_cuMemsetD32Async,
    handle_cuMemsetD2D8Async,
    handle_cuMemsetD2D16Async,
    handle_cuMemsetD2D32Async,
    handle_cuArrayGetDescriptor_v2,
    handle_cuArrayGetSparseProperties,
    handle_cuMipmappedArrayGetSparseProperties,
    handle_cuArrayGetMemoryRequirements,
    handle_cuMipmappedArrayGetMemoryRequirements,
    handle_cuArrayGetPlane,
    handle_cuArrayDestroy,
    handle_cuArray3DGetDescriptor_v2,
    handle_cuMipmappedArrayGetLevel,
    handle_cuMipmappedArrayDestroy,
    handle_cuMemGetHandleForAddressRange,
    handle_cuMemAddressReserve,
    handle_cuMemAddressFree,
    handle_cuMemRelease,
    handle_cuMemMap,
    handle_cuMemMapArrayAsync,
    handle_cuMemUnmap,
    handle_cuMemExportToShareableHandle,
    handle_cuMemImportFromShareableHandle,
    handle_cuMemGetAllocationPropertiesFromHandle,
    handle_cuMemRetainAllocationHandle,
    handle_cuMemFreeAsync,
    handle_cuMemAllocAsync,
    handle_cuMemPoolTrimTo,
    handle_cuMemPoolSetAttribute,
    handle_cuMemPoolGetAttribute,
    handle_cuMemPoolGetAccess,
    handle_cuMemPoolDestroy,
    handle_cuMemAllocFromPoolAsync,
    handle_cuMemPoolExportToShareableHandle,
    handle_cuMemPoolImportFromShareableHandle,
    handle_cuMemPoolExportPointer,
    handle_cuMemPoolImportPointer,
    handle_cuPointerGetAttribute,
    handle_cuMemPrefetchAsync,
    handle_cuMemAdvise,
    handle_cuMemRangeGetAttribute,
    handle_cuMemRangeGetAttributes,
    handle_cuPointerGetAttributes,
    handle_cuStreamCreate,
    handle_cuStreamCreateWithPriority,
    handle_cuStreamGetPriority,
    handle_cuStreamGetFlags,
    handle_cuStreamGetId,
    handle_cuStreamGetCtx,
    handle_cuStreamWaitEvent,
    handle_cuStreamAddCallback,
    handle_cuStreamBeginCapture_v2,
    handle_cuThreadExchangeStreamCaptureMode,
    handle_cuStreamEndCapture,
    handle_cuStreamIsCapturing,
    handle_cuStreamUpdateCaptureDependencies,
    handle_cuStreamAttachMemAsync,
    handle_cuStreamQuery,
    handle_cuStreamSynchronize,
    handle_cuStreamDestroy_v2,
    handle_cuStreamCopyAttributes,
    handle_cuStreamGetAttribute,
    handle_cuEventCreate,
    handle_cuEventRecord,
    handle_cuEventRecordWithFlags,
    handle_cuEventQuery,
    handle_cuEventSynchronize,
    handle_cuEventDestroy_v2,
    handle_cuEventElapsedTime,
    handle_cuDestroyExternalMemory,
    handle_cuDestroyExternalSemaphore,
    handle_cuStreamWaitValue32_v2,
    handle_cuStreamWaitValue64_v2,
    handle_cuStreamWriteValue32_v2,
    handle_cuStreamWriteValue64_v2,
    handle_cuStreamBatchMemOp_v2,
    handle_cuFuncGetAttribute,
    handle_cuFuncSetAttribute,
    handle_cuFuncSetCacheConfig,
    handle_cuFuncSetSharedMemConfig,
    handle_cuFuncGetModule,
    handle_cuLaunchKernel,
    handle_cuLaunchCooperativeKernel,
    handle_cuLaunchCooperativeKernelMultiDevice,
    handle_cuLaunchHostFunc,
    handle_cuFuncSetBlockShape,
    handle_cuFuncSetSharedSize,
    handle_cuParamSetSize,
    handle_cuParamSeti,
    handle_cuParamSetf,
    handle_cuParamSetv,
    handle_cuLaunch,
    handle_cuLaunchGrid,
    handle_cuLaunchGridAsync,
    handle_cuParamSetTexRef,
    handle_cuGraphCreate,
    handle_cuGraphKernelNodeGetParams_v2,
    handle_cuGraphMemcpyNodeGetParams,
    handle_cuGraphMemsetNodeGetParams,
    handle_cuGraphHostNodeGetParams,
    handle_cuGraphChildGraphNodeGetGraph,
    handle_cuGraphEventRecordNodeGetEvent,
    handle_cuGraphEventRecordNodeSetEvent,
    handle_cuGraphEventWaitNodeGetEvent,
    handle_cuGraphEventWaitNodeSetEvent,
    handle_cuGraphExternalSemaphoresSignalNodeGetParams,
    handle_cuGraphExternalSemaphoresWaitNodeGetParams,
    handle_cuGraphBatchMemOpNodeGetParams,
    handle_cuGraphMemAllocNodeGetParams,
    handle_cuGraphMemFreeNodeGetParams,
    handle_cuDeviceGraphMemTrim,
    handle_cuDeviceGetGraphMemAttribute,
    handle_cuDeviceSetGraphMemAttribute,
    handle_cuGraphClone,
    handle_cuGraphNodeFindInClone,
    handle_cuGraphNodeGetType,
    handle_cuGraphGetNodes,
    handle_cuGraphGetRootNodes,
    handle_cuGraphGetEdges,
    handle_cuGraphNodeGetDependencies,
    handle_cuGraphNodeGetDependentNodes,
    handle_cuGraphDestroyNode,
    handle_cuGraphInstantiateWithFlags,
    handle_cuGraphInstantiateWithParams,
    handle_cuGraphExecGetFlags,
    handle_cuGraphExecChildGraphNodeSetParams,
    handle_cuGraphExecEventRecordNodeSetEvent,
    handle_cuGraphExecEventWaitNodeSetEvent,
    handle_cuGraphNodeSetEnabled,
    handle_cuGraphNodeGetEnabled,
    handle_cuGraphUpload,
    handle_cuGraphLaunch,
    handle_cuGraphExecDestroy,
    handle_cuGraphDestroy,
    handle_cuGraphExecUpdate_v2,
    handle_cuGraphKernelNodeCopyAttributes,
    handle_cuGraphKernelNodeGetAttribute,
    handle_cuUserObjectCreate,
    handle_cuUserObjectRetain,
    handle_cuUserObjectRelease,
    handle_cuGraphRetainUserObject,
    handle_cuGraphReleaseUserObject,
    handle_cuOccupancyMaxActiveBlocksPerMultiprocessor,
    handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
    handle_cuOccupancyAvailableDynamicSMemPerBlock,
    handle_cuTexRefSetArray,
    handle_cuTexRefSetMipmappedArray,
    handle_cuTexRefSetAddress_v2,
    handle_cuTexRefSetFormat,
    handle_cuTexRefSetAddressMode,
    handle_cuTexRefSetFilterMode,
    handle_cuTexRefSetMipmapFilterMode,
    handle_cuTexRefSetMipmapLevelBias,
    handle_cuTexRefSetMipmapLevelClamp,
    handle_cuTexRefSetMaxAnisotropy,
    handle_cuTexRefSetBorderColor,
    handle_cuTexRefSetFlags,
    handle_cuTexRefGetAddress_v2,
    handle_cuTexRefGetArray,
    handle_cuTexRefGetMipmappedArray,
    handle_cuTexRefGetAddressMode,
    handle_cuTexRefGetFilterMode,
    handle_cuTexRefGetFormat,
    handle_cuTexRefGetMipmapFilterMode,
    handle_cuTexRefGetMipmapLevelBias,
    handle_cuTexRefGetMipmapLevelClamp,
    handle_cuTexRefGetMaxAnisotropy,
    handle_cuTexRefGetBorderColor,
    handle_cuTexRefGetFlags,
    handle_cuTexRefCreate,
    handle_cuTexRefDestroy,
    handle_cuSurfRefSetArray,
    handle_cuSurfRefGetArray,
    handle_cuTexObjectDestroy,
    handle_cuTexObjectGetResourceDesc,
    handle_cuTexObjectGetTextureDesc,
    handle_cuTexObjectGetResourceViewDesc,
    handle_cuSurfObjectDestroy,
    handle_cuSurfObjectGetResourceDesc,
    handle_cuTensorMapReplaceAddress,
    handle_cuDeviceCanAccessPeer,
    handle_cuCtxEnablePeerAccess,
    handle_cuCtxDisablePeerAccess,
    handle_cuDeviceGetP2PAttribute,
    handle_cuGraphicsUnregisterResource,
    handle_cuGraphicsSubResourceGetMappedArray,
    handle_cuGraphicsResourceGetMappedMipmappedArray,
    handle_cuGraphicsResourceGetMappedPointer_v2,
    handle_cuGraphicsResourceSetMapFlags_v2,
    handle_cuGraphicsMapResources,
    handle_cuGraphicsUnmapResources,
    handle_cudaDeviceReset,
    handle_cudaDeviceSynchronize,
    handle_cudaDeviceSetLimit,
    handle_cudaDeviceGetLimit,
    handle_cudaDeviceGetCacheConfig,
    handle_cudaDeviceGetStreamPriorityRange,
    handle_cudaDeviceSetCacheConfig,
    handle_cudaDeviceGetSharedMemConfig,
    handle_cudaDeviceSetSharedMemConfig,
    handle_cudaDeviceGetPCIBusId,
    handle_cudaIpcGetEventHandle,
    handle_cudaIpcOpenEventHandle,
    handle_cudaIpcGetMemHandle,
    handle_cudaIpcOpenMemHandle,
    handle_cudaIpcCloseMemHandle,
    handle_cudaDeviceFlushGPUDirectRDMAWrites,
    handle_cudaThreadExit,
    handle_cudaThreadSynchronize,
    handle_cudaThreadSetLimit,
    handle_cudaThreadGetLimit,
    handle_cudaThreadGetCacheConfig,
    handle_cudaThreadSetCacheConfig,
    handle_cudaGetLastError,
    handle_cudaPeekAtLastError,
    handle_cudaGetDeviceCount,
    handle_cudaGetDeviceProperties_v2,
    handle_cudaDeviceGetAttribute,
    handle_cudaDeviceGetDefaultMemPool,
    handle_cudaDeviceSetMemPool,
    handle_cudaDeviceGetMemPool,
    handle_cudaDeviceGetNvSciSyncAttributes,
    handle_cudaDeviceGetP2PAttribute,
    handle_cudaInitDevice,
    handle_cudaSetDevice,
    handle_cudaGetDevice,
    handle_cudaSetValidDevices,
    handle_cudaSetDeviceFlags,
    handle_cudaGetDeviceFlags,
    handle_cudaStreamCreate,
    handle_cudaStreamCreateWithFlags,
    handle_cudaStreamCreateWithPriority,
    handle_cudaStreamGetPriority,
    handle_cudaStreamGetFlags,
    handle_cudaStreamGetId,
    handle_cudaCtxResetPersistingL2Cache,
    handle_cudaStreamCopyAttributes,
    handle_cudaStreamGetAttribute,
    handle_cudaStreamDestroy,
    handle_cudaStreamWaitEvent,
    handle_cudaStreamAddCallback,
    handle_cudaStreamSynchronize,
    handle_cudaStreamQuery,
    handle_cudaStreamAttachMemAsync,
    handle_cudaStreamBeginCapture,
    handle_cudaThreadExchangeStreamCaptureMode,
    handle_cudaStreamEndCapture,
    handle_cudaStreamIsCapturing,
    handle_cudaStreamGetCaptureInfo_v2,
    handle_cudaStreamUpdateCaptureDependencies,
    handle_cudaEventCreate,
    handle_cudaEventCreateWithFlags,
    handle_cudaEventRecord,
    handle_cudaEventRecordWithFlags,
    handle_cudaEventQuery,
    handle_cudaEventSynchronize,
    handle_cudaEventDestroy,
    handle_cudaEventElapsedTime,
    handle_cudaDestroyExternalMemory,
    handle_cudaDestroyExternalSemaphore,
    handle_cudaLaunchCooperativeKernelMultiDevice,
    handle_cudaSetDoubleForDevice,
    handle_cudaSetDoubleForHost,
    handle_cudaLaunchHostFunc,
    handle_cudaMallocManaged,
    handle_cudaMalloc,
    handle_cudaMallocHost,
    handle_cudaMallocPitch,
    handle_cudaFree,
    handle_cudaFreeHost,
    handle_cudaFreeArray,
    handle_cudaFreeMipmappedArray,
    handle_cudaHostAlloc,
    handle_cudaHostRegister,
    handle_cudaHostUnregister,
    handle_cudaHostGetDevicePointer,
    handle_cudaHostGetFlags,
    handle_cudaMalloc3D,
    handle_cudaGetMipmappedArrayLevel,
    handle_cudaMemGetInfo,
    handle_cudaArrayGetInfo,
    handle_cudaArrayGetPlane,
    handle_cudaArrayGetMemoryRequirements,
    handle_cudaMipmappedArrayGetMemoryRequirements,
    handle_cudaArrayGetSparseProperties,
    handle_cudaMipmappedArrayGetSparseProperties,
    handle_cudaMemcpy2DFromArray,
    handle_cudaMemcpy2DArrayToArray,
    handle_cudaMemcpyAsync,
    handle_cudaMemcpy2DFromArrayAsync,
    handle_cudaMemset,
    handle_cudaMemset2D,
    handle_cudaMemset3D,
    handle_cudaMemsetAsync,
    handle_cudaMemset2DAsync,
    handle_cudaMemset3DAsync,
    handle_cudaMemcpyFromArray,
    handle_cudaMemcpyArrayToArray,
    handle_cudaMemcpyFromArrayAsync,
    handle_cudaMallocAsync,
    handle_cudaFreeAsync,
    handle_cudaMemPoolTrimTo,
    handle_cudaMemPoolSetAttribute,
    handle_cudaMemPoolGetAttribute,
    handle_cudaMemPoolGetAccess,
    handle_cudaMemPoolDestroy,
    handle_cudaMallocFromPoolAsync,
    handle_cudaMemPoolExportToShareableHandle,
    handle_cudaMemPoolImportFromShareableHandle,
    handle_cudaMemPoolExportPointer,
    handle_cudaMemPoolImportPointer,
    handle_cudaDeviceCanAccessPeer,
    handle_cudaDeviceEnablePeerAccess,
    handle_cudaDeviceDisablePeerAccess,
    handle_cudaGraphicsUnregisterResource,
    handle_cudaGraphicsResourceSetMapFlags,
    handle_cudaGraphicsMapResources,
    handle_cudaGraphicsUnmapResources,
    handle_cudaGraphicsResourceGetMappedPointer,
    handle_cudaGraphicsSubResourceGetMappedArray,
    handle_cudaGraphicsResourceGetMappedMipmappedArray,
    handle_cudaGetChannelDesc,
    handle_cudaDestroyTextureObject,
    handle_cudaGetTextureObjectResourceDesc,
    handle_cudaGetTextureObjectTextureDesc,
    handle_cudaGetTextureObjectResourceViewDesc,
    handle_cudaDestroySurfaceObject,
    handle_cudaGetSurfaceObjectResourceDesc,
    handle_cudaDriverGetVersion,
    handle_cudaRuntimeGetVersion,
    handle_cudaGraphCreate,
    handle_cudaGraphKernelNodeGetParams,
    handle_cudaGraphKernelNodeCopyAttributes,
    handle_cudaGraphKernelNodeGetAttribute,
    handle_cudaGraphMemcpyNodeGetParams,
    handle_cudaGraphMemsetNodeGetParams,
    handle_cudaGraphHostNodeGetParams,
    handle_cudaGraphChildGraphNodeGetGraph,
    handle_cudaGraphEventRecordNodeGetEvent,
    handle_cudaGraphEventRecordNodeSetEvent,
    handle_cudaGraphEventWaitNodeGetEvent,
    handle_cudaGraphEventWaitNodeSetEvent,
    handle_cudaGraphExternalSemaphoresSignalNodeGetParams,
    handle_cudaGraphExternalSemaphoresWaitNodeGetParams,
    handle_cudaGraphMemAllocNodeGetParams,
    handle_cudaGraphMemFreeNodeGetParams,
    handle_cudaDeviceGraphMemTrim,
    handle_cudaDeviceGetGraphMemAttribute,
    handle_cudaDeviceSetGraphMemAttribute,
    handle_cudaGraphClone,
    handle_cudaGraphNodeFindInClone,
    handle_cudaGraphNodeGetType,
    handle_cudaGraphGetNodes,
    handle_cudaGraphGetRootNodes,
    handle_cudaGraphGetEdges,
    handle_cudaGraphNodeGetDependencies,
    handle_cudaGraphNodeGetDependentNodes,
    handle_cudaGraphDestroyNode,
    handle_cudaGraphInstantiate,
    handle_cudaGraphInstantiateWithFlags,
    handle_cudaGraphInstantiateWithParams,
    handle_cudaGraphExecGetFlags,
    handle_cudaGraphExecChildGraphNodeSetParams,
    handle_cudaGraphExecEventRecordNodeSetEvent,
    handle_cudaGraphExecEventWaitNodeSetEvent,
    handle_cudaGraphNodeSetEnabled,
    handle_cudaGraphNodeGetEnabled,
    handle_cudaGraphExecUpdate,
    handle_cudaGraphUpload,
    handle_cudaGraphLaunch,
    handle_cudaGraphExecDestroy,
    handle_cudaGraphDestroy,
    handle_cudaUserObjectCreate,
    handle_cudaUserObjectRetain,
    handle_cudaUserObjectRelease,
    handle_cudaGraphRetainUserObject,
    handle_cudaGraphReleaseUserObject,
};

RequestHandler get_handler(const int op)
{
    return opHandlers[op];
}
