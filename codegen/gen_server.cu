#include <nvml.h>

#include <nvml.h>
#include <cuda.h>

#include "gen_api.h"
#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const size_t size);
extern int rpc_write(const void *conn, const void *data, const size_t size);
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

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *version = (char *)malloc(length * sizeof(char));

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

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *version = (char *)malloc(length * sizeof(char));

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

    if (rpc_read(conn, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

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

    if (rpc_read(conn, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

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
    unsigned int length;

    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *name = (char *)malloc(length * sizeof(char));

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

    if (rpc_read(conn, &unitCount, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlUnit_t unit;

    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;

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
    nvmlUnitInfo_t info;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlUnitInfo_t)) < 0)
        return -1;

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
    nvmlLedState_t state;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &state, sizeof(nvmlLedState_t)) < 0)
        return -1;

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
    nvmlPSUInfo_t psu;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &psu, sizeof(nvmlPSUInfo_t)) < 0)
        return -1;

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
    unsigned int type;
    unsigned int temp;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &type, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &temp, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlUnitFanSpeeds_t fanSpeeds;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return -1;

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
    unsigned int deviceCount;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlDevice_t *devices = (nvmlDevice_t *)malloc(deviceCount * sizeof(nvmlDevice_t));

    nvmlReturn_t result = nvmlUnitGetDevices(unit, &deviceCount, devices);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, devices, deviceCount * sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetHicVersion(void *conn)
{
    unsigned int hwbcCount;

    if (rpc_read(conn, &hwbcCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlHwbcEntry_t *hwbcEntries = (nvmlHwbcEntry_t *)malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));

    nvmlReturn_t result = nvmlSystemGetHicVersion(&hwbcCount, hwbcEntries);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, hwbcEntries, hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCount_v2(void *conn)
{
    unsigned int deviceCount;

    if (rpc_read(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlDeviceAttributes_t attributes;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0)
        return -1;

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
    nvmlDevice_t device;

    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

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
    char serial;
    nvmlDevice_t device;

    if (rpc_read(conn, &serial, sizeof(char)) < 0 ||
        rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleBySerial(&serial, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByUUID(void *conn)
{
    char uuid;
    nvmlDevice_t device;

    if (rpc_read(conn, &uuid, sizeof(char)) < 0 ||
        rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByUUID(&uuid, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(void *conn)
{
    char pciBusId;
    nvmlDevice_t device;

    if (rpc_read(conn, &pciBusId, sizeof(char)) < 0 ||
        rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByPciBusId_v2(&pciBusId, &device);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetName(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *name = (char *)malloc(length * sizeof(char));

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
    nvmlBrandType_t type;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &type, sizeof(nvmlBrandType_t)) < 0)
        return -1;

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
    unsigned int index;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *serial = (char *)malloc(length * sizeof(char));

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
    unsigned int nodeSetSize;
    nvmlAffinityScope_t scope;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &nodeSetSize, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned long *nodeSet = (unsigned long *)malloc(nodeSetSize * sizeof(unsigned long));

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
    unsigned int cpuSetSize;
    nvmlAffinityScope_t scope;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned long *cpuSet = (unsigned long *)malloc(cpuSetSize * sizeof(unsigned long));

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
    unsigned int cpuSetSize;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &cpuSetSize, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned long *cpuSet = (unsigned long *)malloc(cpuSetSize * sizeof(unsigned long));

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
    nvmlDevice_t device2;
    nvmlGpuTopologyLevel_t pathInfo;

    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;

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
    nvmlGpuTopologyLevel_t level;
    unsigned int count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlDevice_t *deviceArray = (nvmlDevice_t *)malloc(count * sizeof(nvmlDevice_t));

    nvmlReturn_t result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetTopologyGpuSet(void *conn)
{
    unsigned int cpuNumber;
    unsigned int count;
    nvmlDevice_t deviceArray;

    if (rpc_read(conn, &cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &deviceArray, sizeof(nvmlDevice_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetTopologyGpuSet(cpuNumber, &count, &deviceArray);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &deviceArray, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetP2PStatus(void *conn)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuP2PCapsIndex_t p2pIndex;
    nvmlGpuP2PStatus_t p2pStatus;

    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        rpc_read(conn, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0)
        return -1;

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
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *uuid = (char *)malloc(length * sizeof(char));

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
    unsigned int size;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *mdevUuid = (char *)malloc(size * sizeof(char));

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
    unsigned int minorNumber;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minorNumber, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *partNumber = (char *)malloc(length * sizeof(char));

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
    nvmlInforomObject_t object;
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &object, sizeof(nvmlInforomObject_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *version = (char *)malloc(length * sizeof(char));

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
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *version = (char *)malloc(length * sizeof(char));

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
    unsigned int checksum;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &checksum, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlEnableState_t display;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &display, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    nvmlEnableState_t isActive;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    nvmlEnableState_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    nvmlPciInfo_t pci;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

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
    unsigned int maxLinkGen;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &maxLinkGen, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int maxLinkGenDevice;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &maxLinkGenDevice, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int maxLinkWidth;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &maxLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int currLinkGen;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &currLinkGen, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int currLinkWidth;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &currLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlPcieUtilCounter_t counter;
    unsigned int value;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_read(conn, &value, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int value;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &value, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlClockType_t type;
    unsigned int clock;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &clock, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlClockType_t type;
    unsigned int clock;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &clock, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlClockType_t clockType;
    unsigned int clockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlClockType_t clockType;
    unsigned int clockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlClockType_t clockType;
    nvmlClockId_t clockId;
    unsigned int clockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_read(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlClockType_t clockType;
    unsigned int clockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int count;
    unsigned int clocksMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedMemoryClocks(device, &count, &clocksMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(void *conn)
{
    nvmlDevice_t device;
    unsigned int memoryClockMHz;
    unsigned int count;
    unsigned int clocksMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &memoryClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, &clocksMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t isEnabled;
    nvmlEnableState_t defaultIsEnabled;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(conn, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(conn, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t enabled;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &enabled, sizeof(nvmlEnableState_t)) < 0)
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
    nvmlEnableState_t enabled;
    unsigned int flags;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
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
    unsigned int speed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &speed, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int fan;
    unsigned int speed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fan, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &speed, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int fan;
    unsigned int targetSpeed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fan, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &targetSpeed, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int fan;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
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
    unsigned int minSpeed;
    unsigned int maxSpeed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fan, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;

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
    unsigned int fan;
    nvmlFanControlPolicy_t policy;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fan, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
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
    unsigned int numFans;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &numFans, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlTemperatureSensors_t sensorType;
    unsigned int temp;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_read(conn, &temp, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlTemperatureThresholds_t thresholdType;
    unsigned int temp;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_read(conn, &temp, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlTemperatureThresholds_t thresholdType;
    int temp;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_read(conn, &temp, sizeof(int)) < 0)
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
    unsigned int sensorIndex;
    nvmlGpuThermalSettings_t pThermalSettings;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
        return -1;

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
    nvmlPstates_t pState;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

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
    unsigned long long clocksThrottleReasons;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &clocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

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
    unsigned long long supportedClocksThrottleReasons;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

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
    nvmlPstates_t pState;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

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
    nvmlEnableState_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int limit;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &limit, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int minLimit;
    unsigned int maxLimit;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minLimit, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &maxLimit, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &minLimit, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &maxLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(void *conn)
{
    nvmlDevice_t device;
    unsigned int defaultLimit;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &defaultLimit, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int power;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &power, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned long long energy;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &energy, sizeof(unsigned long long)) < 0)
        return -1;

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
    unsigned int limit;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &limit, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlGpuOperationMode_t current;
    nvmlGpuOperationMode_t pending;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_read(conn, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuOperationMode(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_write(conn, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlMemory_t memory;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &memory, sizeof(nvmlMemory_t)) < 0)
        return -1;

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
    nvmlMemory_v2_t memory;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &memory, sizeof(nvmlMemory_v2_t)) < 0)
        return -1;

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
    nvmlComputeMode_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;

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
    int major;
    int minor;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &major, sizeof(int)) < 0 ||
        rpc_read(conn, &minor, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &major, sizeof(int)) < 0 ||
        rpc_write(conn, &minor, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEccMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t current;
    nvmlEnableState_t pending;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(conn, &pending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEccMode(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(conn, &pending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDefaultEccMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t defaultMode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &defaultMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int boardId;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &boardId, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int multiGpuBool;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &multiGpuBool, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    unsigned long long eccCounts;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_read(conn, &eccCounts, sizeof(unsigned long long)) < 0)
        return -1;

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
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlEccErrorCounts_t eccCounts;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_read(conn, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0)
        return -1;

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
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlMemoryLocation_t locationType;
    unsigned long long count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_read(conn, &locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned long long)) < 0)
        return -1;

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
    nvmlUtilization_t utilization;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &utilization, sizeof(nvmlUtilization_t)) < 0)
        return -1;

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
    unsigned int utilization;
    unsigned int samplingPeriodUs;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderUtilization(device, &utilization, &samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderCapacity(void *conn)
{
    nvmlDevice_t device;
    nvmlEncoderType_t encoderQueryType;
    unsigned int encoderCapacity;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_read(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderSessions(void *conn)
{
    nvmlDevice_t device;
    unsigned int sessionCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlEncoderSessionInfo_t *sessionInfos = (nvmlEncoderSessionInfo_t *)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));

    nvmlReturn_t result = nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, sessionInfos, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDecoderUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned int utilization;
    unsigned int samplingPeriodUs;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFBCStats(void *conn)
{
    nvmlDevice_t device;
    nvmlFBCStats_t fbcStats;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

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
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t sessionInfo;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFBCSessions(device, &sessionCount, &sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDriverModel(void *conn)
{
    nvmlDevice_t device;
    nvmlDriverModel_t current;
    nvmlDriverModel_t pending;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(conn, &pending, sizeof(nvmlDriverModel_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDriverModel(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(conn, &pending, sizeof(nvmlDriverModel_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVbiosVersion(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *version = (char *)malloc(length * sizeof(char));

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
    nvmlBridgeChipHierarchy_t bridgeHierarchy;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0)
        return -1;

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
    unsigned int infoCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    unsigned int infoCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    nvmlReturn_t result = nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    unsigned int infoCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    nvmlReturn_t result = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceOnSameBoard(void *conn)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    int onSameBoard;

    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &onSameBoard, sizeof(int)) < 0)
        return -1;

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
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_read(conn, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    nvmlSamplingType_t type;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t sampleValType;
    unsigned int sampleCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &type, sizeof(nvmlSamplingType_t)) < 0 ||
        rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(conn, &sampleCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlSample_t *samples = (nvmlSample_t *)malloc(sampleCount * sizeof(nvmlSample_t));

    nvmlReturn_t result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(conn, &sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, samples, sampleCount * sizeof(nvmlSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlBAR1Memory_t bar1Memory;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0)
        return -1;

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
    nvmlPerfPolicyType_t perfPolicyType;
    nvmlViolationTime_t violTime;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        rpc_read(conn, &violTime, sizeof(nvmlViolationTime_t)) < 0)
        return -1;

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
    unsigned int irqNum;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &irqNum, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int numCores;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &numCores, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlPowerSource_t powerSource;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &powerSource, sizeof(nvmlPowerSource_t)) < 0)
        return -1;

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
    unsigned int busWidth;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &busWidth, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int maxSpeed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int pcieSpeed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pcieSpeed, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int adaptiveClockStatus;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &adaptiveClockStatus, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlEnableState_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int pid;
    nvmlAccountingStats_t stats;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pid, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

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
    unsigned int count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned int *pids = (unsigned int *)malloc(count * sizeof(unsigned int));

    nvmlReturn_t result = nvmlDeviceGetAccountingPids(device, &count, pids);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, pids, count * sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingBufferSize(void *conn)
{
    nvmlDevice_t device;
    unsigned int bufferSize;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_read(conn, &pageCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned long long *addresses = (unsigned long long *)malloc(pageCount * sizeof(unsigned long long));

    nvmlReturn_t result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, addresses);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, addresses, pageCount * sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPages_v2(void *conn)
{
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;
    unsigned long long timestamps;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_read(conn, &pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &timestamps, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned long long *addresses = (unsigned long long *)malloc(pageCount * sizeof(unsigned long long));

    nvmlReturn_t result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, addresses, &timestamps);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, addresses, pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, &timestamps, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t isPending;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &isPending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int corrRows;
    unsigned int uncRows;
    unsigned int isPending;
    unsigned int failureOccurred;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &corrRows, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &uncRows, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &isPending, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &failureOccurred, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &corrRows, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &uncRows, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &isPending, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &failureOccurred, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRowRemapperHistogram(void *conn)
{
    nvmlDevice_t device;
    nvmlRowRemapperHistogramValues_t values;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0)
        return -1;

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
    nvmlDeviceArchitecture_t arch;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0)
        return -1;

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
    nvmlLedColor_t color;

    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_read(conn, &color, sizeof(nvmlLedColor_t)) < 0)
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
    nvmlEnableState_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
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
    nvmlComputeMode_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlComputeMode_t)) < 0)
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
    nvmlEnableState_t ecc;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &ecc, sizeof(nvmlEnableState_t)) < 0)
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
    nvmlEccCounterType_t counterType;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
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
    nvmlDriverModel_t driverModel;
    unsigned int flags;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &driverModel, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
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
    unsigned int minGpuClockMHz;
    unsigned int maxGpuClockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &maxGpuClockMHz, sizeof(unsigned int)) < 0)
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
    unsigned int minMemClockMHz;
    unsigned int maxMemClockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &maxMemClockMHz, sizeof(unsigned int)) < 0)
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
    unsigned int memClockMHz;
    unsigned int graphicsClockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &memClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &graphicsClockMHz, sizeof(unsigned int)) < 0)
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
    nvmlClkMonStatus_t status;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &status, sizeof(nvmlClkMonStatus_t)) < 0)
        return -1;

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
    unsigned int limit;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &limit, sizeof(unsigned int)) < 0)
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
    nvmlGpuOperationMode_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlGpuOperationMode_t)) < 0)
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
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_read(conn, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
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
    nvmlEnableState_t mode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
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
    unsigned int link;
    nvmlEnableState_t isActive;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int link;
    unsigned int version;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &version, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int link;
    nvmlNvLinkCapability_t capability;
    unsigned int capResult;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        rpc_read(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int link;
    nvmlPciInfo_t pci;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

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
    unsigned int link;
    nvmlNvLinkErrorCounter_t counter;
    unsigned long long counterValue;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        rpc_read(conn, &counterValue, sizeof(unsigned long long)) < 0)
        return -1;

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
    unsigned int link;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0)
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
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t control;
    unsigned int reset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &counter, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_read(conn, &reset, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, &control, reset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkUtilizationControl(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t control;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &counter, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;

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
    unsigned int link;
    unsigned int counter;
    unsigned long long rxcounter;
    unsigned long long txcounter;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &counter, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &txcounter, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, &rxcounter, &txcounter);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, &txcounter, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceFreezeNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlEnableState_t freeze;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &counter, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &freeze, sizeof(nvmlEnableState_t)) < 0)
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
    unsigned int link;
    unsigned int counter;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
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
    unsigned int link;
    nvmlIntNvLinkDeviceType_t pNvLinkDeviceType;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &link, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0)
        return -1;

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

    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

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
    unsigned long long eventTypes;
    nvmlEventSet_t set;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
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
    unsigned long long eventTypes;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &eventTypes, sizeof(unsigned long long)) < 0)
        return -1;

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
    nvmlEventData_t data;
    unsigned int timeoutms;

    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_read(conn, &data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_read(conn, &timeoutms, sizeof(unsigned int)) < 0)
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
    nvmlEnableState_t newState;

    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_read(conn, &newState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceModifyDrainState(&pciInfo, newState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceQueryDrainState(void *conn)
{
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t currentState;

    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_read(conn, &currentState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceQueryDrainState(&pciInfo, &currentState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(conn, &currentState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceRemoveGpu_v2(void *conn)
{
    nvmlPciInfo_t pciInfo;
    nvmlDetachGpuState_t gpuState;
    nvmlPcieLinkState_t linkState;

    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_read(conn, &gpuState, sizeof(nvmlDetachGpuState_t)) < 0 ||
        rpc_read(conn, &linkState, sizeof(nvmlPcieLinkState_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceRemoveGpu_v2(&pciInfo, gpuState, linkState);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
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

    if (rpc_write(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFieldValues(void *conn)
{
    nvmlDevice_t device;
    int valuesCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &valuesCount, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlFieldValue_t *values = (nvmlFieldValue_t *)malloc(valuesCount * sizeof(nvmlFieldValue_t));

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
    int valuesCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &valuesCount, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlFieldValue_t *values = (nvmlFieldValue_t *)malloc(valuesCount * sizeof(nvmlFieldValue_t));

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
    nvmlGpuVirtualizationMode_t pVirtualMode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        return -1;

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
    nvmlHostVgpuMode_t pHostVgpuMode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0)
        return -1;

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
    nvmlGpuVirtualizationMode_t virtualMode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
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
    nvmlGridLicensableFeatures_t pGridLicensableFeatures;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0)
        return -1;

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
    nvmlProcessUtilizationSample_t utilization;
    unsigned int processSamplesCount;
    unsigned long long lastSeenTimeStamp;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_read(conn, &processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetProcessUtilization(device, &utilization, &processSamplesCount, lastSeenTimeStamp);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_write(conn, &processSamplesCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGspFirmwareVersion(void *conn)
{
    nvmlDevice_t device;
    char version;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &version, sizeof(char)) < 0)
        return -1;

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
    unsigned int isEnabled;
    unsigned int defaultMode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &defaultMode, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &defaultMode, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuDriverCapabilities(void *conn)
{
    nvmlVgpuDriverCapability_t capability;
    unsigned int capResult;

    if (rpc_read(conn, &capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        rpc_read(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlDeviceVgpuCapability_t capability;
    unsigned int capResult;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        rpc_read(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int vgpuCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlVgpuTypeId_t *vgpuTypeIds = (nvmlVgpuTypeId_t *)malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    nvmlReturn_t result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCreatableVgpus(void *conn)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlVgpuTypeId_t *vgpuTypeIds = (nvmlVgpuTypeId_t *)malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    nvmlReturn_t result = nvmlDeviceGetCreatableVgpus(device, &vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetClass(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    char vgpuTypeClass;
    unsigned int size;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &vgpuTypeClass, sizeof(char)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetClass(vgpuTypeId, &vgpuTypeClass, &size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuTypeClass, sizeof(char)) < 0 ||
        rpc_write(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetName(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *vgpuTypeName = (char *)malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, &size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, vgpuTypeName, size * sizeof(char)) < 0 ||
        rpc_write(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int gpuInstanceProfileId;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &gpuInstanceProfileId, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned long long deviceID;
    unsigned long long subsystemID;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &subsystemID, sizeof(unsigned long long)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, &deviceID, &subsystemID);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, &subsystemID, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetFramebufferSize(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long fbSize;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &fbSize, sizeof(unsigned long long)) < 0)
        return -1;

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
    unsigned int numDisplayHeads;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &numDisplayHeads, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int displayIndex;
    unsigned int xdim;
    unsigned int ydim;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &displayIndex, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &xdim, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &ydim, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, &xdim, &ydim);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &xdim, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &ydim, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetLicense(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *vgpuTypeLicenseString = (char *)malloc(size * sizeof(char));

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
    unsigned int frameRateLimit;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int vgpuInstanceCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &vgpuInstanceCount, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int vgpuInstanceCountPerVm;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int vgpuCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlVgpuInstance_t *vgpuInstances = (nvmlVgpuInstance_t *)malloc(vgpuCount * sizeof(nvmlVgpuInstance_t));

    nvmlReturn_t result = nvmlDeviceGetActiveVgpus(device, &vgpuCount, vgpuInstances);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuInstances, vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetVmID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    nvmlVgpuVmIdType_t vmIdType;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *vmId = (char *)malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, &vmIdType);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, vmId, size * sizeof(char)) < 0 ||
        rpc_write(conn, &vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetUUID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *uuid = (char *)malloc(size * sizeof(char));

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
    unsigned int length;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *version = (char *)malloc(length * sizeof(char));

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
    unsigned long long fbUsage;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &fbUsage, sizeof(unsigned long long)) < 0)
        return -1;

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
    unsigned int licensed;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &licensed, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlVgpuTypeId_t vgpuTypeId;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

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
    unsigned int frameRateLimit;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlEnableState_t eccMode;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &eccMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int encoderCapacity;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int encoderCapacity;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
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
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, &sessionCount, &averageFps, &averageLatency);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderSessions(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    nvmlEncoderSessionInfo_t sessionInfo;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &sessionCount, &sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFBCStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlFBCStats_t fbcStats;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

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
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t sessionInfo;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, &sessionCount, &sessionInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetGpuInstanceId(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int gpuInstanceId;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &gpuInstanceId, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int length;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *vgpuPciId = (char *)malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, &length);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, vgpuPciId, length * sizeof(char)) < 0 ||
        rpc_write(conn, &length, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetCapabilities(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    nvmlVgpuCapability_t capability;
    unsigned int capResult;

    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_read(conn, &capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        rpc_read(conn, &capResult, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlVgpuMetadata_t vgpuMetadata;
    unsigned int bufferSize;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetMetadata(vgpuInstance, &vgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuMetadata(void *conn)
{
    nvmlDevice_t device;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    unsigned int bufferSize;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuMetadata(device, &pgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuCompatibility(void *conn)
{
    nvmlVgpuMetadata_t vgpuMetadata;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    nvmlVgpuPgpuCompatibility_t compatibilityInfo;

    if (rpc_read(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(conn, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(conn, &compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &compatibilityInfo);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(conn, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(conn, &compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPgpuMetadataString(void *conn)
{
    nvmlDevice_t device;
    unsigned int bufferSize;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    char *pgpuMetadata = (char *)malloc(bufferSize * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, pgpuMetadata, bufferSize * sizeof(char)) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerLog(void *conn)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerLog_t pSchedulerLog;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0)
        return -1;

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
    nvmlVgpuSchedulerGetState_t pSchedulerState;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0)
        return -1;

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
    nvmlVgpuSchedulerCapabilities_t pCapabilities;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0)
        return -1;

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

    if (rpc_read(conn, &supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_read(conn, &current, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuVersion(&supported, &current);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_write(conn, &current, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSetVgpuVersion(void *conn)
{
    nvmlVgpuVersion_t vgpuVersion;

    if (rpc_read(conn, &vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

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
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t sampleValType;
    unsigned int vgpuInstanceSamplesCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(conn, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlVgpuInstanceUtilizationSample_t *utilizationSamples = (nvmlVgpuInstanceUtilizationSample_t *)malloc(vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t));

    nvmlReturn_t result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, &sampleValType, &vgpuInstanceSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(conn, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, utilizationSamples, vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuProcessUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    unsigned int vgpuProcessSamplesCount;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_read(conn, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlVgpuProcessUtilizationSample_t *utilizationSamples = (nvmlVgpuProcessUtilizationSample_t *)malloc(vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t));

    nvmlReturn_t result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, &vgpuProcessSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, utilizationSamples, vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingMode(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t mode;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

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
    unsigned int count;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    unsigned int *pids = (unsigned int *)malloc(count * sizeof(unsigned int));

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, &count, pids);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, pids, count * sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int pid;
    nvmlAccountingStats_t stats;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &pid, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

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
    nvmlVgpuLicenseInfo_t licenseInfo;

    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_read(conn, &licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0)
        return -1;

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

    if (rpc_read(conn, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

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
    nvmlExcludedDeviceInfo_t info;

    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlExcludedDeviceInfo_t)) < 0)
        return -1;

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
    unsigned int mode;
    nvmlReturn_t activationStatus;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &mode, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &activationStatus, sizeof(nvmlReturn_t)) < 0)
        return -1;

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
    unsigned int currentMode;
    unsigned int pendingMode;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &currentMode, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &pendingMode, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &currentMode, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &pendingMode, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfo(void *conn)
{
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_t info;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profile, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0)
        return -1;

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
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_v2_t info;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profile, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0)
        return -1;

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
    unsigned int profileId;
    unsigned int count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlGpuInstancePlacement_t *placements = (nvmlGpuInstancePlacement_t *)malloc(count * sizeof(nvmlGpuInstancePlacement_t));

    nvmlReturn_t result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, placements, count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceRemainingCapacity(void *conn)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int profileId;
    nvmlGpuInstance_t gpuInstance;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

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
    unsigned int profileId;
    nvmlGpuInstancePlacement_t placement;
    nvmlGpuInstance_t gpuInstance;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &placement, sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, &placement, &gpuInstance);

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
    unsigned int profileId;
    unsigned int count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlGpuInstance_t *gpuInstances = (nvmlGpuInstance_t *)malloc(count * sizeof(nvmlGpuInstance_t));

    nvmlReturn_t result = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, gpuInstances, count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceById(void *conn)
{
    nvmlDevice_t device;
    unsigned int id;
    nvmlGpuInstance_t gpuInstance;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &id, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

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
    nvmlGpuInstanceInfo_t info;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlGpuInstanceInfo_t)) < 0)
        return -1;

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
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_t info;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profile, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &engProfile, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0)
        return -1;

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
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_v2_t info;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profile, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &engProfile, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0)
        return -1;

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
    unsigned int profileId;
    unsigned int count;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int profileId;
    nvmlComputeInstancePlacement_t placements;
    unsigned int count;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, &placements, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceCreateComputeInstance(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    nvmlComputeInstance_t computeInstance;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

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
    unsigned int profileId;
    nvmlComputeInstancePlacement_t placement;
    nvmlComputeInstance_t computeInstance;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &placement, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, &placement, &computeInstance);

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
    unsigned int profileId;
    unsigned int count;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &profileId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlComputeInstance_t *computeInstances = (nvmlComputeInstance_t *)malloc(count * sizeof(nvmlComputeInstance_t));

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, &count);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, computeInstances, count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceById(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int id;
    nvmlComputeInstance_t computeInstance;

    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_read(conn, &id, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

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
    nvmlComputeInstanceInfo_t info;

    if (rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlComputeInstanceInfo_t)) < 0)
        return -1;

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
    unsigned int isMigDevice;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &isMigDevice, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int id;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &id, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int id;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &id, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int count;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        return -1;

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
    unsigned int index;
    nvmlDevice_t migDevice;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &index, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &migDevice, sizeof(nvmlDevice_t)) < 0)
        return -1;

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
    nvmlDevice_t device;

    if (rpc_read(conn, &migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

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
    nvmlBusType_t type;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &type, sizeof(nvmlBusType_t)) < 0)
        return -1;

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
    nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
        return -1;

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
    unsigned int fan;
    unsigned int speed;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &fan, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &speed, sizeof(unsigned int)) < 0)
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
    int offset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;

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
    int offset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &offset, sizeof(int)) < 0)
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
    int offset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &offset, sizeof(int)) < 0)
        return -1;

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
    int offset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &offset, sizeof(int)) < 0)
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
    nvmlClockType_t type;
    nvmlPstates_t pstate;
    unsigned int minClockMHz;
    unsigned int maxClockMHz;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_read(conn, &pstate, sizeof(nvmlPstates_t)) < 0 ||
        rpc_read(conn, &minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &maxClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &maxClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(void *conn)
{
    nvmlDevice_t device;
    nvmlPstates_t pstates;
    unsigned int size;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &pstates, sizeof(nvmlPstates_t)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedPerformanceStates(device, &pstates, size);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &pstates, sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(void *conn)
{
    nvmlDevice_t device;
    int minOffset;
    int maxOffset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minOffset, sizeof(int)) < 0 ||
        rpc_read(conn, &maxOffset, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &minOffset, sizeof(int)) < 0 ||
        rpc_write(conn, &maxOffset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemClkMinMaxVfOffset(void *conn)
{
    nvmlDevice_t device;
    int minOffset;
    int maxOffset;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &minOffset, sizeof(int)) < 0 ||
        rpc_read(conn, &maxOffset, sizeof(int)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    if (rpc_write(conn, &minOffset, sizeof(int)) < 0 ||
        rpc_write(conn, &maxOffset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuFabricInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuFabricInfo_t gpuFabricInfo;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0)
        return -1;

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

    if (rpc_read(conn, &metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0)
        return -1;

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

    if (rpc_read(conn, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleFree(gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmSampleAlloc(void *conn)
{
    nvmlGpmSample_t gpmSample;

    if (rpc_read(conn, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

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
    nvmlGpmSample_t gpmSample;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleGet(device, gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmMigSampleGet(void *conn)
{
    nvmlDevice_t device;
    unsigned int gpuInstanceId;
    nvmlGpmSample_t gpmSample;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_read(conn, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample);

    if (rpc_start_response(conn, request_id) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmQueryDeviceSupport(void *conn)
{
    nvmlDevice_t device;
    nvmlGpmSupport_t gpmSupport;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &gpmSupport, sizeof(nvmlGpmSupport_t)) < 0)
        return -1;

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
    nvmlNvLinkPowerThres_t info;

    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_read(conn, &info, sizeof(nvmlNvLinkPowerThres_t)) < 0)
        return -1;

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

int handle_cuGetProcAddress(void *conn) {
    int symbol_length;
    char symbol[256];
    void *func = nullptr;
    int cudaVersion;
    cuuint64_t flags;
    CUdriverProcAddressQueryResult result;

    // Read inputs from the client
    if (rpc_read(conn, &symbol_length, sizeof(int)) < 0 ||
        rpc_read(conn, symbol, symbol_length) < 0 ||
        rpc_read(conn, &cudaVersion, sizeof(int)) < 0 ||
        rpc_read(conn, &flags, sizeof(cuuint64_t)) < 0)
        return -1;

    // End request to process the function
    int request_id = rpc_end_request(conn);
    if (request_id < 0) return -1;

    // Call cuGetProcAddress to get the function pointer and status
    CUresult result_code = cuGetProcAddress(symbol, &func, cudaVersion, flags, &result);

    // Start response and send the result code, function pointer, and status back to the client
    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result_code, sizeof(CUresult)) < 0 || // Send result_code first
        rpc_write(conn, &func, sizeof(void *)) < 0 || // Send the function pointer
        rpc_write(conn, &result, sizeof(CUdriverProcAddressQueryResult)) < 0) // Send the query result status
        return -1;

    return result_code;
}

int handle_cuGetExportTable(void *conn) {
    CUuuid table_uuid;
    const void *export_table = nullptr;

    if (rpc_read(conn, &table_uuid, sizeof(CUuuid)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0) return -1;

    CUresult result = cuGetExportTable(&export_table, &table_uuid);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(CUresult)) < 0 ||
        rpc_write(conn, &export_table, sizeof(void *)) < 0)
        return -1;

    return result;
}

int handle_cudaGetDeviceCount(void *conn) {
    int deviceCount = 0;

    cudaError_t result = cudaGetDeviceCount(&deviceCount);

    int request_id = rpc_end_request(conn);
    if (request_id < 0) return -1;

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(cudaError_t)) < 0 ||
        rpc_write(conn, &deviceCount, sizeof(int)) < 0)
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
    handle_cuGetProcAddress,
    handle_cudaGetDeviceCount,
};

RequestHandler get_handler(const int op)
{
    return opHandlers[op];
}
