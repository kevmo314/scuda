#include <nvml.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"

#include "manual_server.h"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_response(const void *conn, void *return_value);

int handle_nvmlInit_v2(void *conn)
{
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlInit_v2();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlInitWithFlags(void *conn)
{
    unsigned int flags;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlInitWithFlags(flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlShutdown(void *conn)
{
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlShutdown();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetDriverVersion(void *conn)
{
    unsigned int length;
    char* version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    version = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlSystemGetDriverVersion(version, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, version, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) version);
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetNVMLVersion(void *conn)
{
    unsigned int length;
    char* version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    version = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlSystemGetNVMLVersion(version, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, version, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) version);
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetCudaDriverVersion(void *conn)
{
    int cudaDriverVersion;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlSystemGetCudaDriverVersion(&cudaDriverVersion);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetCudaDriverVersion_v2(void *conn)
{
    int cudaDriverVersion;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlSystemGetCudaDriverVersion_v2(&cudaDriverVersion);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetProcessName(void *conn)
{
    unsigned int pid;
    unsigned int length;
    char* name;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    name = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlSystemGetProcessName(pid, name, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, name, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetCount(void *conn)
{
    unsigned int unitCount;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetCount(&unitCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &unitCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetHandleByIndex(void *conn)
{
    unsigned int index;
    nvmlUnit_t unit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetHandleByIndex(index, &unit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetUnitInfo(void *conn)
{
    nvmlUnit_t unit;
    nvmlUnitInfo_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetUnitInfo(unit, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetLedState(void *conn)
{
    nvmlUnit_t unit;
    nvmlLedState_t state;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetLedState(unit, &state);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetPsuInfo(void *conn)
{
    nvmlUnit_t unit;
    nvmlPSUInfo_t psu;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetPsuInfo(unit, &psu);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetTemperature(void *conn)
{
    nvmlUnit_t unit;
    unsigned int type;
    unsigned int temp;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetTemperature(unit, type, &temp);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &temp, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetFanSpeedInfo(void *conn)
{
    nvmlUnit_t unit;
    nvmlUnitFanSpeeds_t fanSpeeds;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitGetDevices(void *conn)
{
    nvmlUnit_t unit;
    unsigned int deviceCount;
    nvmlDevice_t* devices;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &deviceCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    devices = (nvmlDevice_t*)malloc(deviceCount * sizeof(nvmlDevice_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlUnitGetDevices(unit, &deviceCount, devices);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, devices, deviceCount * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) devices);
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetHicVersion(void *conn)
{
    unsigned int hwbcCount;
    nvmlHwbcEntry_t* hwbcEntries;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &hwbcCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    hwbcEntries = (nvmlHwbcEntry_t*)malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlSystemGetHicVersion(&hwbcCount, hwbcEntries);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, hwbcEntries, hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) hwbcEntries);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCount_v2(void *conn)
{
    unsigned int deviceCount;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetCount_v2(&deviceCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAttributes_v2(void *conn)
{
    nvmlDevice_t device;
    nvmlDeviceAttributes_t attributes;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAttributes_v2(device, &attributes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetHandleByIndex_v2(void *conn)
{
    unsigned int index;
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetHandleByIndex_v2(index, &device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetHandleBySerial(void *conn)
{
    std::size_t serial_len;
    const char* serial;
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &serial_len, sizeof(serial_len)) < 0)
        goto ERROR_0;
    serial = (const char*)malloc(serial_len);
    if (rpc_read(conn, &serial, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetHandleBySerial(serial, &device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) serial);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetHandleByUUID(void *conn)
{
    std::size_t uuid_len;
    const char* uuid;
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &uuid_len, sizeof(uuid_len)) < 0)
        goto ERROR_0;
    uuid = (const char*)malloc(uuid_len);
    if (rpc_read(conn, &uuid, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetHandleByUUID(uuid, &device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) uuid);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(void *conn)
{
    std::size_t pciBusId_len;
    const char* pciBusId;
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &pciBusId_len, sizeof(pciBusId_len)) < 0)
        goto ERROR_0;
    pciBusId = (const char*)malloc(pciBusId_len);
    if (rpc_read(conn, &pciBusId, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pciBusId);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetName(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;
    char* name;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    name = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetName(device, name, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, name, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetBrand(void *conn)
{
    nvmlDevice_t device;
    nvmlBrandType_t type;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetBrand(device, &type);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetIndex(void *conn)
{
    nvmlDevice_t device;
    unsigned int index;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetIndex(device, &index);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &index, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSerial(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;
    char* serial;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    serial = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetSerial(device, serial, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, serial, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) serial);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemoryAffinity(void *conn)
{
    nvmlDevice_t device;
    unsigned int nodeSetSize;
    unsigned long* nodeSet;
    nvmlAffinityScope_t scope;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodeSetSize, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    nodeSet = (unsigned long*)malloc(nodeSetSize * sizeof(unsigned long));
    if (rpc_read(conn, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, nodeSet, nodeSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) nodeSet);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCpuAffinityWithinScope(void *conn)
{
    nvmlDevice_t device;
    unsigned int cpuSetSize;
    unsigned long* cpuSet;
    nvmlAffinityScope_t scope;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &cpuSetSize, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    cpuSet = (unsigned long*)malloc(cpuSetSize * sizeof(unsigned long));
    if (rpc_read(conn, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) cpuSet);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCpuAffinity(void *conn)
{
    nvmlDevice_t device;
    unsigned int cpuSetSize;
    unsigned long* cpuSet;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &cpuSetSize, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    cpuSet = (unsigned long*)malloc(cpuSetSize * sizeof(unsigned long));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) cpuSet);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetCpuAffinity(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetCpuAffinity(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceClearCpuAffinity(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceClearCpuAffinity(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTopologyCommonAncestor(void *conn)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuTopologyLevel_t pathInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &pathInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTopologyNearestGpus(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuTopologyLevel_t level;
    unsigned int count;
    nvmlDevice_t* deviceArray;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    deviceArray = (nvmlDevice_t*)malloc(count * sizeof(nvmlDevice_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, deviceArray, count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) deviceArray);
ERROR_0:
    return -1;
}

int handle_nvmlSystemGetTopologyGpuSet(void *conn)
{
    unsigned int cpuNumber;
    unsigned int count;
    nvmlDevice_t* deviceArray;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &cpuNumber, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    deviceArray = (nvmlDevice_t*)malloc(count * sizeof(nvmlDevice_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlSystemGetTopologyGpuSet(cpuNumber, &count, deviceArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, deviceArray, count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) deviceArray);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetP2PStatus(void *conn)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuP2PCapsIndex_t p2pIndex;
    nvmlGpuP2PStatus_t p2pStatus;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, &p2pStatus);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetUUID(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;
    char* uuid;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    uuid = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetUUID(device, uuid, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, uuid, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) uuid);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetMdevUUID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    char* mdevUuid;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    mdevUuid = (char*)malloc(size * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, mdevUuid, size * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) mdevUuid);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMinorNumber(void *conn)
{
    nvmlDevice_t device;
    unsigned int minorNumber;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMinorNumber(device, &minorNumber);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetBoardPartNumber(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;
    char* partNumber;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    partNumber = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, partNumber, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) partNumber);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetInforomVersion(void *conn)
{
    nvmlDevice_t device;
    nvmlInforomObject_t object;
    unsigned int length;
    char* version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &object, sizeof(nvmlInforomObject_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    version = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetInforomVersion(device, object, version, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, version, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) version);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetInforomImageVersion(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;
    char* version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    version = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetInforomImageVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, version, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) version);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetInforomConfigurationChecksum(void *conn)
{
    nvmlDevice_t device;
    unsigned int checksum;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetInforomConfigurationChecksum(device, &checksum);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &checksum, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceValidateInforom(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceValidateInforom(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDisplayMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t display;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDisplayMode(device, &display);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDisplayActive(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t isActive;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDisplayActive(device, &isActive);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPersistenceMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPersistenceMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPciInfo_v3(void *conn)
{
    nvmlDevice_t device;
    nvmlPciInfo_t pci;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPciInfo_v3(device, &pci);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(void *conn)
{
    nvmlDevice_t device;
    unsigned int maxLinkGen;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMaxPcieLinkGeneration(device, &maxLinkGen);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuMaxPcieLinkGeneration(void *conn)
{
    nvmlDevice_t device;
    unsigned int maxLinkGenDevice;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &maxLinkGenDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(void *conn)
{
    nvmlDevice_t device;
    unsigned int maxLinkWidth;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMaxPcieLinkWidth(device, &maxLinkWidth);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(void *conn)
{
    nvmlDevice_t device;
    unsigned int currLinkGen;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(void *conn)
{
    nvmlDevice_t device;
    unsigned int currLinkWidth;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPcieThroughput(void *conn)
{
    nvmlDevice_t device;
    nvmlPcieUtilCounter_t counter;
    unsigned int value;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPcieThroughput(device, counter, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPcieReplayCounter(void *conn)
{
    nvmlDevice_t device;
    unsigned int value;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPcieReplayCounter(device, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetClockInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int clock;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetClockInfo(device, type, &clock);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clock, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMaxClockInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int clock;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMaxClockInfo(device, type, &clock);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clock, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetApplicationsClock(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDefaultApplicationsClock(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceResetApplicationsClocks(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceResetApplicationsClocks(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetClock(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    nvmlClockId_t clockId;
    unsigned int clockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &clockId, sizeof(nvmlClockId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMaxCustomerBoostClock(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &clockType, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, &clockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSupportedMemoryClocks(void *conn)
{
    nvmlDevice_t device;
    unsigned int count;
    unsigned int* clocksMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    clocksMHz = (unsigned int*)malloc(count * sizeof(unsigned int));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetSupportedMemoryClocks(device, &count, clocksMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, clocksMHz, count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) clocksMHz);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(void *conn)
{
    nvmlDevice_t device;
    unsigned int memoryClockMHz;
    unsigned int count;
    unsigned int* clocksMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memoryClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    clocksMHz = (unsigned int*)malloc(count * sizeof(unsigned int));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, clocksMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, clocksMHz, count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) clocksMHz);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t isEnabled;
    nvmlEnableState_t defaultIsEnabled;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(conn, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t enabled;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &enabled, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t enabled;
    unsigned int flags;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &enabled, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetFanSpeed(void *conn)
{
    nvmlDevice_t device;
    unsigned int speed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetFanSpeed(device, &speed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &speed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetFanSpeed_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int speed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetFanSpeed_v2(device, fan, &speed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &speed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTargetFanSpeed(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int targetSpeed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetTargetFanSpeed(device, fan, &targetSpeed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetDefaultFanSpeed_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetDefaultFanSpeed_v2(device, fan);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMinMaxFanSpeed(void *conn)
{
    nvmlDevice_t device;
    unsigned int minSpeed;
    unsigned int maxSpeed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetFanControlPolicy_v2(device, fan, &policy);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetFanControlPolicy(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetFanControlPolicy(device, fan, policy);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNumFans(void *conn)
{
    nvmlDevice_t device;
    unsigned int numFans;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNumFans(device, &numFans);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &numFans, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTemperature(void *conn)
{
    nvmlDevice_t device;
    nvmlTemperatureSensors_t sensorType;
    unsigned int temp;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetTemperature(device, sensorType, &temp);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &temp, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTemperatureThreshold(void *conn)
{
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    unsigned int temp;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, &temp);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &temp, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetTemperatureThreshold(void *conn)
{
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    int temp;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &temp, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetTemperatureThreshold(device, thresholdType, &temp);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &temp, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetThermalSettings(void *conn)
{
    nvmlDevice_t device;
    unsigned int sensorIndex;
    nvmlGpuThermalSettings_t pThermalSettings;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sensorIndex, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetThermalSettings(device, sensorIndex, &pThermalSettings);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPerformanceState(void *conn)
{
    nvmlDevice_t device;
    nvmlPstates_t pState;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPerformanceState(device, &pState);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCurrentClocksThrottleReasons(void *conn)
{
    nvmlDevice_t device;
    unsigned long long clocksThrottleReasons;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clocksThrottleReasons);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSupportedClocksThrottleReasons(void *conn)
{
    nvmlDevice_t device;
    unsigned long long supportedClocksThrottleReasons;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetSupportedClocksThrottleReasons(device, &supportedClocksThrottleReasons);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerState(void *conn)
{
    nvmlDevice_t device;
    nvmlPstates_t pState;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerState(device, &pState);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerManagementMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerManagementMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerManagementLimit(void *conn)
{
    nvmlDevice_t device;
    unsigned int limit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerManagementLimit(device, &limit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &limit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerManagementLimitConstraints(void *conn)
{
    nvmlDevice_t device;
    unsigned int minLimit;
    unsigned int maxLimit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &minLimit, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(void *conn)
{
    nvmlDevice_t device;
    unsigned int defaultLimit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerUsage(void *conn)
{
    nvmlDevice_t device;
    unsigned int power;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerUsage(device, &power);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &power, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTotalEnergyConsumption(void *conn)
{
    nvmlDevice_t device;
    unsigned long long energy;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &energy, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetEnforcedPowerLimit(void *conn)
{
    nvmlDevice_t device;
    unsigned int limit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetEnforcedPowerLimit(device, &limit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &limit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuOperationMode(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuOperationMode_t current;
    nvmlGpuOperationMode_t pending;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuOperationMode(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_write(conn, &pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemoryInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlMemory_t memory;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMemoryInfo(device, &memory);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemoryInfo_v2(void *conn)
{
    nvmlDevice_t device;
    nvmlMemory_v2_t memory;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMemoryInfo_v2(device, &memory);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetComputeMode(void *conn)
{
    nvmlDevice_t device;
    nvmlComputeMode_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetComputeMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCudaComputeCapability(void *conn)
{
    nvmlDevice_t device;
    int major;
    int minor;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &major, sizeof(int)) < 0 ||
        rpc_write(conn, &minor, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetEccMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t current;
    nvmlEnableState_t pending;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetEccMode(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(conn, &pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDefaultEccMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t defaultMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDefaultEccMode(device, &defaultMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetBoardId(void *conn)
{
    nvmlDevice_t device;
    unsigned int boardId;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetBoardId(device, &boardId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &boardId, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMultiGpuBoard(void *conn)
{
    nvmlDevice_t device;
    unsigned int multiGpuBool;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMultiGpuBoard(device, &multiGpuBool);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetTotalEccErrors(void *conn)
{
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    unsigned long long eccCounts;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, &eccCounts);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDetailedEccErrors(void *conn)
{
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlEccErrorCounts_t eccCounts;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, &eccCounts);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemoryErrorCounter(void *conn)
{
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlMemoryLocation_t locationType;
    unsigned long long count;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &locationType, sizeof(nvmlMemoryLocation_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetUtilizationRates(void *conn)
{
    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetEncoderUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned int utilization;
    unsigned int samplingPeriodUs;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetEncoderUtilization(device, &utilization, &samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetEncoderCapacity(void *conn)
{
    nvmlDevice_t device;
    nvmlEncoderType_t encoderQueryType;
    unsigned int encoderCapacity;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, &encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetEncoderStats(void *conn)
{
    nvmlDevice_t device;
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetEncoderSessions(void *conn)
{
    nvmlDevice_t device;
    unsigned int sessionCount;
    nvmlEncoderSessionInfo_t* sessionInfos;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    sessionInfos = (nvmlEncoderSessionInfo_t*)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, sessionInfos, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) sessionInfos);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDecoderUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned int utilization;
    unsigned int samplingPeriodUs;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetFBCStats(void *conn)
{
    nvmlDevice_t device;
    nvmlFBCStats_t fbcStats;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetFBCStats(device, &fbcStats);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetFBCSessions(void *conn)
{
    nvmlDevice_t device;
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t* sessionInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    sessionInfo = (nvmlFBCSessionInfo_t*)malloc(sessionCount * sizeof(nvmlFBCSessionInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetFBCSessions(device, &sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, sessionInfo, sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) sessionInfo);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDriverModel(void *conn)
{
    nvmlDevice_t device;
    nvmlDriverModel_t current;
    nvmlDriverModel_t pending;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDriverModel(device, &current, &pending);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(conn, &pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVbiosVersion(void *conn)
{
    nvmlDevice_t device;
    unsigned int length;
    char* version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    version = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetVbiosVersion(device, version, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, version, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) version);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetBridgeChipInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlBridgeChipHierarchy_t bridgeHierarchy;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    unsigned int infoCount;
    nvmlProcessInfo_t* infos;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) infos);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    unsigned int infoCount;
    nvmlProcessInfo_t* infos;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) infos);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(void *conn)
{
    nvmlDevice_t device;
    unsigned int infoCount;
    nvmlProcessInfo_t* infos;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &infoCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    infos = (nvmlProcessInfo_t*)malloc(infoCount * sizeof(nvmlProcessInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, infos);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) infos);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceOnSameBoard(void *conn)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    int onSameBoard;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device1, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device2, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceOnSameBoard(device1, device2, &onSameBoard);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &onSameBoard, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAPIRestriction(void *conn)
{
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSamples(void *conn)
{
    nvmlDevice_t device;
    nvmlSamplingType_t type;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t sampleValType;
    unsigned int sampleCount;
    nvmlSample_t* samples;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(nvmlSamplingType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sampleCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    samples = (nvmlSample_t*)malloc(sampleCount * sizeof(nvmlSample_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(conn, &sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, samples, sampleCount * sizeof(nvmlSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) samples);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlBAR1Memory_t bar1Memory;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetViolationStatus(void *conn)
{
    nvmlDevice_t device;
    nvmlPerfPolicyType_t perfPolicyType;
    nvmlViolationTime_t violTime;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetViolationStatus(device, perfPolicyType, &violTime);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetIrqNum(void *conn)
{
    nvmlDevice_t device;
    unsigned int irqNum;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetIrqNum(device, &irqNum);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &irqNum, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNumGpuCores(void *conn)
{
    nvmlDevice_t device;
    unsigned int numCores;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNumGpuCores(device, &numCores);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &numCores, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPowerSource(void *conn)
{
    nvmlDevice_t device;
    nvmlPowerSource_t powerSource;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPowerSource(device, &powerSource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemoryBusWidth(void *conn)
{
    nvmlDevice_t device;
    unsigned int busWidth;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMemoryBusWidth(device, &busWidth);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &busWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPcieLinkMaxSpeed(void *conn)
{
    nvmlDevice_t device;
    unsigned int maxSpeed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPcieLinkMaxSpeed(device, &maxSpeed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPcieSpeed(void *conn)
{
    nvmlDevice_t device;
    unsigned int pcieSpeed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAdaptiveClockInfoStatus(void *conn)
{
    nvmlDevice_t device;
    unsigned int adaptiveClockStatus;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAccountingMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAccountingMode(device, &mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAccountingStats(void *conn)
{
    nvmlDevice_t device;
    unsigned int pid;
    nvmlAccountingStats_t stats;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAccountingStats(device, pid, &stats);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAccountingPids(void *conn)
{
    nvmlDevice_t device;
    unsigned int count;
    unsigned int* pids;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    pids = (unsigned int*)malloc(count * sizeof(unsigned int));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetAccountingPids(device, &count, pids);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, pids, count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pids);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetAccountingBufferSize(void *conn)
{
    nvmlDevice_t device;
    unsigned int bufferSize;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetAccountingBufferSize(device, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetRetiredPages(void *conn)
{
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;
    unsigned long long* addresses;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pageCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    addresses = (unsigned long long*)malloc(pageCount * sizeof(unsigned long long));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, addresses);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, addresses, pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) addresses);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetRetiredPages_v2(void *conn)
{
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;
    unsigned long long* addresses;
    unsigned long long* timestamps;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pageCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    addresses = (unsigned long long*)malloc(pageCount * sizeof(unsigned long long));
    timestamps = (unsigned long long*)malloc(pageCount * sizeof(unsigned long long));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_2;
    result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, addresses, timestamps);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, addresses, pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, timestamps, pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_2;

    return 0;
ERROR_2:
    free((void *) addresses);
ERROR_1:
    free((void *) timestamps);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t isPending;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetRetiredPagesPendingStatus(device, &isPending);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetRemappedRows(void *conn)
{
    nvmlDevice_t device;
    unsigned int corrRows;
    unsigned int uncRows;
    unsigned int isPending;
    unsigned int failureOccurred;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &corrRows, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &uncRows, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &isPending, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetRowRemapperHistogram(void *conn)
{
    nvmlDevice_t device;
    nvmlRowRemapperHistogramValues_t values;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetRowRemapperHistogram(device, &values);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetArchitecture(void *conn)
{
    nvmlDevice_t device;
    nvmlDeviceArchitecture_t arch;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetArchitecture(device, &arch);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlUnitSetLedState(void *conn)
{
    nvmlUnit_t unit;
    nvmlLedColor_t color;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &unit, sizeof(nvmlUnit_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &color, sizeof(nvmlLedColor_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlUnitSetLedState(unit, color);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetPersistenceMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetPersistenceMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetComputeMode(void *conn)
{
    nvmlDevice_t device;
    nvmlComputeMode_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(nvmlComputeMode_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetComputeMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetEccMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t ecc;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ecc, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetEccMode(device, ecc);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceClearEccErrorCounts(void *conn)
{
    nvmlDevice_t device;
    nvmlEccCounterType_t counterType;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceClearEccErrorCounts(device, counterType);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetDriverModel(void *conn)
{
    nvmlDevice_t device;
    nvmlDriverModel_t driverModel;
    unsigned int flags;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &driverModel, sizeof(nvmlDriverModel_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetDriverModel(device, driverModel, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetGpuLockedClocks(void *conn)
{
    nvmlDevice_t device;
    unsigned int minGpuClockMHz;
    unsigned int maxGpuClockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &minGpuClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &maxGpuClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceResetGpuLockedClocks(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceResetGpuLockedClocks(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetMemoryLockedClocks(void *conn)
{
    nvmlDevice_t device;
    unsigned int minMemClockMHz;
    unsigned int maxMemClockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &minMemClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &maxMemClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceResetMemoryLockedClocks(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceResetMemoryLockedClocks(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetApplicationsClocks(void *conn)
{
    nvmlDevice_t device;
    unsigned int memClockMHz;
    unsigned int graphicsClockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &graphicsClockMHz, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetClkMonStatus(void *conn)
{
    nvmlDevice_t device;
    nvmlClkMonStatus_t status;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetClkMonStatus(device, &status);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetPowerManagementLimit(void *conn)
{
    nvmlDevice_t device;
    unsigned int limit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &limit, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetPowerManagementLimit(device, limit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetGpuOperationMode(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuOperationMode_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(nvmlGpuOperationMode_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetGpuOperationMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetAPIRestriction(void *conn)
{
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetAPIRestriction(device, apiType, isRestricted);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetAccountingMode(void *conn)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetAccountingMode(device, mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceClearAccountingPids(void *conn)
{
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceClearAccountingPids(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkState(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlEnableState_t isActive;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkState(device, link, &isActive);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkVersion(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkVersion(device, link, &version);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &version, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkCapability(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlNvLinkCapability_t capability;
    unsigned int capResult;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &capability, sizeof(nvmlNvLinkCapability_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkCapability(device, link, capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkRemotePciInfo_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlPciInfo_t pci;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, &pci);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkErrorCounter(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlNvLinkErrorCounter_t counter;
    unsigned long long counterValue;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, &counterValue);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceResetNvLinkErrorCounters(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceResetNvLinkErrorCounters(device, link);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetNvLinkUtilizationControl(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t control;
    unsigned int reset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &reset, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, &control, reset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkUtilizationControl(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t control;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, &control);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    unsigned long long rxcounter;
    unsigned long long txcounter;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, &rxcounter, &txcounter);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, &txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceFreezeNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlEnableState_t freeze;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &freeze, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceResetNvLinkUtilizationCounter(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &counter, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetNvLinkRemoteDeviceType(void *conn)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlIntNvLinkDeviceType_t pNvLinkDeviceType;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &link, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, &pNvLinkDeviceType);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlEventSetCreate(void *conn)
{
    nvmlEventSet_t set;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlEventSetCreate(&set);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceRegisterEvents(void *conn)
{
    nvmlDevice_t device;
    unsigned long long eventTypes;
    nvmlEventSet_t set;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &eventTypes, sizeof(unsigned long long)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceRegisterEvents(device, eventTypes, set);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSupportedEventTypes(void *conn)
{
    nvmlDevice_t device;
    unsigned long long eventTypes;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetSupportedEventTypes(device, &eventTypes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlEventSetWait_v2(void *conn)
{
    nvmlEventSet_t set;
    nvmlEventData_t data;
    unsigned int timeoutms;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &timeoutms, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlEventSetWait_v2(set, &data, timeoutms);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlEventSetFree(void *conn)
{
    nvmlEventSet_t set;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &set, sizeof(nvmlEventSet_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlEventSetFree(set);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceModifyDrainState(void *conn)
{
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t newState;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &newState, sizeof(nvmlEnableState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceModifyDrainState(&pciInfo, newState);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceQueryDrainState(void *conn)
{
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t currentState;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceQueryDrainState(&pciInfo, &currentState);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceRemoveGpu_v2(void *conn)
{
    nvmlPciInfo_t pciInfo;
    nvmlDetachGpuState_t gpuState;
    nvmlPcieLinkState_t linkState;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gpuState, sizeof(nvmlDetachGpuState_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &linkState, sizeof(nvmlPcieLinkState_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceRemoveGpu_v2(&pciInfo, gpuState, linkState);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceDiscoverGpus(void *conn)
{
    nvmlPciInfo_t pciInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceDiscoverGpus(&pciInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetFieldValues(void *conn)
{
    nvmlDevice_t device;
    int valuesCount;
    nvmlFieldValue_t* values;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &valuesCount, sizeof(int)) < 0)
        goto ERROR_0;
    values = (nvmlFieldValue_t*)malloc(valuesCount * sizeof(nvmlFieldValue_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetFieldValues(device, valuesCount, values);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) values);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceClearFieldValues(void *conn)
{
    nvmlDevice_t device;
    int valuesCount;
    nvmlFieldValue_t* values;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &valuesCount, sizeof(int)) < 0)
        goto ERROR_0;
    values = (nvmlFieldValue_t*)malloc(valuesCount * sizeof(nvmlFieldValue_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceClearFieldValues(device, valuesCount, values);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) values);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVirtualizationMode(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuVirtualizationMode_t pVirtualMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetVirtualizationMode(device, &pVirtualMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetHostVgpuMode(void *conn)
{
    nvmlDevice_t device;
    nvmlHostVgpuMode_t pHostVgpuMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetHostVgpuMode(device, &pHostVgpuMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetVirtualizationMode(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuVirtualizationMode_t virtualMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetVirtualizationMode(device, virtualMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGridLicensableFeatures_v4(void *conn)
{
    nvmlDevice_t device;
    nvmlGridLicensableFeatures_t pGridLicensableFeatures;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGridLicensableFeatures_v4(device, &pGridLicensableFeatures);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetProcessUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned int processSamplesCount;
    nvmlProcessUtilizationSample_t* utilization;
    unsigned long long lastSeenTimeStamp;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &processSamplesCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    utilization = (nvmlProcessUtilizationSample_t*)malloc(processSamplesCount * sizeof(nvmlProcessUtilizationSample_t));
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetProcessUtilization(device, utilization, &processSamplesCount, lastSeenTimeStamp);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, utilization, processSamplesCount * sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) utilization);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGspFirmwareVersion(void *conn)
{
    nvmlDevice_t device;
    char version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGspFirmwareVersion(device, &version);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &version, sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGspFirmwareMode(void *conn)
{
    nvmlDevice_t device;
    unsigned int isEnabled;
    unsigned int defaultMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGetVgpuDriverCapabilities(void *conn)
{
    nvmlVgpuDriverCapability_t capability;
    unsigned int capResult;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &capability, sizeof(nvmlVgpuDriverCapability_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGetVgpuDriverCapabilities(capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuCapabilities(void *conn)
{
    nvmlDevice_t device;
    nvmlDeviceVgpuCapability_t capability;
    unsigned int capResult;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetVgpuCapabilities(device, capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSupportedVgpus(void *conn)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;
    nvmlVgpuTypeId_t* vgpuTypeIds;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuTypeIds = (nvmlVgpuTypeId_t*)malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuTypeIds);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetCreatableVgpus(void *conn)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;
    nvmlVgpuTypeId_t* vgpuTypeIds;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuTypeIds = (nvmlVgpuTypeId_t*)malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetCreatableVgpus(device, &vgpuCount, vgpuTypeIds);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuTypeIds);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetClass(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;
    char* vgpuTypeClass;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;
    vgpuTypeClass = (char*)malloc(size * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, &size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &size, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuTypeClass, size * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuTypeClass);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetName(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;
    char* vgpuTypeName;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuTypeName = (char*)malloc(size * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, &size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &size, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuTypeName, size * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuTypeName);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int gpuInstanceProfileId;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, &gpuInstanceProfileId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetDeviceID(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long deviceID;
    unsigned long long subsystemID;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, &deviceID, &subsystemID);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, &subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetFramebufferSize(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long fbSize;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, &fbSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetNumDisplayHeads(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int numDisplayHeads;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, &numDisplayHeads);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetResolution(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int displayIndex;
    unsigned int xdim;
    unsigned int ydim;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &displayIndex, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, &xdim, &ydim);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &xdim, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &ydim, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetLicense(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;
    char* vgpuTypeLicenseString;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuTypeLicenseString = (char*)malloc(size * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, vgpuTypeLicenseString, size * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuTypeLicenseString);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetFrameRateLimit(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int frameRateLimit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, &frameRateLimit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetMaxInstances(void *conn)
{
    nvmlDevice_t device;
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int vgpuInstanceCount;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, &vgpuInstanceCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetMaxInstancesPerVm(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int vgpuInstanceCountPerVm;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, &vgpuInstanceCountPerVm);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetActiveVgpus(void *conn)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;
    nvmlVgpuInstance_t* vgpuInstances;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &vgpuCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuInstances = (nvmlVgpuInstance_t*)malloc(vgpuCount * sizeof(nvmlVgpuInstance_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetActiveVgpus(device, &vgpuCount, vgpuInstances);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuInstances, vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuInstances);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetVmID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    char* vmId;
    nvmlVgpuVmIdType_t vmIdType;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vmId = (char*)malloc(size * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, &vmIdType);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, vmId, size * sizeof(char)) < 0 ||
        rpc_write(conn, &vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vmId);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetUUID(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    char* uuid;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    uuid = (char*)malloc(size * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, uuid, size * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) uuid);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetVmDriverVersion(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int length;
    char* version;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    version = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, version, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) version);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetFbUsage(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned long long fbUsage;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetFbUsage(vgpuInstance, &fbUsage);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetLicenseStatus(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int licensed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, &licensed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &licensed, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetType(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuTypeId_t vgpuTypeId;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetType(vgpuInstance, &vgpuTypeId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetFrameRateLimit(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int frameRateLimit;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, &frameRateLimit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetEccMode(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t eccMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetEccMode(vgpuInstance, &eccMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetEncoderCapacity(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int encoderCapacity;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, &encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceSetEncoderCapacity(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int encoderCapacity;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &encoderCapacity, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetEncoderStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, &sessionCount, &averageFps, &averageLatency);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetEncoderSessions(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    nvmlEncoderSessionInfo_t* sessionInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    sessionInfo = (nvmlEncoderSessionInfo_t*)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, sessionInfo, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) sessionInfo);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetFBCStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlFBCStats_t fbcStats;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetFBCStats(vgpuInstance, &fbcStats);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetFBCSessions(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t* sessionInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sessionCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    sessionInfo = (nvmlFBCSessionInfo_t*)malloc(sessionCount * sizeof(nvmlFBCSessionInfo_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, &sessionCount, sessionInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, sessionInfo, sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) sessionInfo);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetGpuInstanceId(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int gpuInstanceId;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, &gpuInstanceId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetGpuPciId(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int length;
    char* vgpuPciId;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuPciId = (char*)malloc(length * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, &length);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &length, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuPciId, length * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuPciId);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuTypeGetCapabilities(void *conn)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    nvmlVgpuCapability_t capability;
    unsigned int capResult;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &capability, sizeof(nvmlVgpuCapability_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, &capResult);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetMetadata(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int bufferSize;
    nvmlVgpuMetadata_t* vgpuMetadata;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    vgpuMetadata = (nvmlVgpuMetadata_t*)malloc(bufferSize * sizeof(nvmlVgpuMetadata_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, vgpuMetadata, bufferSize * sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) vgpuMetadata);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuMetadata(void *conn)
{
    nvmlDevice_t device;
    unsigned int bufferSize;
    nvmlVgpuPgpuMetadata_t* pgpuMetadata;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    pgpuMetadata = (nvmlVgpuPgpuMetadata_t*)malloc(bufferSize * sizeof(nvmlVgpuPgpuMetadata_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, pgpuMetadata, bufferSize * sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pgpuMetadata);
ERROR_0:
    return -1;
}

int handle_nvmlGetVgpuCompatibility(void *conn)
{
    nvmlVgpuMetadata_t vgpuMetadata;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    nvmlVgpuPgpuCompatibility_t compatibilityInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &compatibilityInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(conn, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(conn, &compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetPgpuMetadataString(void *conn)
{
    nvmlDevice_t device;
    unsigned int bufferSize;
    char* pgpuMetadata;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bufferSize, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    pgpuMetadata = (char*)malloc(bufferSize * sizeof(char));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, &bufferSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, pgpuMetadata, bufferSize * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pgpuMetadata);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuSchedulerLog(void *conn)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerLog_t pSchedulerLog;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetVgpuSchedulerLog(device, &pSchedulerLog);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuSchedulerState(void *conn)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerGetState_t pSchedulerState;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetVgpuSchedulerState(device, &pSchedulerState);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuSchedulerCapabilities(void *conn)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerCapabilities_t pCapabilities;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetVgpuSchedulerCapabilities(device, &pCapabilities);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGetVgpuVersion(void *conn)
{
    nvmlVgpuVersion_t supported;
    nvmlVgpuVersion_t current;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGetVgpuVersion(&supported, &current);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_write(conn, &current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlSetVgpuVersion(void *conn)
{
    nvmlVgpuVersion_t vgpuVersion;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlSetVgpuVersion(&vgpuVersion);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t sampleValType;
    unsigned int vgpuInstanceSamplesCount;
    nvmlVgpuInstanceUtilizationSample_t* utilizationSamples;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    utilizationSamples = (nvmlVgpuInstanceUtilizationSample_t*)malloc(vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, &sampleValType, &vgpuInstanceSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(conn, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, utilizationSamples, vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) utilizationSamples);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetVgpuProcessUtilization(void *conn)
{
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    unsigned int vgpuProcessSamplesCount;
    nvmlVgpuProcessUtilizationSample_t* utilizationSamples;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    utilizationSamples = (nvmlVgpuProcessUtilizationSample_t*)malloc(vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, &vgpuProcessSamplesCount, utilizationSamples);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, utilizationSamples, vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) utilizationSamples);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetAccountingMode(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t mode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, &mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetAccountingPids(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int count;
    unsigned int* pids;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    pids = (unsigned int*)malloc(count * sizeof(unsigned int));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, &count, pids);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, pids, count * sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pids);
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetAccountingStats(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int pid;
    nvmlAccountingStats_t stats;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pid, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, &stats);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceClearAccountingPids(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceClearAccountingPids(vgpuInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlVgpuInstanceGetLicenseInfo_v2(void *conn)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuLicenseInfo_t licenseInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, &licenseInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGetExcludedDeviceCount(void *conn)
{
    unsigned int deviceCount;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGetExcludedDeviceCount(&deviceCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGetExcludedDeviceInfoByIndex(void *conn)
{
    unsigned int index;
    nvmlExcludedDeviceInfo_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGetExcludedDeviceInfoByIndex(index, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetMigMode(void *conn)
{
    nvmlDevice_t device;
    unsigned int mode;
    nvmlReturn_t activationStatus;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetMigMode(device, mode, &activationStatus);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMigMode(void *conn)
{
    nvmlDevice_t device;
    unsigned int currentMode;
    unsigned int pendingMode;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &currentMode, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfo(void *conn)
{
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfoV(void *conn)
{
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_v2_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstancePossiblePlacements_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;
    nvmlGpuInstancePlacement_t* placements;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    placements = (nvmlGpuInstancePlacement_t*)malloc(count * sizeof(nvmlGpuInstancePlacement_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, placements, count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) placements);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstanceRemainingCapacity(void *conn)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceCreateGpuInstance(void *conn)
{
    nvmlDevice_t device;
    unsigned int profileId;
    nvmlGpuInstance_t gpuInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceCreateGpuInstance(device, profileId, &gpuInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceDestroy(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceDestroy(gpuInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstances(void *conn)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;
    nvmlGpuInstance_t* gpuInstances;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    gpuInstances = (nvmlGpuInstance_t*)malloc(count * sizeof(nvmlGpuInstance_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, gpuInstances, count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) gpuInstances);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstanceById(void *conn)
{
    nvmlDevice_t device;
    unsigned int id;
    nvmlGpuInstance_t gpuInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &id, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuInstanceById(device, id, &gpuInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetInfo(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    nvmlGpuInstanceInfo_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceGetInfo(gpuInstance, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfo(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &engProfile, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_v2_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profile, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &engProfile, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile, engProfile, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int count;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int count;
    nvmlComputeInstancePlacement_t* placements;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    placements = (nvmlComputeInstancePlacement_t*)malloc(count * sizeof(nvmlComputeInstancePlacement_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placements, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, placements, count * sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) placements);
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceCreateComputeInstance(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    nvmlComputeInstance_t computeInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, &computeInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlComputeInstanceDestroy(void *conn)
{
    nvmlComputeInstance_t computeInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlComputeInstanceDestroy(computeInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetComputeInstances(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int count;
    nvmlComputeInstance_t* computeInstances;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &profileId, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    computeInstances = (nvmlComputeInstance_t*)malloc(count * sizeof(nvmlComputeInstance_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, computeInstances, count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) computeInstances);
ERROR_0:
    return -1;
}

int handle_nvmlGpuInstanceGetComputeInstanceById(void *conn)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int id;
    nvmlComputeInstance_t computeInstance;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &id, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, &computeInstance);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlComputeInstanceGetInfo_v2(void *conn)
{
    nvmlComputeInstance_t computeInstance;
    nvmlComputeInstanceInfo_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlComputeInstanceGetInfo_v2(computeInstance, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceIsMigDeviceHandle(void *conn)
{
    nvmlDevice_t device;
    unsigned int isMigDevice;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceIsMigDeviceHandle(device, &isMigDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuInstanceId(void *conn)
{
    nvmlDevice_t device;
    unsigned int id;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuInstanceId(device, &id);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &id, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetComputeInstanceId(void *conn)
{
    nvmlDevice_t device;
    unsigned int id;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetComputeInstanceId(device, &id);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &id, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMaxMigDeviceCount(void *conn)
{
    nvmlDevice_t device;
    unsigned int count;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMaxMigDeviceCount(device, &count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMigDeviceHandleByIndex(void *conn)
{
    nvmlDevice_t device;
    unsigned int index;
    nvmlDevice_t migDevice;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &index, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMigDeviceHandleByIndex(device, index, &migDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDeviceHandleFromMigDeviceHandle(void *conn)
{
    nvmlDevice_t migDevice;
    nvmlDevice_t device;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &migDevice, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, &device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetBusType(void *conn)
{
    nvmlDevice_t device;
    nvmlBusType_t type;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetBusType(device, &type);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetDynamicPstatesInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetFanSpeed_v2(void *conn)
{
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int speed;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fan, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &speed, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetFanSpeed_v2(device, fan, speed);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpcClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    int offset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpcClkVfOffset(device, &offset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &offset, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetGpcClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    int offset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetGpcClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    int offset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMemClkVfOffset(device, &offset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &offset, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetMemClkVfOffset(void *conn)
{
    nvmlDevice_t device;
    int offset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetMemClkVfOffset(device, offset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMinMaxClockOfPState(void *conn)
{
    nvmlDevice_t device;
    nvmlClockType_t type;
    nvmlPstates_t pstate;
    unsigned int minClockMHz;
    unsigned int maxClockMHz;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(nvmlClockType_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pstate, sizeof(nvmlPstates_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(void *conn)
{
    nvmlDevice_t device;
    unsigned int size;
    nvmlPstates_t* pstates;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    pstates = (nvmlPstates_t*)malloc(size * sizeof(nvmlPstates_t));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = nvmlDeviceGetSupportedPerformanceStates(device, pstates, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, pstates, size * sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pstates);
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(void *conn)
{
    nvmlDevice_t device;
    int minOffset;
    int maxOffset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &minOffset, sizeof(int)) < 0 ||
        rpc_write(conn, &maxOffset, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetMemClkMinMaxVfOffset(void *conn)
{
    nvmlDevice_t device;
    int minOffset;
    int maxOffset;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetMemClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &minOffset, sizeof(int)) < 0 ||
        rpc_write(conn, &maxOffset, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceGetGpuFabricInfo(void *conn)
{
    nvmlDevice_t device;
    nvmlGpuFabricInfo_t gpuFabricInfo;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpmMetricsGet(void *conn)
{
    nvmlGpmMetricsGet_t metricsGet;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpmMetricsGet(&metricsGet);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpmSampleFree(void *conn)
{
    nvmlGpmSample_t gpmSample;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpmSampleFree(gpmSample);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpmSampleAlloc(void *conn)
{
    nvmlGpmSample_t gpmSample;
    int request_id;
    nvmlReturn_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpmSampleAlloc(&gpmSample);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpmSampleGet(void *conn)
{
    nvmlDevice_t device;
    nvmlGpmSample_t gpmSample;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpmSampleGet(device, gpmSample);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpmMigSampleGet(void *conn)
{
    nvmlDevice_t device;
    unsigned int gpuInstanceId;
    nvmlGpmSample_t gpmSample;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gpuInstanceId, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlGpmQueryDeviceSupport(void *conn)
{
    nvmlDevice_t device;
    nvmlGpmSupport_t gpmSupport;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlGpmQueryDeviceSupport(device, &gpmSupport);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold(void *conn)
{
    nvmlDevice_t device;
    nvmlNvLinkPowerThres_t info;
    int request_id;
    nvmlReturn_t result;
    if (rpc_read(conn, &device, sizeof(nvmlDevice_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, &info);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuInit(void *conn)
{
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuInit(Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDriverGetVersion(void *conn)
{
    int driverVersion;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDriverGetVersion(&driverVersion);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &driverVersion, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGet(void *conn)
{
    CUdevice device;
    int ordinal;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ordinal, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGet(&device, ordinal);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetCount(void *conn)
{
    int count;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetCount(&count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetName(void *conn)
{
    int len;
    char* name;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        goto ERROR_0;
    name = (char*)malloc(len * sizeof(char));
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuDeviceGetName(name, len, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, name, len * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuDeviceGetUuid(void *conn)
{
    CUuuid* uuid;
    CUdevice dev;
    int request_id;
    CUresult result;
    uuid = (CUuuid*)malloc(16);
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuDeviceGetUuid(uuid, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, uuid, 16) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) uuid);
ERROR_0:
    return -1;
}

int handle_cuDeviceGetUuid_v2(void *conn)
{
    CUuuid* uuid;
    CUdevice dev;
    int request_id;
    CUresult result;
    uuid = (CUuuid*)malloc(16);
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuDeviceGetUuid_v2(uuid, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, uuid, 16) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) uuid);
ERROR_0:
    return -1;
}

int handle_cuDeviceGetLuid(void *conn)
{
    std::size_t luid_len;
    char* luid;
    unsigned int deviceNodeMask;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &luid_len, sizeof(luid_len)) < 0)
        goto ERROR_0;
    luid = (char*)malloc(luid_len);
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuDeviceGetLuid(luid, &deviceNodeMask, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &luid_len, sizeof(luid_len)) < 0 ||
        rpc_write(conn, luid, luid_len) < 0 ||
        rpc_write(conn, &deviceNodeMask, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) luid);
ERROR_0:
    return -1;
}

int handle_cuDeviceTotalMem_v2(void *conn)
{
    size_t bytes;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceTotalMem_v2(&bytes, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetTexture1DLinearMaxWidth(void *conn)
{
    size_t maxWidthInElements;
    CUarray_format format;
    unsigned numChannels;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &format, sizeof(CUarray_format)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numChannels, sizeof(unsigned)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetTexture1DLinearMaxWidth(&maxWidthInElements, format, numChannels, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &maxWidthInElements, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetAttribute(void *conn)
{
    int pi;
    CUdevice_attribute attrib;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &attrib, sizeof(CUdevice_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetAttribute(&pi, attrib, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pi, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceSetMemPool(void *conn)
{
    CUdevice dev;
    CUmemoryPool pool;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceSetMemPool(dev, pool);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetMemPool(void *conn)
{
    CUmemoryPool pool;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetMemPool(&pool, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetDefaultMemPool(void *conn)
{
    CUmemoryPool pool_out;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetDefaultMemPool(&pool_out, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetExecAffinitySupport(void *conn)
{
    int pi;
    CUexecAffinityType type;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &type, sizeof(CUexecAffinityType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetExecAffinitySupport(&pi, type, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pi, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFlushGPUDirectRDMAWrites(void *conn)
{
    CUflushGPUDirectRDMAWritesTarget target;
    CUflushGPUDirectRDMAWritesScope scope;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFlushGPUDirectRDMAWrites(target, scope);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetProperties(void *conn)
{
    CUdevprop prop;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetProperties(&prop, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &prop, sizeof(CUdevprop)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceComputeCapability(void *conn)
{
    int major;
    int minor;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceComputeCapability(&major, &minor, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &major, sizeof(int)) < 0 ||
        rpc_write(conn, &minor, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDevicePrimaryCtxRetain(void *conn)
{
    CUcontext pctx;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDevicePrimaryCtxRetain(&pctx, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDevicePrimaryCtxRelease_v2(void *conn)
{
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDevicePrimaryCtxRelease_v2(dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDevicePrimaryCtxSetFlags_v2(void *conn)
{
    CUdevice dev;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDevicePrimaryCtxSetFlags_v2(dev, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDevicePrimaryCtxGetState(void *conn)
{
    CUdevice dev;
    unsigned int flags;
    int active;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDevicePrimaryCtxGetState(dev, &flags, &active);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &active, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDevicePrimaryCtxReset_v2(void *conn)
{
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDevicePrimaryCtxReset_v2(dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxCreate_v2(void *conn)
{
    CUcontext pctx;
    unsigned int flags;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxCreate_v2(&pctx, flags, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxCreate_v3(void *conn)
{
    CUcontext pctx;
    int numParams;
    CUexecAffinityParam* paramsArray;
    unsigned int flags;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &numParams, sizeof(int)) < 0)
        goto ERROR_0;
    paramsArray = (CUexecAffinityParam*)malloc(numParams * sizeof(CUexecAffinityParam));
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_1;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuCtxCreate_v3(&pctx, paramsArray, numParams, flags, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_write(conn, paramsArray, numParams * sizeof(CUexecAffinityParam)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) paramsArray);
ERROR_0:
    return -1;
}

int handle_cuCtxDestroy_v2(void *conn)
{
    CUcontext ctx;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxDestroy_v2(ctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxPushCurrent_v2(void *conn)
{
    CUcontext ctx;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxPushCurrent_v2(ctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxPopCurrent_v2(void *conn)
{
    CUcontext pctx;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxPopCurrent_v2(&pctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxSetCurrent(void *conn)
{
    CUcontext ctx;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxSetCurrent(ctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetCurrent(void *conn)
{
    CUcontext pctx;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetCurrent(&pctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetDevice(void *conn)
{
    CUdevice device;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetDevice(&device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetFlags(void *conn)
{
    unsigned int flags;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetFlags(&flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetId(void *conn)
{
    CUcontext ctx;
    unsigned long long ctxId;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetId(ctx, &ctxId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ctxId, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxSynchronize(void *conn)
{
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxSynchronize();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxSetLimit(void *conn)
{
    CUlimit limit;
    size_t value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &limit, sizeof(CUlimit)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetLimit(void *conn)
{
    size_t pvalue;
    CUlimit limit;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &limit, sizeof(CUlimit)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetLimit(&pvalue, limit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pvalue, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetCacheConfig(void *conn)
{
    CUfunc_cache pconfig;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetCacheConfig(&pconfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pconfig, sizeof(CUfunc_cache)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxSetCacheConfig(void *conn)
{
    CUfunc_cache config;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxSetCacheConfig(config);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetSharedMemConfig(void *conn)
{
    CUsharedconfig pConfig;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetSharedMemConfig(&pConfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pConfig, sizeof(CUsharedconfig)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxSetSharedMemConfig(void *conn)
{
    CUsharedconfig config;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &config, sizeof(CUsharedconfig)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxSetSharedMemConfig(config);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetApiVersion(void *conn)
{
    CUcontext ctx;
    unsigned int version;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetApiVersion(ctx, &version);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &version, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetStreamPriorityRange(void *conn)
{
    int leastPriority;
    int greatestPriority;
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &leastPriority, sizeof(int)) < 0 ||
        rpc_write(conn, &greatestPriority, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxResetPersistingL2Cache(void *conn)
{
    int request_id;
    CUresult result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxResetPersistingL2Cache();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxGetExecAffinity(void *conn)
{
    CUexecAffinityParam pExecAffinity;
    CUexecAffinityType type;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &type, sizeof(CUexecAffinityType)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxGetExecAffinity(&pExecAffinity, type);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pExecAffinity, sizeof(CUexecAffinityParam)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxAttach(void *conn)
{
    CUcontext pctx;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxAttach(&pctx, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxDetach(void *conn)
{
    CUcontext ctx;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxDetach(ctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuModuleLoad(void *conn)
{
    CUmodule module;
    std::size_t fname_len;
    const char* fname;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &fname_len, sizeof(fname_len)) < 0)
        goto ERROR_0;
    fname = (const char*)malloc(fname_len);
    if (rpc_read(conn, &fname, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuModuleLoad(&module, fname);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &module, sizeof(CUmodule)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) fname);
ERROR_0:
    return -1;
}

int handle_cuModuleUnload(void *conn)
{
    CUmodule hmod;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuModuleUnload(hmod);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuModuleGetLoadingMode(void *conn)
{
    CUmoduleLoadingMode mode;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &mode, sizeof(CUmoduleLoadingMode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuModuleGetLoadingMode(&mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(CUmoduleLoadingMode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuModuleGetFunction(void *conn)
{
    CUfunction hfunc;
    CUmodule hmod;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuModuleGetFunction(&hfunc, hmod, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuModuleGetGlobal_v2(void *conn)
{
    CUdeviceptr dptr;
    size_t bytes;
    CUmodule hmod;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuModuleGetGlobal_v2(&dptr, &bytes, hmod, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuLinkCreate_v2(void *conn)
{
    unsigned int numOptions;
    CUjit_option options;
    void* optionValues;
    CUlinkState stateOut;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &options, sizeof(CUjit_option)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &optionValues, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stateOut, sizeof(CUlinkState)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLinkCreate_v2(numOptions, &options, &optionValues, &stateOut);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &options, sizeof(CUjit_option)) < 0 ||
        rpc_write(conn, &optionValues, sizeof(void*)) < 0 ||
        rpc_write(conn, &stateOut, sizeof(CUlinkState)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLinkAddData_v2(void *conn)
{
    CUlinkState state;
    CUjitInputType type;
    void* data;
    size_t size;
    std::size_t name_len;
    const char* name;
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(CUjitInputType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;
    if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0)
        goto ERROR_1;
    options = (CUjit_option*)malloc(numOptions * sizeof(CUjit_option));
    if (rpc_read(conn, &options, sizeof(CUjit_option*)) < 0)
        goto ERROR_2;
    optionValues = (void**)malloc(numOptions * sizeof(void*));
    if (rpc_read(conn, &optionValues, sizeof(void**)) < 0)
        goto ERROR_3;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_3;
    result = cuLinkAddData_v2(state, type, &data, size, name, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &data, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_3;

    return 0;
ERROR_3:
    free((void *) name);
ERROR_2:
    free((void *) options);
ERROR_1:
    free((void *) optionValues);
ERROR_0:
    return -1;
}

int handle_cuLinkAddFile_v2(void *conn)
{
    CUlinkState state;
    CUjitInputType type;
    std::size_t path_len;
    const char* path;
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(CUjitInputType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &path_len, sizeof(path_len)) < 0)
        goto ERROR_0;
    path = (const char*)malloc(path_len);
    if (rpc_read(conn, &path, sizeof(const char*)) < 0)
        goto ERROR_1;
    if (rpc_read(conn, &numOptions, sizeof(unsigned int)) < 0)
        goto ERROR_1;
    options = (CUjit_option*)malloc(numOptions * sizeof(CUjit_option));
    if (rpc_read(conn, &options, sizeof(CUjit_option*)) < 0)
        goto ERROR_2;
    optionValues = (void**)malloc(numOptions * sizeof(void*));
    if (rpc_read(conn, &optionValues, sizeof(void**)) < 0)
        goto ERROR_3;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_3;
    result = cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_3;

    return 0;
ERROR_3:
    free((void *) path);
ERROR_2:
    free((void *) options);
ERROR_1:
    free((void *) optionValues);
ERROR_0:
    return -1;
}

int handle_cuLinkComplete(void *conn)
{
    CUlinkState state;
    void* cubinOut;
    size_t sizeOut;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLinkComplete(state, &cubinOut, &sizeOut);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &cubinOut, sizeof(void*)) < 0 ||
        rpc_write(conn, &sizeOut, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLinkDestroy(void *conn)
{
    CUlinkState state;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &state, sizeof(CUlinkState)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLinkDestroy(state);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuModuleGetTexRef(void *conn)
{
    CUtexref pTexRef;
    CUmodule hmod;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuModuleGetTexRef(&pTexRef, hmod, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuModuleGetSurfRef(void *conn)
{
    CUsurfref pSurfRef;
    CUmodule hmod;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuModuleGetSurfRef(&pSurfRef, hmod, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pSurfRef, sizeof(CUsurfref)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuLibraryLoadFromFile(void *conn)
{
    CUlibrary library;
    std::size_t fileName_len;
    const char* fileName;
    unsigned int numJitOptions;
    CUjit_option* jitOptions;
    void** jitOptionsValues;
    unsigned int numLibraryOptions;
    CUlibraryOption* libraryOptions;
    void** libraryOptionValues;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &fileName_len, sizeof(fileName_len)) < 0)
        goto ERROR_0;
    fileName = (const char*)malloc(fileName_len);
    if (rpc_read(conn, &fileName, sizeof(const char*)) < 0)
        goto ERROR_1;
    if (rpc_read(conn, &numJitOptions, sizeof(unsigned int)) < 0)
        goto ERROR_1;
    jitOptions = (CUjit_option*)malloc(numJitOptions * sizeof(CUjit_option));
    if (rpc_read(conn, &jitOptions, sizeof(CUjit_option*)) < 0)
        goto ERROR_2;
    jitOptionsValues = (void**)malloc(numJitOptions * sizeof(void*));
    if (rpc_read(conn, &jitOptionsValues, sizeof(void**)) < 0)
        goto ERROR_3;
    if (rpc_read(conn, &numLibraryOptions, sizeof(unsigned int)) < 0)
        goto ERROR_3;
    libraryOptions = (CUlibraryOption*)malloc(numLibraryOptions * sizeof(CUlibraryOption));
    if (rpc_read(conn, &libraryOptions, sizeof(CUlibraryOption*)) < 0)
        goto ERROR_4;
    libraryOptionValues = (void**)malloc(numLibraryOptions * sizeof(void*));
    if (rpc_read(conn, &libraryOptionValues, sizeof(void**)) < 0)
        goto ERROR_5;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_5;
    result = cuLibraryLoadFromFile(&library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_5;

    return 0;
ERROR_5:
    free((void *) fileName);
ERROR_4:
    free((void *) jitOptions);
ERROR_3:
    free((void *) jitOptionsValues);
ERROR_2:
    free((void *) libraryOptions);
ERROR_1:
    free((void *) libraryOptionValues);
ERROR_0:
    return -1;
}

int handle_cuLibraryUnload(void *conn)
{
    CUlibrary library;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLibraryUnload(library);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLibraryGetKernel(void *conn)
{
    CUkernel pKernel;
    CUlibrary library;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuLibraryGetKernel(&pKernel, library, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pKernel, sizeof(CUkernel)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuLibraryGetModule(void *conn)
{
    CUmodule pMod;
    CUlibrary library;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLibraryGetModule(&pMod, library);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pMod, sizeof(CUmodule)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuKernelGetFunction(void *conn)
{
    CUfunction pFunc;
    CUkernel kernel;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuKernelGetFunction(&pFunc, kernel);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pFunc, sizeof(CUfunction)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLibraryGetGlobal(void *conn)
{
    CUdeviceptr dptr;
    size_t bytes;
    CUlibrary library;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuLibraryGetGlobal(&dptr, &bytes, library, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuLibraryGetManaged(void *conn)
{
    CUdeviceptr dptr;
    size_t bytes;
    CUlibrary library;
    std::size_t name_len;
    const char* name;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0)
        goto ERROR_0;
    name = (const char*)malloc(name_len);
    if (rpc_read(conn, &name, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuLibraryGetManaged(&dptr, &bytes, library, name);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) name);
ERROR_0:
    return -1;
}

int handle_cuLibraryGetUnifiedFunction(void *conn)
{
    void* fptr;
    CUlibrary library;
    std::size_t symbol_len;
    const char* symbol;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &symbol_len, sizeof(symbol_len)) < 0)
        goto ERROR_0;
    symbol = (const char*)malloc(symbol_len);
    if (rpc_read(conn, &symbol, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuLibraryGetUnifiedFunction(&fptr, library, symbol);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &fptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) symbol);
ERROR_0:
    return -1;
}

int handle_cuKernelGetAttribute(void *conn)
{
    int pi;
    CUfunction_attribute attrib;
    CUkernel kernel;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pi, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuKernelGetAttribute(&pi, attrib, kernel, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pi, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuKernelSetAttribute(void *conn)
{
    CUfunction_attribute attrib;
    int val;
    CUkernel kernel;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &val, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuKernelSetAttribute(attrib, val, kernel, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuKernelSetCacheConfig(void *conn)
{
    CUkernel kernel;
    CUfunc_cache config;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuKernelSetCacheConfig(kernel, config, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemGetInfo_v2(void *conn)
{
    size_t free;
    size_t total;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &free, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &total, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemGetInfo_v2(&free, &total);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &free, sizeof(size_t)) < 0 ||
        rpc_write(conn, &total, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAlloc_v2(void *conn)
{
    CUdeviceptr dptr;
    size_t bytesize;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAlloc_v2(&dptr, bytesize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAllocPitch_v2(void *conn)
{
    CUdeviceptr dptr;
    size_t pPitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &WidthInBytes, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ElementSizeBytes, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAllocPitch_v2(&dptr, &pPitch, WidthInBytes, Height, ElementSizeBytes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &pPitch, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemFree_v2(void *conn)
{
    CUdeviceptr dptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemFree_v2(dptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemGetAddressRange_v2(void *conn)
{
    CUdeviceptr pbase;
    size_t psize;
    CUdeviceptr dptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pbase, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &psize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemGetAddressRange_v2(&pbase, &psize, dptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pbase, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &psize, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAllocHost_v2(void *conn)
{
    void* pp;
    size_t bytesize;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pp, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAllocHost_v2(&pp, bytesize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pp, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemFreeHost(void *conn)
{
    void* p;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemFreeHost(&p);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &p, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemHostAlloc(void *conn)
{
    void* pp;
    size_t bytesize;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pp, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemHostAlloc(&pp, bytesize, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pp, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemHostGetDevicePointer_v2(void *conn)
{
    CUdeviceptr pdptr;
    void* p;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemHostGetDevicePointer_v2(&pdptr, &p, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &p, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemHostGetFlags(void *conn)
{
    unsigned int pFlags;
    void* p;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemHostGetFlags(&pFlags, &p);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &p, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAllocManaged(void *conn)
{
    CUdeviceptr dptr;
    size_t bytesize;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAllocManaged(&dptr, bytesize, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetByPCIBusId(void *conn)
{
    CUdevice dev;
    std::size_t pciBusId_len;
    const char* pciBusId;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pciBusId_len, sizeof(pciBusId_len)) < 0)
        goto ERROR_0;
    pciBusId = (const char*)malloc(pciBusId_len);
    if (rpc_read(conn, &pciBusId, sizeof(const char*)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuDeviceGetByPCIBusId(&dev, pciBusId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pciBusId);
ERROR_0:
    return -1;
}

int handle_cuDeviceGetPCIBusId(void *conn)
{
    int len;
    char* pciBusId;
    CUdevice dev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        goto ERROR_0;
    pciBusId = (char*)malloc(len * sizeof(char));
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cuDeviceGetPCIBusId(pciBusId, len, dev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, pciBusId, len * sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) pciBusId);
ERROR_0:
    return -1;
}

int handle_cuIpcGetEventHandle(void *conn)
{
    CUipcEventHandle pHandle;
    CUevent event;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pHandle, sizeof(CUipcEventHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuIpcGetEventHandle(&pHandle, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pHandle, sizeof(CUipcEventHandle)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuIpcOpenEventHandle(void *conn)
{
    CUevent phEvent;
    CUipcEventHandle handle;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(CUipcEventHandle)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuIpcOpenEventHandle(&phEvent, handle);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phEvent, sizeof(CUevent)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuIpcGetMemHandle(void *conn)
{
    CUipcMemHandle pHandle;
    CUdeviceptr dptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pHandle, sizeof(CUipcMemHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuIpcGetMemHandle(&pHandle, dptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pHandle, sizeof(CUipcMemHandle)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuIpcOpenMemHandle_v2(void *conn)
{
    CUdeviceptr pdptr;
    CUipcMemHandle handle;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(CUipcMemHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuIpcOpenMemHandle_v2(&pdptr, handle, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuIpcCloseMemHandle(void *conn)
{
    CUdeviceptr dptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuIpcCloseMemHandle(dptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemHostRegister_v2(void *conn)
{
    void* p;
    size_t bytesize;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemHostRegister_v2(&p, bytesize, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &p, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemHostUnregister(void *conn)
{
    void* p;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &p, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemHostUnregister(&p);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &p, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpy(void *conn)
{
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dst, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpy(dst, src, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyPeer(void *conn)
{
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstContext, sizeof(CUcontext)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcContext, sizeof(CUcontext)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyDtoH_v2(void *conn)
{
    void* dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyDtoH_v2(&dstHost, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dstHost, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyDtoD_v2(void *conn)
{
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyDtoA_v2(void *conn)
{
    CUarray dstArray;
    size_t dstOffset;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyAtoD_v2(void *conn)
{
    CUdeviceptr dstDevice;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyAtoH_v2(void *conn)
{
    void* dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyAtoH_v2(&dstHost, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyAtoA_v2(void *conn)
{
    CUarray dstArray;
    size_t dstOffset;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyAsync(void *conn)
{
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t ByteCount;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dst, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyAsync(dst, src, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyPeerAsync(void *conn)
{
    CUdeviceptr dstDevice;
    CUcontext dstContext;
    CUdeviceptr srcDevice;
    CUcontext srcContext;
    size_t ByteCount;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstContext, sizeof(CUcontext)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcContext, sizeof(CUcontext)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyDtoHAsync_v2(void *conn)
{
    void* dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyDtoHAsync_v2(&dstHost, srcDevice, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dstHost, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyDtoDAsync_v2(void *conn)
{
    CUdeviceptr dstDevice;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemcpyAtoHAsync_v2(void *conn)
{
    void* dstHost;
    CUarray srcArray;
    size_t srcOffset;
    size_t ByteCount;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstHost, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ByteCount, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemcpyAtoHAsync_v2(&dstHost, srcArray, srcOffset, ByteCount, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dstHost, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD8_v2(void *conn)
{
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD8_v2(dstDevice, uc, N);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD16_v2(void *conn)
{
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD16_v2(dstDevice, us, N);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD32_v2(void *conn)
{
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD32_v2(dstDevice, ui, N);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD2D8_v2(void *conn)
{
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD2D16_v2(void *conn)
{
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD2D32_v2(void *conn)
{
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD8Async(void *conn)
{
    CUdeviceptr dstDevice;
    unsigned char uc;
    size_t N;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD8Async(dstDevice, uc, N, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD16Async(void *conn)
{
    CUdeviceptr dstDevice;
    unsigned short us;
    size_t N;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD16Async(dstDevice, us, N, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD32Async(void *conn)
{
    CUdeviceptr dstDevice;
    unsigned int ui;
    size_t N;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &N, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD32Async(dstDevice, ui, N, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD2D8Async(void *conn)
{
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned char uc;
    size_t Width;
    size_t Height;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD2D16Async(void *conn)
{
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned short us;
    size_t Width;
    size_t Height;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemsetD2D32Async(void *conn)
{
    CUdeviceptr dstDevice;
    size_t dstPitch;
    unsigned int ui;
    size_t Width;
    size_t Height;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dstDevice, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuArrayGetDescriptor_v2(void *conn)
{
    CUDA_ARRAY_DESCRIPTOR pArrayDescriptor;
    CUarray hArray;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuArrayGetDescriptor_v2(&pArrayDescriptor, hArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuArrayGetSparseProperties(void *conn)
{
    CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties;
    CUarray array;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &array, sizeof(CUarray)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuArrayGetSparseProperties(&sparseProperties, array);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMipmappedArrayGetSparseProperties(void *conn)
{
    CUDA_ARRAY_SPARSE_PROPERTIES sparseProperties;
    CUmipmappedArray mipmap;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMipmappedArrayGetSparseProperties(&sparseProperties, mipmap);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuArrayGetMemoryRequirements(void *conn)
{
    CUDA_ARRAY_MEMORY_REQUIREMENTS memoryRequirements;
    CUarray array;
    CUdevice device;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &array, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuArrayGetMemoryRequirements(&memoryRequirements, array, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMipmappedArrayGetMemoryRequirements(void *conn)
{
    CUDA_ARRAY_MEMORY_REQUIREMENTS memoryRequirements;
    CUmipmappedArray mipmap;
    CUdevice device;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipmap, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMipmappedArrayGetMemoryRequirements(&memoryRequirements, mipmap, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memoryRequirements, sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuArrayGetPlane(void *conn)
{
    CUarray pPlaneArray;
    CUarray hArray;
    unsigned int planeIdx;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pPlaneArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &planeIdx, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuArrayGetPlane(&pPlaneArray, hArray, planeIdx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pPlaneArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuArrayDestroy(void *conn)
{
    CUarray hArray;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuArrayDestroy(hArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuArray3DGetDescriptor_v2(void *conn)
{
    CUDA_ARRAY3D_DESCRIPTOR pArrayDescriptor;
    CUarray hArray;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuArray3DGetDescriptor_v2(&pArrayDescriptor, hArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMipmappedArrayGetLevel(void *conn)
{
    CUarray pLevelArray;
    CUmipmappedArray hMipmappedArray;
    unsigned int level;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pLevelArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &level, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMipmappedArrayGetLevel(&pLevelArray, hMipmappedArray, level);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pLevelArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMipmappedArrayDestroy(void *conn)
{
    CUmipmappedArray hMipmappedArray;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMipmappedArrayDestroy(hMipmappedArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemGetHandleForAddressRange(void *conn)
{
    void* handle;
    CUdeviceptr dptr;
    size_t size;
    CUmemRangeHandleType handleType;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &handle, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handleType, sizeof(CUmemRangeHandleType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemGetHandleForAddressRange(&handle, dptr, size, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &handle, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAddressReserve(void *conn)
{
    CUdeviceptr ptr;
    size_t size;
    size_t alignment;
    CUdeviceptr addr;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &alignment, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAddressReserve(&ptr, size, alignment, addr, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAddressFree(void *conn)
{
    CUdeviceptr ptr;
    size_t size;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAddressFree(ptr, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemRelease(void *conn)
{
    CUmemGenericAllocationHandle handle;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemRelease(handle);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemMap(void *conn)
{
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUmemGenericAllocationHandle handle;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &offset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemMap(ptr, size, offset, handle, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemMapArrayAsync(void *conn)
{
    CUarrayMapInfo mapInfoList;
    unsigned int count;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &mapInfoList, sizeof(CUarrayMapInfo)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemMapArrayAsync(&mapInfoList, count, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemUnmap(void *conn)
{
    CUdeviceptr ptr;
    size_t size;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemUnmap(ptr, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemExportToShareableHandle(void *conn)
{
    void* shareableHandle;
    CUmemGenericAllocationHandle handle;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &shareableHandle, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemExportToShareableHandle(&shareableHandle, handle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &shareableHandle, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemImportFromShareableHandle(void *conn)
{
    CUmemGenericAllocationHandle handle;
    void* osHandle;
    CUmemAllocationHandleType shHandleType;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &osHandle, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &shHandleType, sizeof(CUmemAllocationHandleType)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemImportFromShareableHandle(&handle, &osHandle, shHandleType);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(conn, &osHandle, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemGetAllocationPropertiesFromHandle(void *conn)
{
    CUmemAllocationProp prop;
    CUmemGenericAllocationHandle handle;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &prop, sizeof(CUmemAllocationProp)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemGetAllocationPropertiesFromHandle(&prop, handle);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &prop, sizeof(CUmemAllocationProp)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemRetainAllocationHandle(void *conn)
{
    CUmemGenericAllocationHandle handle;
    void* addr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &addr, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemRetainAllocationHandle(&handle, &addr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
        rpc_write(conn, &addr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemFreeAsync(void *conn)
{
    CUdeviceptr dptr;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemFreeAsync(dptr, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAllocAsync(void *conn)
{
    CUdeviceptr dptr;
    size_t bytesize;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAllocAsync(&dptr, bytesize, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolTrimTo(void *conn)
{
    CUmemoryPool pool;
    size_t minBytesToKeep;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &minBytesToKeep, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolTrimTo(pool, minBytesToKeep);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolSetAttribute(void *conn)
{
    CUmemoryPool pool;
    CUmemPool_attribute attr;
    void* value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(CUmemPool_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolSetAttribute(pool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolGetAttribute(void *conn)
{
    CUmemoryPool pool;
    CUmemPool_attribute attr;
    void* value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(CUmemPool_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolGetAttribute(pool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolGetAccess(void *conn)
{
    CUmemAccess_flags flags;
    CUmemoryPool memPool;
    CUmemLocation location;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &flags, sizeof(CUmemAccess_flags)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memPool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &location, sizeof(CUmemLocation)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolGetAccess(&flags, memPool, &location);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(CUmemAccess_flags)) < 0 ||
        rpc_write(conn, &location, sizeof(CUmemLocation)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolDestroy(void *conn)
{
    CUmemoryPool pool;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolDestroy(pool);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAllocFromPoolAsync(void *conn)
{
    CUdeviceptr dptr;
    size_t bytesize;
    CUmemoryPool pool;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAllocFromPoolAsync(&dptr, bytesize, pool, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolExportToShareableHandle(void *conn)
{
    void* handle_out;
    CUmemoryPool pool;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &handle_out, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolExportToShareableHandle(&handle_out, pool, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &handle_out, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolImportFromShareableHandle(void *conn)
{
    CUmemoryPool pool_out;
    void* handle;
    CUmemAllocationHandleType handleType;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pool_out, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handleType, sizeof(CUmemAllocationHandleType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolImportFromShareableHandle(&pool_out, &handle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pool_out, sizeof(CUmemoryPool)) < 0 ||
        rpc_write(conn, &handle, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolExportPointer(void *conn)
{
    CUmemPoolPtrExportData shareData_out;
    CUdeviceptr ptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &shareData_out, sizeof(CUmemPoolPtrExportData)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolExportPointer(&shareData_out, ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPoolImportPointer(void *conn)
{
    CUdeviceptr ptr_out;
    CUmemoryPool pool;
    CUmemPoolPtrExportData shareData;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ptr_out, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &shareData, sizeof(CUmemPoolPtrExportData)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPoolImportPointer(&ptr_out, pool, &shareData);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuPointerGetAttribute(void *conn)
{
    void* data;
    CUpointer_attribute attribute;
    CUdeviceptr ptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attribute, sizeof(CUpointer_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuPointerGetAttribute(&data, attribute, ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &data, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemPrefetchAsync(void *conn)
{
    CUdeviceptr devPtr;
    size_t count;
    CUdevice dstDevice;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstDevice, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemAdvise(void *conn)
{
    CUdeviceptr devPtr;
    size_t count;
    CUmem_advise advice;
    CUdevice device;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &advice, sizeof(CUmem_advise)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemAdvise(devPtr, count, advice, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemRangeGetAttribute(void *conn)
{
    void* data;
    size_t dataSize;
    CUmem_range_attribute attribute;
    CUdeviceptr devPtr;
    size_t count;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dataSize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attribute, sizeof(CUmem_range_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemRangeGetAttribute(&data, dataSize, attribute, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &data, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuMemRangeGetAttributes(void *conn)
{
    void* data;
    size_t dataSizes;
    CUmem_range_attribute attributes;
    size_t numAttributes;
    CUdeviceptr devPtr;
    size_t count;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dataSizes, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attributes, sizeof(CUmem_range_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numAttributes, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuMemRangeGetAttributes(&data, &dataSizes, &attributes, numAttributes, devPtr, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &data, sizeof(void*)) < 0 ||
        rpc_write(conn, &dataSizes, sizeof(size_t)) < 0 ||
        rpc_write(conn, &attributes, sizeof(CUmem_range_attribute)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuPointerGetAttributes(void *conn)
{
    unsigned int numAttributes;
    CUpointer_attribute attributes;
    void* data;
    CUdeviceptr ptr;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &numAttributes, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attributes, sizeof(CUpointer_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &data, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuPointerGetAttributes(numAttributes, &attributes, &data, ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &attributes, sizeof(CUpointer_attribute)) < 0 ||
        rpc_write(conn, &data, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamCreate(void *conn)
{
    CUstream phStream;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamCreate(&phStream, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phStream, sizeof(CUstream)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamCreateWithPriority(void *conn)
{
    CUstream phStream;
    unsigned int flags;
    int priority;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamCreateWithPriority(&phStream, flags, priority);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phStream, sizeof(CUstream)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamGetPriority(void *conn)
{
    CUstream hStream;
    int priority;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamGetPriority(hStream, &priority);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &priority, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamGetFlags(void *conn)
{
    CUstream hStream;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamGetFlags(hStream, &flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamGetId(void *conn)
{
    CUstream hStream;
    unsigned long long streamId;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &streamId, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamGetId(hStream, &streamId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &streamId, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamGetCtx(void *conn)
{
    CUstream hStream;
    CUcontext pctx;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pctx, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamGetCtx(hStream, &pctx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamWaitEvent(void *conn)
{
    CUstream hStream;
    CUevent hEvent;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamWaitEvent(hStream, hEvent, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamAddCallback(void *conn)
{
    CUstream hStream;
    CUstreamCallback callback;
    void* userData;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &callback, sizeof(CUstreamCallback)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamAddCallback(hStream, callback, &userData, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &userData, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamBeginCapture_v2(void *conn)
{
    CUstream hStream;
    CUstreamCaptureMode mode;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(CUstreamCaptureMode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamBeginCapture_v2(hStream, mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuThreadExchangeStreamCaptureMode(void *conn)
{
    CUstreamCaptureMode mode;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &mode, sizeof(CUstreamCaptureMode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuThreadExchangeStreamCaptureMode(&mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamEndCapture(void *conn)
{
    CUstream hStream;
    CUgraph phGraph;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamEndCapture(hStream, &phGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamIsCapturing(void *conn)
{
    CUstream hStream;
    CUstreamCaptureStatus captureStatus;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &captureStatus, sizeof(CUstreamCaptureStatus)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamIsCapturing(hStream, &captureStatus);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamUpdateCaptureDependencies(void *conn)
{
    CUstream hStream;
    CUgraphNode dependencies;
    size_t numDependencies;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dependencies, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamUpdateCaptureDependencies(hStream, &dependencies, numDependencies, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamAttachMemAsync(void *conn)
{
    CUstream hStream;
    CUdeviceptr dptr;
    size_t length;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamAttachMemAsync(hStream, dptr, length, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamQuery(void *conn)
{
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamQuery(hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamSynchronize(void *conn)
{
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamSynchronize(hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamDestroy_v2(void *conn)
{
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamDestroy_v2(hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamCopyAttributes(void *conn)
{
    CUstream dst;
    CUstream src;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dst, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamGetAttribute(void *conn)
{
    CUstream hStream;
    CUstreamAttrID attr;
    CUstreamAttrValue value_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(CUstreamAttrID)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value_out, sizeof(CUstreamAttrValue)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamGetAttribute(hStream, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value_out, sizeof(CUstreamAttrValue)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventCreate(void *conn)
{
    CUevent phEvent;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventCreate(&phEvent, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phEvent, sizeof(CUevent)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventRecord(void *conn)
{
    CUevent hEvent;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventRecord(hEvent, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventRecordWithFlags(void *conn)
{
    CUevent hEvent;
    CUstream hStream;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventRecordWithFlags(hEvent, hStream, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventQuery(void *conn)
{
    CUevent hEvent;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventQuery(hEvent);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventSynchronize(void *conn)
{
    CUevent hEvent;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventSynchronize(hEvent);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventDestroy_v2(void *conn)
{
    CUevent hEvent;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hEvent, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventDestroy_v2(hEvent);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuEventElapsedTime(void *conn)
{
    float pMilliseconds;
    CUevent hStart;
    CUevent hEnd;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pMilliseconds, sizeof(float)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStart, sizeof(CUevent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hEnd, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuEventElapsedTime(&pMilliseconds, hStart, hEnd);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pMilliseconds, sizeof(float)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDestroyExternalMemory(void *conn)
{
    CUexternalMemory extMem;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &extMem, sizeof(CUexternalMemory)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDestroyExternalMemory(extMem);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDestroyExternalSemaphore(void *conn)
{
    CUexternalSemaphore extSem;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &extSem, sizeof(CUexternalSemaphore)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDestroyExternalSemaphore(extSem);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamWaitValue32_v2(void *conn)
{
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(cuuint32_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamWaitValue32_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamWaitValue64_v2(void *conn)
{
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(cuuint64_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamWaitValue64_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamWriteValue32_v2(void *conn)
{
    CUstream stream;
    CUdeviceptr addr;
    cuuint32_t value;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(cuuint32_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamWriteValue32_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamWriteValue64_v2(void *conn)
{
    CUstream stream;
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &addr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(cuuint64_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamWriteValue64_v2(stream, addr, value, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuStreamBatchMemOp_v2(void *conn)
{
    CUstream stream;
    unsigned int count;
    CUstreamBatchMemOpParams paramArray;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &stream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &paramArray, sizeof(CUstreamBatchMemOpParams)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuStreamBatchMemOp_v2(stream, count, &paramArray, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncGetAttribute(void *conn)
{
    int pi;
    CUfunction_attribute attrib;
    CUfunction hfunc;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pi, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncGetAttribute(&pi, attrib, hfunc);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pi, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncSetAttribute(void *conn)
{
    CUfunction hfunc;
    CUfunction_attribute attrib;
    int value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attrib, sizeof(CUfunction_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncSetAttribute(hfunc, attrib, value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncSetCacheConfig(void *conn)
{
    CUfunction hfunc;
    CUfunc_cache config;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &config, sizeof(CUfunc_cache)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncSetCacheConfig(hfunc, config);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncSetSharedMemConfig(void *conn)
{
    CUfunction hfunc;
    CUsharedconfig config;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &config, sizeof(CUsharedconfig)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncSetSharedMemConfig(hfunc, config);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncGetModule(void *conn)
{
    CUmodule hmod;
    CUfunction hfunc;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncGetModule(&hmod, hfunc);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunchKernel(void *conn)
{
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void* kernelParams;
    void* extra;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gridDimX, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gridDimY, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gridDimZ, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockDimX, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockDimY, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockDimZ, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sharedMemBytes, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kernelParams, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &extra, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams, &extra);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunchCooperativeKernel(void *conn)
{
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void* kernelParams;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gridDimX, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gridDimY, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &gridDimZ, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockDimX, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockDimY, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockDimZ, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &sharedMemBytes, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kernelParams, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, &kernelParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &kernelParams, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunchCooperativeKernelMultiDevice(void *conn)
{
    CUDA_LAUNCH_PARAMS launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numDevices, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunchCooperativeKernelMultiDevice(&launchParamsList, numDevices, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunchHostFunc(void *conn)
{
    CUstream hStream;
    CUhostFn fn;
    void* userData;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fn, sizeof(CUhostFn)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunchHostFunc(hStream, fn, &userData);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &userData, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncSetBlockShape(void *conn)
{
    CUfunction hfunc;
    int x;
    int y;
    int z;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &x, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &y, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &z, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncSetBlockShape(hfunc, x, y, z);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuFuncSetSharedSize(void *conn)
{
    CUfunction hfunc;
    unsigned int bytes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytes, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuFuncSetSharedSize(hfunc, bytes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuParamSetSize(void *conn)
{
    CUfunction hfunc;
    unsigned int numbytes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numbytes, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuParamSetSize(hfunc, numbytes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuParamSeti(void *conn)
{
    CUfunction hfunc;
    int offset;
    unsigned int value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuParamSeti(hfunc, offset, value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuParamSetf(void *conn)
{
    CUfunction hfunc;
    int offset;
    float value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(float)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuParamSetf(hfunc, offset, value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuParamSetv(void *conn)
{
    CUfunction hfunc;
    int offset;
    void* ptr;
    unsigned int numbytes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &offset, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numbytes, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuParamSetv(hfunc, offset, &ptr, numbytes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunch(void *conn)
{
    CUfunction f;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunch(f);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunchGrid(void *conn)
{
    CUfunction f;
    int grid_width;
    int grid_height;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &grid_width, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &grid_height, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunchGrid(f, grid_width, grid_height);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuLaunchGridAsync(void *conn)
{
    CUfunction f;
    int grid_width;
    int grid_height;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &f, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &grid_width, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &grid_height, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuLaunchGridAsync(f, grid_width, grid_height, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuParamSetTexRef(void *conn)
{
    CUfunction hfunc;
    int texunit;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hfunc, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texunit, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuParamSetTexRef(hfunc, texunit, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphCreate(void *conn)
{
    CUgraph phGraph;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphCreate(&phGraph, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphKernelNodeGetParams_v2(void *conn)
{
    CUgraphNode hNode;
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphKernelNodeGetParams_v2(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(CUDA_KERNEL_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphMemcpyNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_MEMCPY3D nodeParams;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_MEMCPY3D)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphMemcpyNodeGetParams(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphMemsetNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_MEMSET_NODE_PARAMS nodeParams;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphMemsetNodeGetParams(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphHostNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_HOST_NODE_PARAMS nodeParams;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphHostNodeGetParams(hNode, &nodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphChildGraphNodeGetGraph(void *conn)
{
    CUgraphNode hNode;
    CUgraph phGraph;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &phGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphChildGraphNodeGetGraph(hNode, &phGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phGraph, sizeof(CUgraph)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphEventRecordNodeGetEvent(void *conn)
{
    CUgraphNode hNode;
    CUevent event_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event_out, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphEventRecordNodeGetEvent(hNode, &event_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event_out, sizeof(CUevent)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphEventRecordNodeSetEvent(void *conn)
{
    CUgraphNode hNode;
    CUevent event;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphEventRecordNodeSetEvent(hNode, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphEventWaitNodeGetEvent(void *conn)
{
    CUgraphNode hNode;
    CUevent event_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event_out, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphEventWaitNodeGetEvent(hNode, &event_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event_out, sizeof(CUevent)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphEventWaitNodeSetEvent(void *conn)
{
    CUgraphNode hNode;
    CUevent event;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphEventWaitNodeSetEvent(hNode, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExternalSemaphoresSignalNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_EXT_SEM_SIGNAL_NODE_PARAMS params_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExternalSemaphoresSignalNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &params_out, sizeof(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExternalSemaphoresWaitNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_EXT_SEM_WAIT_NODE_PARAMS params_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExternalSemaphoresWaitNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &params_out, sizeof(CUDA_EXT_SEM_WAIT_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphBatchMemOpNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_BATCH_MEM_OP_NODE_PARAMS nodeParams_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphBatchMemOpNodeGetParams(hNode, &nodeParams_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodeParams_out, sizeof(CUDA_BATCH_MEM_OP_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphMemAllocNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUDA_MEM_ALLOC_NODE_PARAMS params_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphMemAllocNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphMemFreeNodeGetParams(void *conn)
{
    CUgraphNode hNode;
    CUdeviceptr dptr_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr_out, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphMemFreeNodeGetParams(hNode, &dptr_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr_out, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGraphMemTrim(void *conn)
{
    CUdevice device;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGraphMemTrim(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetGraphMemAttribute(void *conn)
{
    CUdevice device;
    CUgraphMem_attribute attr;
    void* value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(CUgraphMem_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceSetGraphMemAttribute(void *conn)
{
    CUdevice device;
    CUgraphMem_attribute attr;
    void* value;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &device, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(CUgraphMem_attribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceSetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphClone(void *conn)
{
    CUgraph phGraphClone;
    CUgraph originalGraph;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phGraphClone, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &originalGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphClone(&phGraphClone, originalGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phGraphClone, sizeof(CUgraph)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphNodeFindInClone(void *conn)
{
    CUgraphNode phNode;
    CUgraphNode hOriginalNode;
    CUgraph hClonedGraph;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOriginalNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hClonedGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphNodeFindInClone(&phNode, hOriginalNode, hClonedGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phNode, sizeof(CUgraphNode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphNodeGetType(void *conn)
{
    CUgraphNode hNode;
    CUgraphNodeType type;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &type, sizeof(CUgraphNodeType)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphNodeGetType(hNode, &type);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &type, sizeof(CUgraphNodeType)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphGetNodes(void *conn)
{
    CUgraph hGraph;
    CUgraphNode nodes;
    size_t numNodes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodes, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numNodes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphGetNodes(hGraph, &nodes, &numNodes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &numNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphGetRootNodes(void *conn)
{
    CUgraph hGraph;
    CUgraphNode rootNodes;
    size_t numRootNodes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &rootNodes, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numRootNodes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphGetRootNodes(hGraph, &rootNodes, &numRootNodes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &rootNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &numRootNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphGetEdges(void *conn)
{
    CUgraph hGraph;
    CUgraphNode from;
    CUgraphNode to;
    size_t numEdges;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &from, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &to, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numEdges, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphGetEdges(hGraph, &from, &to, &numEdges);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &from, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &to, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &numEdges, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphNodeGetDependencies(void *conn)
{
    CUgraphNode hNode;
    CUgraphNode dependencies;
    size_t numDependencies;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dependencies, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphNodeGetDependencies(hNode, &dependencies, &numDependencies);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dependencies, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphNodeGetDependentNodes(void *conn)
{
    CUgraphNode hNode;
    CUgraphNode dependentNodes;
    size_t numDependentNodes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dependentNodes, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numDependentNodes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphNodeGetDependentNodes(hNode, &dependentNodes, &numDependentNodes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dependentNodes, sizeof(CUgraphNode)) < 0 ||
        rpc_write(conn, &numDependentNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphDestroyNode(void *conn)
{
    CUgraphNode hNode;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphDestroyNode(hNode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphInstantiateWithFlags(void *conn)
{
    CUgraphExec phGraphExec;
    CUgraph hGraph;
    unsigned long long flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphInstantiateWithFlags(&phGraphExec, hGraph, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphInstantiateWithParams(void *conn)
{
    CUgraphExec phGraphExec;
    CUgraph hGraph;
    CUDA_GRAPH_INSTANTIATE_PARAMS instantiateParams;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphInstantiateWithParams(&phGraphExec, hGraph, &instantiateParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phGraphExec, sizeof(CUgraphExec)) < 0 ||
        rpc_write(conn, &instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExecGetFlags(void *conn)
{
    CUgraphExec hGraphExec;
    cuuint64_t flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(cuuint64_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExecGetFlags(hGraphExec, &flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(cuuint64_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExecChildGraphNodeSetParams(void *conn)
{
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    CUgraph childGraph;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &childGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExecEventRecordNodeSetEvent(void *conn)
{
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    CUevent event;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExecEventWaitNodeSetEvent(void *conn)
{
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    CUevent event;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(CUevent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphNodeSetEnabled(void *conn)
{
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    unsigned int isEnabled;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphNodeGetEnabled(void *conn)
{
    CUgraphExec hGraphExec;
    CUgraphNode hNode;
    unsigned int isEnabled;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphNodeGetEnabled(hGraphExec, hNode, &isEnabled);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphUpload(void *conn)
{
    CUgraphExec hGraphExec;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphUpload(hGraphExec, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphLaunch(void *conn)
{
    CUgraphExec hGraphExec;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphLaunch(hGraphExec, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExecDestroy(void *conn)
{
    CUgraphExec hGraphExec;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExecDestroy(hGraphExec);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphDestroy(void *conn)
{
    CUgraph hGraph;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphDestroy(hGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphExecUpdate_v2(void *conn)
{
    CUgraphExec hGraphExec;
    CUgraph hGraph;
    CUgraphExecUpdateResultInfo resultInfo;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hGraphExec, sizeof(CUgraphExec)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hGraph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphExecUpdate_v2(hGraphExec, hGraph, &resultInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphKernelNodeCopyAttributes(void *conn)
{
    CUgraphNode dst;
    CUgraphNode src;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dst, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphKernelNodeCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphKernelNodeGetAttribute(void *conn)
{
    CUgraphNode hNode;
    CUkernelNodeAttrID attr;
    CUkernelNodeAttrValue value_out;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hNode, sizeof(CUgraphNode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value_out, sizeof(CUkernelNodeAttrValue)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphKernelNodeGetAttribute(hNode, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuUserObjectCreate(void *conn)
{
    CUuserObject object_out;
    void* ptr;
    CUhostFn destroy;
    unsigned int initialRefcount;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &object_out, sizeof(CUuserObject)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &destroy, sizeof(CUhostFn)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &initialRefcount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuUserObjectCreate(&object_out, &ptr, destroy, initialRefcount, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &object_out, sizeof(CUuserObject)) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuUserObjectRetain(void *conn)
{
    CUuserObject object;
    unsigned int count;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuUserObjectRetain(object, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuUserObjectRelease(void *conn)
{
    CUuserObject object;
    unsigned int count;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuUserObjectRelease(object, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphRetainUserObject(void *conn)
{
    CUgraph graph;
    CUuserObject object;
    unsigned int count;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &graph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphRetainUserObject(graph, object, count, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphReleaseUserObject(void *conn)
{
    CUgraph graph;
    CUuserObject object;
    unsigned int count;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &graph, sizeof(CUgraph)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &object, sizeof(CUuserObject)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphReleaseUserObject(graph, object, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessor(void *conn)
{
    int numBlocks;
    CUfunction func;
    int blockSize;
    size_t dynamicSMemSize;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &numBlocks, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &func, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockSize, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func, blockSize, dynamicSMemSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(void *conn)
{
    int numBlocks;
    CUfunction func;
    int blockSize;
    size_t dynamicSMemSize;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &numBlocks, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &func, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockSize, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dynamicSMemSize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, func, blockSize, dynamicSMemSize, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuOccupancyAvailableDynamicSMemPerBlock(void *conn)
{
    size_t dynamicSmemSize;
    CUfunction func;
    int numBlocks;
    int blockSize;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &dynamicSmemSize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &func, sizeof(CUfunction)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numBlocks, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &blockSize, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, func, numBlocks, blockSize);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dynamicSmemSize, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetArray(void *conn)
{
    CUtexref hTexRef;
    CUarray hArray;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetArray(hTexRef, hArray, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetMipmappedArray(void *conn)
{
    CUtexref hTexRef;
    CUmipmappedArray hMipmappedArray;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetAddress_v2(void *conn)
{
    size_t ByteOffset;
    CUtexref hTexRef;
    CUdeviceptr dptr;
    size_t bytes;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &ByteOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bytes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetAddress_v2(&ByteOffset, hTexRef, dptr, bytes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ByteOffset, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetFormat(void *conn)
{
    CUtexref hTexRef;
    CUarray_format fmt;
    int NumPackedComponents;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fmt, sizeof(CUarray_format)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &NumPackedComponents, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetAddressMode(void *conn)
{
    CUtexref hTexRef;
    int dim;
    CUaddress_mode am;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dim, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &am, sizeof(CUaddress_mode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetAddressMode(hTexRef, dim, am);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetFilterMode(void *conn)
{
    CUtexref hTexRef;
    CUfilter_mode fm;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fm, sizeof(CUfilter_mode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetFilterMode(hTexRef, fm);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetMipmapFilterMode(void *conn)
{
    CUtexref hTexRef;
    CUfilter_mode fm;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fm, sizeof(CUfilter_mode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetMipmapFilterMode(hTexRef, fm);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetMipmapLevelBias(void *conn)
{
    CUtexref hTexRef;
    float bias;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &bias, sizeof(float)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetMipmapLevelBias(hTexRef, bias);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetMipmapLevelClamp(void *conn)
{
    CUtexref hTexRef;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &minMipmapLevelClamp, sizeof(float)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &maxMipmapLevelClamp, sizeof(float)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetMaxAnisotropy(void *conn)
{
    CUtexref hTexRef;
    unsigned int maxAniso;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &maxAniso, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetMaxAnisotropy(hTexRef, maxAniso);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetBorderColor(void *conn)
{
    CUtexref hTexRef;
    float pBorderColor;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pBorderColor, sizeof(float)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetBorderColor(hTexRef, &pBorderColor);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pBorderColor, sizeof(float)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefSetFlags(void *conn)
{
    CUtexref hTexRef;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefSetFlags(hTexRef, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetAddress_v2(void *conn)
{
    CUdeviceptr pdptr;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pdptr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetAddress_v2(&pdptr, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pdptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetArray(void *conn)
{
    CUarray phArray;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetArray(&phArray, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetMipmappedArray(void *conn)
{
    CUmipmappedArray phMipmappedArray;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetMipmappedArray(&phMipmappedArray, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetAddressMode(void *conn)
{
    CUaddress_mode pam;
    CUtexref hTexRef;
    int dim;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pam, sizeof(CUaddress_mode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dim, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetAddressMode(&pam, hTexRef, dim);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pam, sizeof(CUaddress_mode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetFilterMode(void *conn)
{
    CUfilter_mode pfm;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pfm, sizeof(CUfilter_mode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetFilterMode(&pfm, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetFormat(void *conn)
{
    CUarray_format pFormat;
    int pNumChannels;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pFormat, sizeof(CUarray_format)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNumChannels, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetFormat(&pFormat, &pNumChannels, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pFormat, sizeof(CUarray_format)) < 0 ||
        rpc_write(conn, &pNumChannels, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetMipmapFilterMode(void *conn)
{
    CUfilter_mode pfm;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pfm, sizeof(CUfilter_mode)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetMipmapFilterMode(&pfm, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pfm, sizeof(CUfilter_mode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetMipmapLevelBias(void *conn)
{
    float pbias;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pbias, sizeof(float)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetMipmapLevelBias(&pbias, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pbias, sizeof(float)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetMipmapLevelClamp(void *conn)
{
    float pminMipmapLevelClamp;
    float pmaxMipmapLevelClamp;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pminMipmapLevelClamp, sizeof(float)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pmaxMipmapLevelClamp, sizeof(float)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetMipmapLevelClamp(&pminMipmapLevelClamp, &pmaxMipmapLevelClamp, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pminMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_write(conn, &pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetMaxAnisotropy(void *conn)
{
    int pmaxAniso;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pmaxAniso, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetMaxAnisotropy(&pmaxAniso, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pmaxAniso, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetBorderColor(void *conn)
{
    float pBorderColor;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pBorderColor, sizeof(float)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetBorderColor(&pBorderColor, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pBorderColor, sizeof(float)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefGetFlags(void *conn)
{
    unsigned int pFlags;
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefGetFlags(&pFlags, hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefCreate(void *conn)
{
    CUtexref pTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefCreate(&pTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pTexRef, sizeof(CUtexref)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexRefDestroy(void *conn)
{
    CUtexref hTexRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hTexRef, sizeof(CUtexref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexRefDestroy(hTexRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuSurfRefSetArray(void *conn)
{
    CUsurfref hSurfRef;
    CUarray hArray;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &hSurfRef, sizeof(CUsurfref)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuSurfRefSetArray(hSurfRef, hArray, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuSurfRefGetArray(void *conn)
{
    CUarray phArray;
    CUsurfref hSurfRef;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &phArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hSurfRef, sizeof(CUsurfref)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuSurfRefGetArray(&phArray, hSurfRef);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &phArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexObjectDestroy(void *conn)
{
    CUtexObject texObject;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexObjectDestroy(texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexObjectGetResourceDesc(void *conn)
{
    CUDA_RESOURCE_DESC pResDesc;
    CUtexObject texObject;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexObjectGetResourceDesc(&pResDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexObjectGetTextureDesc(void *conn)
{
    CUDA_TEXTURE_DESC pTexDesc;
    CUtexObject texObject;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexObjectGetTextureDesc(&pTexDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTexObjectGetResourceViewDesc(void *conn)
{
    CUDA_RESOURCE_VIEW_DESC pResViewDesc;
    CUtexObject texObject;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texObject, sizeof(CUtexObject)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTexObjectGetResourceViewDesc(&pResViewDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuSurfObjectDestroy(void *conn)
{
    CUsurfObject surfObject;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &surfObject, sizeof(CUsurfObject)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuSurfObjectDestroy(surfObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuSurfObjectGetResourceDesc(void *conn)
{
    CUDA_RESOURCE_DESC pResDesc;
    CUsurfObject surfObject;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &surfObject, sizeof(CUsurfObject)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuSurfObjectGetResourceDesc(&pResDesc, surfObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuTensorMapReplaceAddress(void *conn)
{
    CUtensorMap tensorMap;
    void* globalAddress;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &tensorMap, sizeof(CUtensorMap)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &globalAddress, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuTensorMapReplaceAddress(&tensorMap, &globalAddress);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &tensorMap, sizeof(CUtensorMap)) < 0 ||
        rpc_write(conn, &globalAddress, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceCanAccessPeer(void *conn)
{
    int canAccessPeer;
    CUdevice dev;
    CUdevice peerDev;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &canAccessPeer, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &peerDev, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceCanAccessPeer(&canAccessPeer, dev, peerDev);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &canAccessPeer, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxEnablePeerAccess(void *conn)
{
    CUcontext peerContext;
    unsigned int Flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &peerContext, sizeof(CUcontext)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxEnablePeerAccess(peerContext, Flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuCtxDisablePeerAccess(void *conn)
{
    CUcontext peerContext;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &peerContext, sizeof(CUcontext)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuCtxDisablePeerAccess(peerContext);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuDeviceGetP2PAttribute(void *conn)
{
    int value;
    CUdevice_P2PAttribute attrib;
    CUdevice srcDevice;
    CUdevice dstDevice;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attrib, sizeof(CUdevice_P2PAttribute)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(CUdevice)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstDevice, sizeof(CUdevice)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuDeviceGetP2PAttribute(&value, attrib, srcDevice, dstDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsUnregisterResource(void *conn)
{
    CUgraphicsResource resource;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsUnregisterResource(resource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsSubResourceGetMappedArray(void *conn)
{
    CUarray pArray;
    CUgraphicsResource resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pArray, sizeof(CUarray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &arrayIndex, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipLevel, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsSubResourceGetMappedArray(&pArray, resource, arrayIndex, mipLevel);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pArray, sizeof(CUarray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsResourceGetMappedMipmappedArray(void *conn)
{
    CUmipmappedArray pMipmappedArray;
    CUgraphicsResource resource;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pMipmappedArray, sizeof(CUmipmappedArray)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsResourceGetMappedMipmappedArray(&pMipmappedArray, resource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsResourceGetMappedPointer_v2(void *conn)
{
    CUdeviceptr pDevPtr;
    size_t pSize;
    CUgraphicsResource resource;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &pDevPtr, sizeof(CUdeviceptr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pSize, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsResourceGetMappedPointer_v2(&pDevPtr, &pSize, resource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pDevPtr, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(conn, &pSize, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsResourceSetMapFlags_v2(void *conn)
{
    CUgraphicsResource resource;
    unsigned int flags;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &resource, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsResourceSetMapFlags_v2(resource, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsMapResources(void *conn)
{
    unsigned int count;
    CUgraphicsResource resources;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resources, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsMapResources(count, &resources, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cuGraphicsUnmapResources(void *conn)
{
    unsigned int count;
    CUgraphicsResource resources;
    CUstream hStream;
    int request_id;
    CUresult result;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resources, sizeof(CUgraphicsResource)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cuGraphicsUnmapResources(count, &resources, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &resources, sizeof(CUgraphicsResource)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceReset(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceReset();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceSynchronize(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceSynchronize();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceSetLimit(void *conn)
{
    enum cudaLimit limit;
    size_t value;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetLimit(void *conn)
{
    size_t pValue;
    enum cudaLimit limit;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pValue, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetLimit(&pValue, limit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pValue, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetCacheConfig(void *conn)
{
    enum cudaFuncCache pCacheConfig;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetCacheConfig(&pCacheConfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetStreamPriorityRange(void *conn)
{
    int leastPriority;
    int greatestPriority;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &leastPriority, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &greatestPriority, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &leastPriority, sizeof(int)) < 0 ||
        rpc_write(conn, &greatestPriority, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceSetCacheConfig(void *conn)
{
    enum cudaFuncCache cacheConfig;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &cacheConfig, sizeof(enum cudaFuncCache)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceSetCacheConfig(cacheConfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetSharedMemConfig(void *conn)
{
    enum cudaSharedMemConfig pConfig;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pConfig, sizeof(enum cudaSharedMemConfig)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetSharedMemConfig(&pConfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pConfig, sizeof(enum cudaSharedMemConfig)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceSetSharedMemConfig(void *conn)
{
    enum cudaSharedMemConfig config;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &config, sizeof(enum cudaSharedMemConfig)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceSetSharedMemConfig(config);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetPCIBusId(void *conn)
{
    char pciBusId;
    int len;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pciBusId, sizeof(char)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetPCIBusId(&pciBusId, len, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pciBusId, sizeof(char)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaIpcGetEventHandle(void *conn)
{
    cudaIpcEventHandle_t handle;
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &handle, sizeof(cudaIpcEventHandle_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaIpcGetEventHandle(&handle, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &handle, sizeof(cudaIpcEventHandle_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaIpcOpenEventHandle(void *conn)
{
    cudaEvent_t event;
    cudaIpcEventHandle_t handle;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(cudaIpcEventHandle_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaIpcOpenEventHandle(&event, handle);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaIpcGetMemHandle(void *conn)
{
    cudaIpcMemHandle_t handle;
    void* devPtr;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &handle, sizeof(cudaIpcMemHandle_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaIpcGetMemHandle(&handle, &devPtr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &handle, sizeof(cudaIpcMemHandle_t)) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaIpcOpenMemHandle(void *conn)
{
    void* devPtr;
    cudaIpcMemHandle_t handle;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handle, sizeof(cudaIpcMemHandle_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaIpcOpenMemHandle(&devPtr, handle, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaIpcCloseMemHandle(void *conn)
{
    void* devPtr;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaIpcCloseMemHandle(&devPtr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceFlushGPUDirectRDMAWrites(void *conn)
{
    enum cudaFlushGPUDirectRDMAWritesTarget target;
    enum cudaFlushGPUDirectRDMAWritesScope scope;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &target, sizeof(enum cudaFlushGPUDirectRDMAWritesTarget)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &scope, sizeof(enum cudaFlushGPUDirectRDMAWritesScope)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceFlushGPUDirectRDMAWrites(target, scope);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadExit(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadExit();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadSynchronize(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadSynchronize();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadSetLimit(void *conn)
{
    enum cudaLimit limit;
    size_t value;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadSetLimit(limit, value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadGetLimit(void *conn)
{
    size_t pValue;
    enum cudaLimit limit;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pValue, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &limit, sizeof(enum cudaLimit)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadGetLimit(&pValue, limit);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pValue, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadGetCacheConfig(void *conn)
{
    enum cudaFuncCache pCacheConfig;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadGetCacheConfig(&pCacheConfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pCacheConfig, sizeof(enum cudaFuncCache)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadSetCacheConfig(void *conn)
{
    enum cudaFuncCache cacheConfig;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &cacheConfig, sizeof(enum cudaFuncCache)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadSetCacheConfig(cacheConfig);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetLastError(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetLastError();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaPeekAtLastError(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaPeekAtLastError();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetDeviceCount(void *conn)
{
    int count;
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetDeviceCount(&count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &count, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetDeviceProperties_v2(void *conn)
{
    struct cudaDeviceProp prop;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetDeviceProperties_v2(&prop, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &prop, sizeof(struct cudaDeviceProp)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetAttribute(void *conn)
{
    int value;
    enum cudaDeviceAttr attr;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(enum cudaDeviceAttr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetAttribute(&value, attr, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetDefaultMemPool(void *conn)
{
    cudaMemPool_t memPool;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetDefaultMemPool(&memPool, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceSetMemPool(void *conn)
{
    int device;
    cudaMemPool_t memPool;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceSetMemPool(device, memPool);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetMemPool(void *conn)
{
    cudaMemPool_t memPool;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetMemPool(&memPool, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetNvSciSyncAttributes(void *conn)
{
    void* nvSciSyncAttrList;
    int device;
    int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &nvSciSyncAttrList, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetNvSciSyncAttributes(&nvSciSyncAttrList, device, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nvSciSyncAttrList, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetP2PAttribute(void *conn)
{
    int value;
    enum cudaDeviceP2PAttr attr;
    int srcDevice;
    int dstDevice;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(enum cudaDeviceP2PAttr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &srcDevice, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dstDevice, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetP2PAttribute(&value, attr, srcDevice, dstDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaInitDevice(void *conn)
{
    int device;
    unsigned int deviceFlags;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &deviceFlags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaInitDevice(device, deviceFlags, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaSetDevice(void *conn)
{
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaSetDevice(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetDevice(void *conn)
{
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetDevice(&device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaSetValidDevices(void *conn)
{
    int device_arr;
    int len;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device_arr, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &len, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaSetValidDevices(&device_arr, len);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &device_arr, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaSetDeviceFlags(void *conn)
{
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaSetDeviceFlags(flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetDeviceFlags(void *conn)
{
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetDeviceFlags(&flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamCreate(void *conn)
{
    cudaStream_t pStream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamCreate(&pStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamCreateWithFlags(void *conn)
{
    cudaStream_t pStream;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamCreateWithFlags(&pStream, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamCreateWithPriority(void *conn)
{
    cudaStream_t pStream;
    unsigned int flags;
    int priority;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamCreateWithPriority(&pStream, flags, priority);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pStream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamGetPriority(void *conn)
{
    cudaStream_t hStream;
    int priority;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &priority, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamGetPriority(hStream, &priority);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &priority, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamGetFlags(void *conn)
{
    cudaStream_t hStream;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamGetFlags(hStream, &flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamGetId(void *conn)
{
    cudaStream_t hStream;
    unsigned long long streamId;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &streamId, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamGetId(hStream, &streamId);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &streamId, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaCtxResetPersistingL2Cache(void *conn)
{
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaCtxResetPersistingL2Cache();

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamCopyAttributes(void *conn)
{
    cudaStream_t dst;
    cudaStream_t src;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamCopyAttributes(dst, src);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamGetAttribute(void *conn)
{
    cudaStream_t hStream;
    cudaLaunchAttributeID attr;
    cudaLaunchAttributeValue value_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(cudaLaunchAttributeID)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamGetAttribute(hStream, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamDestroy(void *conn)
{
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamDestroy(stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamWaitEvent(void *conn)
{
    cudaStream_t stream;
    cudaEvent_t event;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamWaitEvent(stream, event, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamAddCallback(void *conn)
{
    cudaStream_t stream;
    cudaStreamCallback_t callback;
    void* userData;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &callback, sizeof(cudaStreamCallback_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamAddCallback(stream, callback, &userData, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &userData, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamSynchronize(void *conn)
{
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamSynchronize(stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamQuery(void *conn)
{
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamQuery(stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamAttachMemAsync(void *conn)
{
    cudaStream_t stream;
    void* devPtr;
    size_t length;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &length, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamAttachMemAsync(stream, &devPtr, length, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamBeginCapture(void *conn)
{
    cudaStream_t stream;
    enum cudaStreamCaptureMode mode;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mode, sizeof(enum cudaStreamCaptureMode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamBeginCapture(stream, mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaThreadExchangeStreamCaptureMode(void *conn)
{
    enum cudaStreamCaptureMode mode;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &mode, sizeof(enum cudaStreamCaptureMode)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaThreadExchangeStreamCaptureMode(&mode);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mode, sizeof(enum cudaStreamCaptureMode)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamEndCapture(void *conn)
{
    cudaStream_t stream;
    cudaGraph_t pGraph;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamEndCapture(stream, &pGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamIsCapturing(void *conn)
{
    cudaStream_t stream;
    enum cudaStreamCaptureStatus pCaptureStatus;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaStreamIsCapturing(stream, &pCaptureStatus);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pCaptureStatus, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaStreamGetCaptureInfo_v2(void *conn)
{
    cudaStream_t stream;
    enum cudaStreamCaptureStatus captureStatus_out;
    unsigned long long id_out;
    cudaGraph_t graph_out;
    size_t numDependencies_out;
    const cudaGraphNode_t** dependencies_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    dependencies_out = (const cudaGraphNode_t**)malloc(numDependencies_out * sizeof(const cudaGraphNode_t*));

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cudaStreamGetCaptureInfo_v2(stream, &captureStatus_out, &id_out, &graph_out, dependencies_out, &numDependencies_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &captureStatus_out, sizeof(enum cudaStreamCaptureStatus)) < 0 ||
        rpc_write(conn, &id_out, sizeof(unsigned long long)) < 0 ||
        rpc_write(conn, &graph_out, sizeof(cudaGraph_t)) < 0 ||
        rpc_write(conn, &numDependencies_out, sizeof(size_t)) < 0 ||
        rpc_write(conn, dependencies_out, numDependencies_out * sizeof(const cudaGraphNode_t*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) dependencies_out);
ERROR_0:
    return -1;
}

int handle_cudaStreamUpdateCaptureDependencies(void *conn)
{
    cudaStream_t stream;
    size_t numDependencies;
    cudaGraphNode_t* dependencies;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numDependencies, sizeof(size_t)) < 0)
        goto ERROR_0;
    dependencies = (cudaGraphNode_t*)malloc(numDependencies * sizeof(cudaGraphNode_t));
    if (rpc_read(conn, &dependencies, sizeof(cudaGraphNode_t*)) < 0)
        goto ERROR_1;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_1;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;
    result = cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *) dependencies);
ERROR_0:
    return -1;
}

int handle_cudaEventCreate(void *conn)
{
    cudaEvent_t event;
    int request_id;
    cudaError_t result;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventCreate(&event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventCreateWithFlags(void *conn)
{
    cudaEvent_t event;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventCreateWithFlags(&event, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventRecord(void *conn)
{
    cudaEvent_t event;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventRecord(event, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventRecordWithFlags(void *conn)
{
    cudaEvent_t event;
    cudaStream_t stream;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventRecordWithFlags(event, stream, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventQuery(void *conn)
{
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventQuery(event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventSynchronize(void *conn)
{
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventSynchronize(event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventDestroy(void *conn)
{
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventDestroy(event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaEventElapsedTime(void *conn)
{
    float ms;
    cudaEvent_t start;
    cudaEvent_t end;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &start, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &end, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaEventElapsedTime(&ms, start, end);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ms, sizeof(float)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDestroyExternalMemory(void *conn)
{
    cudaExternalMemory_t extMem;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &extMem, sizeof(cudaExternalMemory_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDestroyExternalMemory(extMem);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDestroyExternalSemaphore(void *conn)
{
    cudaExternalSemaphore_t extSem;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &extSem, sizeof(cudaExternalSemaphore_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDestroyExternalSemaphore(extSem);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaLaunchCooperativeKernelMultiDevice(void *conn)
{
    struct cudaLaunchParams launchParamsList;
    unsigned int numDevices;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &launchParamsList, sizeof(struct cudaLaunchParams)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numDevices, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaLaunchCooperativeKernelMultiDevice(&launchParamsList, numDevices, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &launchParamsList, sizeof(struct cudaLaunchParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaSetDoubleForDevice(void *conn)
{
    double d;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &d, sizeof(double)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaSetDoubleForDevice(&d);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &d, sizeof(double)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaSetDoubleForHost(void *conn)
{
    double d;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &d, sizeof(double)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaSetDoubleForHost(&d);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &d, sizeof(double)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaLaunchHostFunc(void *conn)
{
    cudaStream_t stream;
    cudaHostFn_t fn;
    void* userData;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &fn, sizeof(cudaHostFn_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &userData, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaLaunchHostFunc(stream, fn, &userData);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &userData, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMallocManaged(void *conn)
{
    void* devPtr;
    size_t size;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMallocManaged(&devPtr, size, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMalloc(void *conn)
{
    void* devPtr;
    size_t size;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMalloc(&devPtr, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMallocHost(void *conn)
{
    void* ptr;
    size_t size;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMallocHost(&ptr, size);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMallocPitch(void *conn)
{
    void* devPtr;
    size_t pitch;
    size_t width;
    size_t height;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMallocPitch(&devPtr, &pitch, width, height);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_write(conn, &pitch, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaFreeHost(void *conn)
{
    void* ptr;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaFreeHost(&ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaFreeArray(void *conn)
{
    cudaArray_t array;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaFreeArray(array);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaFreeMipmappedArray(void *conn)
{
    cudaMipmappedArray_t mipmappedArray;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaFreeMipmappedArray(mipmappedArray);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaHostAlloc(void *conn)
{
    void* pHost;
    size_t size;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pHost, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaHostAlloc(&pHost, size, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pHost, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaHostRegister(void *conn)
{
    void* ptr;
    size_t size;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaHostRegister(&ptr, size, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaHostUnregister(void *conn)
{
    void* ptr;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaHostUnregister(&ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaHostGetDevicePointer(void *conn)
{
    void* pDevice;
    void* pHost;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pDevice, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pHost, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaHostGetDevicePointer(&pDevice, &pHost, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pDevice, sizeof(void*)) < 0 ||
        rpc_write(conn, &pHost, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaHostGetFlags(void *conn)
{
    unsigned int pFlags;
    void* pHost;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pFlags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pHost, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaHostGetFlags(&pFlags, &pHost);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pFlags, sizeof(unsigned int)) < 0 ||
        rpc_write(conn, &pHost, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMalloc3D(void *conn)
{
    struct cudaPitchedPtr pitchedDevPtr;
    struct cudaExtent extent;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMalloc3D(&pitchedDevPtr, extent);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetMipmappedArrayLevel(void *conn)
{
    cudaArray_t levelArray;
    cudaMipmappedArray_const_t mipmappedArray;
    unsigned int level;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &levelArray, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipmappedArray, sizeof(cudaMipmappedArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &level, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, level);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &levelArray, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemGetInfo(void *conn)
{
    size_t free;
    size_t total;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &free, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &total, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemGetInfo(&free, &total);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &free, sizeof(size_t)) < 0 ||
        rpc_write(conn, &total, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaArrayGetInfo(void *conn)
{
    struct cudaChannelFormatDesc desc;
    struct cudaExtent extent;
    unsigned int flags;
    cudaArray_t array;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaArrayGetInfo(&desc, &extent, &flags, array);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_write(conn, &extent, sizeof(struct cudaExtent)) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaArrayGetPlane(void *conn)
{
    cudaArray_t pPlaneArray;
    cudaArray_t hArray;
    unsigned int planeIdx;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pPlaneArray, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hArray, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &planeIdx, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaArrayGetPlane(&pPlaneArray, hArray, planeIdx);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pPlaneArray, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaArrayGetMemoryRequirements(void *conn)
{
    struct cudaArrayMemoryRequirements memoryRequirements;
    cudaArray_t array;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaArrayGetMemoryRequirements(&memoryRequirements, array, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMipmappedArrayGetMemoryRequirements(void *conn)
{
    struct cudaArrayMemoryRequirements memoryRequirements;
    cudaMipmappedArray_t mipmap;
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipmap, sizeof(cudaMipmappedArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMipmappedArrayGetMemoryRequirements(&memoryRequirements, mipmap, device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memoryRequirements, sizeof(struct cudaArrayMemoryRequirements)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaArrayGetSparseProperties(void *conn)
{
    struct cudaArraySparseProperties sparseProperties;
    cudaArray_t array;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaArrayGetSparseProperties(&sparseProperties, array);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMipmappedArrayGetSparseProperties(void *conn)
{
    struct cudaArraySparseProperties sparseProperties;
    cudaMipmappedArray_t mipmap;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipmap, sizeof(cudaMipmappedArray_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMipmappedArrayGetSparseProperties(&sparseProperties, mipmap);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &sparseProperties, sizeof(struct cudaArraySparseProperties)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpy2DFromArray(void *conn)
{
    void* dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dpitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemcpy2DFromArray(&dst, dpitch, src, wOffset, hOffset, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dst, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpy2DArrayToArray(void *conn)
{
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffsetDst, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffsetDst, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffsetSrc, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffsetSrc, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpy2DFromArrayAsync(void *conn)
{
    void* dst;
    size_t dpitch;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t width;
    size_t height;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dpitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemcpy2DFromArrayAsync(&dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dst, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemset(void *conn)
{
    void* devPtr;
    int value;
    size_t count;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemset(&devPtr, value, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemset2D(void *conn)
{
    void* devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemset2D(&devPtr, pitch, value, width, height);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemset3D(void *conn)
{
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemset3D(pitchedDevPtr, value, extent);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemsetAsync(void *conn)
{
    void* devPtr;
    int value;
    size_t count;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemsetAsync(&devPtr, value, count, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemset2DAsync(void *conn)
{
    void* devPtr;
    size_t pitch;
    int value;
    size_t width;
    size_t height;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pitch, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &width, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &height, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemset2DAsync(&devPtr, pitch, value, width, height, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemset3DAsync(void *conn)
{
    struct cudaPitchedPtr pitchedDevPtr;
    int value;
    struct cudaExtent extent;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pitchedDevPtr, sizeof(struct cudaPitchedPtr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &extent, sizeof(struct cudaExtent)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpyFromArray(void *conn)
{
    void* dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemcpyFromArray(&dst, src, wOffset, hOffset, count, kind);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dst, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpyArrayToArray(void *conn)
{
    cudaArray_t dst;
    size_t wOffsetDst;
    size_t hOffsetDst;
    cudaArray_const_t src;
    size_t wOffsetSrc;
    size_t hOffsetSrc;
    size_t count;
    enum cudaMemcpyKind kind;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffsetDst, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffsetDst, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffsetSrc, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffsetSrc, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpyFromArrayAsync(void *conn)
{
    void* dst;
    cudaArray_const_t src;
    size_t wOffset;
    size_t hOffset;
    size_t count;
    enum cudaMemcpyKind kind;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &src, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &wOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hOffset, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemcpyFromArrayAsync(&dst, src, wOffset, hOffset, count, kind, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dst, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMallocAsync(void *conn)
{
    void* devPtr;
    size_t size;
    cudaStream_t hStream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMallocAsync(&devPtr, size, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaFreeAsync(void *conn)
{
    void* devPtr;
    cudaStream_t hStream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hStream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaFreeAsync(&devPtr, hStream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolTrimTo(void *conn)
{
    cudaMemPool_t memPool;
    size_t minBytesToKeep;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &minBytesToKeep, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolTrimTo(memPool, minBytesToKeep);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolSetAttribute(void *conn)
{
    cudaMemPool_t memPool;
    enum cudaMemPoolAttr attr;
    void* value;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(enum cudaMemPoolAttr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolSetAttribute(memPool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolGetAttribute(void *conn)
{
    cudaMemPool_t memPool;
    enum cudaMemPoolAttr attr;
    void* value;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(enum cudaMemPoolAttr)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolGetAttribute(memPool, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolGetAccess(void *conn)
{
    enum cudaMemAccessFlags flags;
    cudaMemPool_t memPool;
    struct cudaMemLocation location;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &flags, sizeof(enum cudaMemAccessFlags)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &location, sizeof(struct cudaMemLocation)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolGetAccess(&flags, memPool, &location);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(enum cudaMemAccessFlags)) < 0 ||
        rpc_write(conn, &location, sizeof(struct cudaMemLocation)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolDestroy(void *conn)
{
    cudaMemPool_t memPool;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolDestroy(memPool);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMallocFromPoolAsync(void *conn)
{
    void* ptr;
    size_t size;
    cudaMemPool_t memPool;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMallocFromPoolAsync(&ptr, size, memPool, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolExportToShareableHandle(void *conn)
{
    void* shareableHandle;
    cudaMemPool_t memPool;
    enum cudaMemAllocationHandleType handleType;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &shareableHandle, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handleType, sizeof(enum cudaMemAllocationHandleType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolExportToShareableHandle(&shareableHandle, memPool, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &shareableHandle, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolImportFromShareableHandle(void *conn)
{
    cudaMemPool_t memPool;
    void* shareableHandle;
    enum cudaMemAllocationHandleType handleType;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &shareableHandle, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &handleType, sizeof(enum cudaMemAllocationHandleType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolImportFromShareableHandle(&memPool, &shareableHandle, handleType, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &memPool, sizeof(cudaMemPool_t)) < 0 ||
        rpc_write(conn, &shareableHandle, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolExportPointer(void *conn)
{
    struct cudaMemPoolPtrExportData exportData;
    void* ptr;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolExportPointer(&exportData, &ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemPoolImportPointer(void *conn)
{
    void* ptr;
    cudaMemPool_t memPool;
    struct cudaMemPoolPtrExportData exportData;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &memPool, sizeof(cudaMemPool_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaMemPoolImportPointer(&ptr, memPool, &exportData);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_write(conn, &exportData, sizeof(struct cudaMemPoolPtrExportData)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceCanAccessPeer(void *conn)
{
    int canAccessPeer;
    int device;
    int peerDevice;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &canAccessPeer, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &peerDevice, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &canAccessPeer, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceEnablePeerAccess(void *conn)
{
    int peerDevice;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &peerDevice, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceEnablePeerAccess(peerDevice, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceDisablePeerAccess(void *conn)
{
    int peerDevice;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &peerDevice, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceDisablePeerAccess(peerDevice);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsUnregisterResource(void *conn)
{
    cudaGraphicsResource_t resource;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsUnregisterResource(resource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsResourceSetMapFlags(void *conn)
{
    cudaGraphicsResource_t resource;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsResourceSetMapFlags(resource, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsMapResources(void *conn)
{
    int count;
    cudaGraphicsResource_t resources;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &count, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsMapResources(count, &resources, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsUnmapResources(void *conn)
{
    int count;
    cudaGraphicsResource_t resources;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &count, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsUnmapResources(count, &resources, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &resources, sizeof(cudaGraphicsResource_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsResourceGetMappedPointer(void *conn)
{
    void* devPtr;
    size_t size;
    cudaGraphicsResource_t resource;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &devPtr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsResourceGetMappedPointer(&devPtr, &size, resource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &devPtr, sizeof(void*)) < 0 ||
        rpc_write(conn, &size, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsSubResourceGetMappedArray(void *conn)
{
    cudaArray_t array;
    cudaGraphicsResource_t resource;
    unsigned int arrayIndex;
    unsigned int mipLevel;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &array, sizeof(cudaArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &arrayIndex, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &mipLevel, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsSubResourceGetMappedArray(&array, resource, arrayIndex, mipLevel);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &array, sizeof(cudaArray_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphicsResourceGetMappedMipmappedArray(void *conn)
{
    cudaMipmappedArray_t mipmappedArray;
    cudaGraphicsResource_t resource;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resource, sizeof(cudaGraphicsResource_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphicsResourceGetMappedMipmappedArray(&mipmappedArray, resource);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &mipmappedArray, sizeof(cudaMipmappedArray_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetChannelDesc(void *conn)
{
    struct cudaChannelFormatDesc desc;
    cudaArray_const_t array;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &array, sizeof(cudaArray_const_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetChannelDesc(&desc, array);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &desc, sizeof(struct cudaChannelFormatDesc)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDestroyTextureObject(void *conn)
{
    cudaTextureObject_t texObject;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDestroyTextureObject(texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetTextureObjectResourceDesc(void *conn)
{
    struct cudaResourceDesc pResDesc;
    cudaTextureObject_t texObject;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetTextureObjectResourceDesc(&pResDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetTextureObjectTextureDesc(void *conn)
{
    struct cudaTextureDesc pTexDesc;
    cudaTextureObject_t texObject;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pTexDesc, sizeof(struct cudaTextureDesc)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetTextureObjectTextureDesc(&pTexDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pTexDesc, sizeof(struct cudaTextureDesc)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetTextureObjectResourceViewDesc(void *conn)
{
    struct cudaResourceViewDesc pResViewDesc;
    cudaTextureObject_t texObject;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &texObject, sizeof(cudaTextureObject_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetTextureObjectResourceViewDesc(&pResViewDesc, texObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pResViewDesc, sizeof(struct cudaResourceViewDesc)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDestroySurfaceObject(void *conn)
{
    cudaSurfaceObject_t surfObject;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &surfObject, sizeof(cudaSurfaceObject_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDestroySurfaceObject(surfObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGetSurfaceObjectResourceDesc(void *conn)
{
    struct cudaResourceDesc pResDesc;
    cudaSurfaceObject_t surfObject;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &surfObject, sizeof(cudaSurfaceObject_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGetSurfaceObjectResourceDesc(&pResDesc, surfObject);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pResDesc, sizeof(struct cudaResourceDesc)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDriverGetVersion(void *conn)
{
    int driverVersion;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &driverVersion, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDriverGetVersion(&driverVersion);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &driverVersion, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaRuntimeGetVersion(void *conn)
{
    int runtimeVersion;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &runtimeVersion, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaRuntimeGetVersion(&runtimeVersion);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &runtimeVersion, sizeof(int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphCreate(void *conn)
{
    cudaGraph_t pGraph;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphCreate(&pGraph, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphKernelNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    struct cudaKernelNodeParams pNodeParams;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphKernelNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pNodeParams, sizeof(struct cudaKernelNodeParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphKernelNodeCopyAttributes(void *conn)
{
    cudaGraphNode_t hSrc;
    cudaGraphNode_t hDst;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hSrc, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hDst, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphKernelNodeCopyAttributes(hSrc, hDst);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphKernelNodeGetAttribute(void *conn)
{
    cudaGraphNode_t hNode;
    cudaLaunchAttributeID attr;
    cudaLaunchAttributeValue value_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(cudaLaunchAttributeID)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphKernelNodeGetAttribute(hNode, attr, &value_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value_out, sizeof(cudaLaunchAttributeValue)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphMemcpyNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    struct cudaMemcpy3DParms pNodeParams;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphMemcpyNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pNodeParams, sizeof(struct cudaMemcpy3DParms)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphMemsetNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    struct cudaMemsetParams pNodeParams;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaMemsetParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphMemsetNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pNodeParams, sizeof(struct cudaMemsetParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphHostNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    struct cudaHostNodeParams pNodeParams;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNodeParams, sizeof(struct cudaHostNodeParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphHostNodeGetParams(node, &pNodeParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pNodeParams, sizeof(struct cudaHostNodeParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphChildGraphNodeGetGraph(void *conn)
{
    cudaGraphNode_t node;
    cudaGraph_t pGraph;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphChildGraphNodeGetGraph(node, &pGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraph, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphEventRecordNodeGetEvent(void *conn)
{
    cudaGraphNode_t node;
    cudaEvent_t event_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event_out, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphEventRecordNodeGetEvent(node, &event_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphEventRecordNodeSetEvent(void *conn)
{
    cudaGraphNode_t node;
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphEventRecordNodeSetEvent(node, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphEventWaitNodeGetEvent(void *conn)
{
    cudaGraphNode_t node;
    cudaEvent_t event_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event_out, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphEventWaitNodeGetEvent(node, &event_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &event_out, sizeof(cudaEvent_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphEventWaitNodeSetEvent(void *conn)
{
    cudaGraphNode_t node;
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphEventWaitNodeSetEvent(node, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExternalSemaphoresSignalNodeGetParams(void *conn)
{
    cudaGraphNode_t hNode;
    struct cudaExternalSemaphoreSignalNodeParams params_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &params_out, sizeof(struct cudaExternalSemaphoreSignalNodeParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExternalSemaphoresWaitNodeGetParams(void *conn)
{
    cudaGraphNode_t hNode;
    struct cudaExternalSemaphoreWaitNodeParams params_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, &params_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &params_out, sizeof(struct cudaExternalSemaphoreWaitNodeParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphMemAllocNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    struct cudaMemAllocNodeParams params_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &params_out, sizeof(struct cudaMemAllocNodeParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphMemAllocNodeGetParams(node, &params_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &params_out, sizeof(struct cudaMemAllocNodeParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphMemFreeNodeGetParams(void *conn)
{
    cudaGraphNode_t node;
    void* dptr_out;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &dptr_out, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphMemFreeNodeGetParams(node, &dptr_out);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &dptr_out, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGraphMemTrim(void *conn)
{
    int device;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGraphMemTrim(device);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceGetGraphMemAttribute(void *conn)
{
    int device;
    enum cudaGraphMemAttributeType attr;
    void* value;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceGetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaDeviceSetGraphMemAttribute(void *conn)
{
    int device;
    enum cudaGraphMemAttributeType attr;
    void* value;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &device, sizeof(int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &attr, sizeof(enum cudaGraphMemAttributeType)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &value, sizeof(void*)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaDeviceSetGraphMemAttribute(device, attr, &value);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &value, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphClone(void *conn)
{
    cudaGraph_t pGraphClone;
    cudaGraph_t originalGraph;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pGraphClone, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &originalGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphClone(&pGraphClone, originalGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraphClone, sizeof(cudaGraph_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphNodeFindInClone(void *conn)
{
    cudaGraphNode_t pNode;
    cudaGraphNode_t originalNode;
    cudaGraph_t clonedGraph;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &originalNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &clonedGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphNodeFindInClone(&pNode, originalNode, clonedGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pNode, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphNodeGetType(void *conn)
{
    cudaGraphNode_t node;
    enum cudaGraphNodeType pType;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pType, sizeof(enum cudaGraphNodeType)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphNodeGetType(node, &pType);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pType, sizeof(enum cudaGraphNodeType)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphGetNodes(void *conn)
{
    cudaGraph_t graph;
    cudaGraphNode_t nodes;
    size_t numNodes;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &nodes, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numNodes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphGetNodes(graph, &nodes, &numNodes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &nodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(conn, &numNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphGetRootNodes(void *conn)
{
    cudaGraph_t graph;
    cudaGraphNode_t pRootNodes;
    size_t pNumRootNodes;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pRootNodes, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNumRootNodes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphGetRootNodes(graph, &pRootNodes, &pNumRootNodes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pRootNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(conn, &pNumRootNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphGetEdges(void *conn)
{
    cudaGraph_t graph;
    cudaGraphNode_t from;
    cudaGraphNode_t to;
    size_t numEdges;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &from, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &to, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &numEdges, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphGetEdges(graph, &from, &to, &numEdges);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &from, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(conn, &to, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(conn, &numEdges, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphNodeGetDependencies(void *conn)
{
    cudaGraphNode_t node;
    cudaGraphNode_t pDependencies;
    size_t pNumDependencies;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pDependencies, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNumDependencies, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphNodeGetDependencies(node, &pDependencies, &pNumDependencies);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pDependencies, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(conn, &pNumDependencies, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphNodeGetDependentNodes(void *conn)
{
    cudaGraphNode_t node;
    cudaGraphNode_t pDependentNodes;
    size_t pNumDependentNodes;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pDependentNodes, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &pNumDependentNodes, sizeof(size_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphNodeGetDependentNodes(node, &pDependentNodes, &pNumDependentNodes);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pDependentNodes, sizeof(cudaGraphNode_t)) < 0 ||
        rpc_write(conn, &pNumDependentNodes, sizeof(size_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphDestroyNode(void *conn)
{
    cudaGraphNode_t node;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphDestroyNode(node);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphInstantiate(void *conn)
{
    cudaGraphExec_t pGraphExec;
    cudaGraph_t graph;
    unsigned long long flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphInstantiate(&pGraphExec, graph, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphInstantiateWithFlags(void *conn)
{
    cudaGraphExec_t pGraphExec;
    cudaGraph_t graph;
    unsigned long long flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphInstantiateWithFlags(&pGraphExec, graph, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphInstantiateWithParams(void *conn)
{
    cudaGraphExec_t pGraphExec;
    cudaGraph_t graph;
    cudaGraphInstantiateParams instantiateParams;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphInstantiateWithParams(&pGraphExec, graph, &instantiateParams);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &pGraphExec, sizeof(cudaGraphExec_t)) < 0 ||
        rpc_write(conn, &instantiateParams, sizeof(cudaGraphInstantiateParams)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExecGetFlags(void *conn)
{
    cudaGraphExec_t graphExec;
    unsigned long long flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned long long)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExecGetFlags(graphExec, &flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExecChildGraphNodeSetParams(void *conn)
{
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t node;
    cudaGraph_t childGraph;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &node, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &childGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExecEventRecordNodeSetEvent(void *conn)
{
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExecEventWaitNodeSetEvent(void *conn)
{
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    cudaEvent_t event;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &event, sizeof(cudaEvent_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphNodeSetEnabled(void *conn)
{
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    unsigned int isEnabled;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphNodeGetEnabled(void *conn)
{
    cudaGraphExec_t hGraphExec;
    cudaGraphNode_t hNode;
    unsigned int isEnabled;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hNode, sizeof(cudaGraphNode_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &isEnabled, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphNodeGetEnabled(hGraphExec, hNode, &isEnabled);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExecUpdate(void *conn)
{
    cudaGraphExec_t hGraphExec;
    cudaGraph_t hGraph;
    cudaGraphExecUpdateResultInfo resultInfo;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &hGraphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &hGraph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &resultInfo, sizeof(cudaGraphExecUpdateResultInfo)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphUpload(void *conn)
{
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphUpload(graphExec, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphLaunch(void *conn)
{
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphLaunch(graphExec, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphExecDestroy(void *conn)
{
    cudaGraphExec_t graphExec;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graphExec, sizeof(cudaGraphExec_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphExecDestroy(graphExec);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphDestroy(void *conn)
{
    cudaGraph_t graph;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphDestroy(graph);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaUserObjectCreate(void *conn)
{
    cudaUserObject_t object_out;
    void* ptr;
    cudaHostFn_t destroy;
    unsigned int initialRefcount;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &object_out, sizeof(cudaUserObject_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &ptr, sizeof(void*)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &destroy, sizeof(cudaHostFn_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &initialRefcount, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaUserObjectCreate(&object_out, &ptr, destroy, initialRefcount, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &object_out, sizeof(cudaUserObject_t)) < 0 ||
        rpc_write(conn, &ptr, sizeof(void*)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaUserObjectRetain(void *conn)
{
    cudaUserObject_t object;
    unsigned int count;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaUserObjectRetain(object, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaUserObjectRelease(void *conn)
{
    cudaUserObject_t object;
    unsigned int count;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaUserObjectRelease(object, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphRetainUserObject(void *conn)
{
    cudaGraph_t graph;
    cudaUserObject_t object;
    unsigned int count;
    unsigned int flags;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphRetainUserObject(graph, object, count, flags);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaGraphReleaseUserObject(void *conn)
{
    cudaGraph_t graph;
    cudaUserObject_t object;
    unsigned int count;
    int request_id;
    cudaError_t result;
    if (rpc_read(conn, &graph, sizeof(cudaGraph_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &object, sizeof(cudaUserObject_t)) < 0)
        goto ERROR_0;
    if (rpc_read(conn, &count, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;
    result = cudaGraphReleaseUserObject(graph, object, count);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

static RequestHandler opHandlers[] = {
    handle___cudaRegisterVar,
    handle___cudaRegisterFunction,
    handle___cudaRegisterFatBinary,
    handle___cudaRegisterFatBinaryEnd,
    handle___cudaPushCallConfiguration,
    handle___cudaPopCallConfiguration,
    handle_nvmlInit_v2,
    handle_nvmlInitWithFlags,
    handle_nvmlShutdown,
    nullptr,
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
    nullptr,
    handle_nvmlGpuInstanceDestroy,
    handle_nvmlDeviceGetGpuInstances,
    handle_nvmlDeviceGetGpuInstanceById,
    handle_nvmlGpuInstanceGetInfo,
    handle_nvmlGpuInstanceGetComputeInstanceProfileInfo,
    handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV,
    handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity,
    handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements,
    handle_nvmlGpuInstanceCreateComputeInstance,
    nullptr,
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
    nullptr,
    nullptr,
    handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold,
    nullptr,
    nullptr,
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
    nullptr,
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
    nullptr,
    nullptr,
    nullptr,
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
    nullptr,
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
    nullptr,
    handle_cuMemcpyAtoH_v2,
    handle_cuMemcpyAtoA_v2,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    handle_cuMemcpyAsync,
    handle_cuMemcpyPeerAsync,
    handle_cuMemcpyDtoHAsync_v2,
    handle_cuMemcpyDtoDAsync_v2,
    nullptr,
    handle_cuMemcpyAtoHAsync_v2,
    nullptr,
    nullptr,
    nullptr,
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
    nullptr,
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
    nullptr,
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
    nullptr,
    nullptr,
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
    nullptr,
    nullptr,
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
    nullptr,
    nullptr,
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
    nullptr,
    handle_cudaDestroyExternalMemory,
    handle_cudaDestroyExternalSemaphore,
    handle_cudaLaunchKernel,
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
    handle_cudaMemcpy,
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
    nullptr,
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
    handle_cublasCreate_v2,
    handle_cublasDestroy_v2,
    handle_cublasSgemm_v2,
};

RequestHandler get_handler(const int op)
{
   if (op > (sizeof(opHandlers) / sizeof(opHandlers[0]))) {
       return NULL;
   }
   return opHandlers[op];
}
