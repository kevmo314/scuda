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
    size_t value;
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
    size_t size;
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

int handle_cuMemAllocHost_v2(void *conn)
{
    void* pp;
    if (rpc_read(conn, &pp, sizeof(void*)) < 0)
        return -1;
    size_t bytesize;
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
    size_t bytesize;
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
    size_t bytesize;
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
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t dstOffset;
    if (rpc_read(conn, &dstOffset, sizeof(size_t)) < 0)
        return -1;
    CUdeviceptr srcDevice;
    if (rpc_read(conn, &srcDevice, sizeof(CUdeviceptr)) < 0)
        return -1;
    size_t ByteCount;
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
    size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    size_t ByteCount;
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
    size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    size_t ByteCount;
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
    size_t dstOffset;
    if (rpc_read(conn, &dstOffset, sizeof(size_t)) < 0)
        return -1;
    CUarray srcArray;
    if (rpc_read(conn, &srcArray, sizeof(CUarray)) < 0)
        return -1;
    size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t ByteCount;
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
    size_t srcOffset;
    if (rpc_read(conn, &srcOffset, sizeof(size_t)) < 0)
        return -1;
    size_t ByteCount;
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
    size_t N;
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
    size_t N;
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
    size_t N;
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
    size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned char uc;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        return -1;
    size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    size_t Height;
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
    size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned short us;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        return -1;
    size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    size_t Height;
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
    size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned int ui;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        return -1;
    size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    size_t Height;
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
    size_t N;
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
    size_t N;
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
    size_t N;
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
    size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned char uc;
    if (rpc_read(conn, &uc, sizeof(unsigned char)) < 0)
        return -1;
    size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    size_t Height;
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
    size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned short us;
    if (rpc_read(conn, &us, sizeof(unsigned short)) < 0)
        return -1;
    size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    size_t Height;
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
    size_t dstPitch;
    if (rpc_read(conn, &dstPitch, sizeof(size_t)) < 0)
        return -1;
    unsigned int ui;
    if (rpc_read(conn, &ui, sizeof(unsigned int)) < 0)
        return -1;
    size_t Width;
    if (rpc_read(conn, &Width, sizeof(size_t)) < 0)
        return -1;
    size_t Height;
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
    size_t size;
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

int handle_cuMemAddressFree(void *conn)
{
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    size_t size;
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
    size_t size;
    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
        return -1;
    size_t offset;
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

int handle_cuMemUnmap(void *conn)
{
    CUdeviceptr ptr;
    if (rpc_read(conn, &ptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    size_t size;
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

int handle_cuMemPoolTrimTo(void *conn)
{
    CUmemoryPool pool;
    if (rpc_read(conn, &pool, sizeof(CUmemoryPool)) < 0)
        return -1;
    size_t minBytesToKeep;
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
    size_t count;
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
    size_t count;
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
    size_t dataSize;
    if (rpc_read(conn, &dataSize, sizeof(size_t)) < 0)
        return -1;
    CUmem_range_attribute attribute;
    if (rpc_read(conn, &attribute, sizeof(CUmem_range_attribute)) < 0)
        return -1;
    CUdeviceptr devPtr;
    if (rpc_read(conn, &devPtr, sizeof(CUdeviceptr)) < 0)
        return -1;
    size_t count;
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

int handle_cuStreamAttachMemAsync(void *conn)
{
    CUstream hStream;
    if (rpc_read(conn, &hStream, sizeof(CUstream)) < 0)
        return -1;
    CUdeviceptr dptr;
    if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0)
        return -1;
    size_t length;
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

static RequestHandler opHandlers[] = {
    handle_nvmlInit_v2,
    handle_nvmlInitWithFlags,
    handle_nvmlShutdown,
    handle_nvmlSystemGetDriverVersion,
    handle_nvmlSystemGetNVMLVersion,
    handle_nvmlSystemGetProcessName,
    handle_nvmlDeviceGetName,
    handle_nvmlDeviceGetSerial,
    handle_nvmlDeviceGetMemoryAffinity,
    handle_nvmlDeviceGetCpuAffinityWithinScope,
    handle_nvmlDeviceGetCpuAffinity,
    handle_nvmlDeviceSetCpuAffinity,
    handle_nvmlDeviceClearCpuAffinity,
    handle_nvmlDeviceGetUUID,
    handle_nvmlVgpuInstanceGetMdevUUID,
    handle_nvmlDeviceGetBoardPartNumber,
    handle_nvmlDeviceGetInforomVersion,
    handle_nvmlDeviceGetInforomImageVersion,
    handle_nvmlDeviceValidateInforom,
    handle_nvmlDeviceResetApplicationsClocks,
    handle_nvmlDeviceSetAutoBoostedClocksEnabled,
    handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled,
    handle_nvmlDeviceSetDefaultFanSpeed_v2,
    handle_nvmlDeviceSetFanControlPolicy,
    handle_nvmlDeviceGetVbiosVersion,
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
    handle_nvmlDeviceSetPowerManagementLimit,
    handle_nvmlDeviceSetGpuOperationMode,
    handle_nvmlDeviceSetAPIRestriction,
    handle_nvmlDeviceSetAccountingMode,
    handle_nvmlDeviceClearAccountingPids,
    handle_nvmlDeviceResetNvLinkErrorCounters,
    handle_nvmlDeviceFreezeNvLinkUtilizationCounter,
    handle_nvmlDeviceResetNvLinkUtilizationCounter,
    handle_nvmlDeviceRegisterEvents,
    handle_nvmlEventSetFree,
    handle_nvmlDeviceGetFieldValues,
    handle_nvmlDeviceClearFieldValues,
    handle_nvmlDeviceSetVirtualizationMode,
    handle_nvmlVgpuTypeGetLicense,
    handle_nvmlVgpuInstanceGetUUID,
    handle_nvmlVgpuInstanceGetVmDriverVersion,
    handle_nvmlVgpuInstanceSetEncoderCapacity,
    handle_nvmlVgpuInstanceClearAccountingPids,
    handle_nvmlGpuInstanceDestroy,
    handle_nvmlComputeInstanceDestroy,
    handle_nvmlDeviceSetFanSpeed_v2,
    handle_nvmlDeviceSetGpcClkVfOffset,
    handle_nvmlDeviceSetMemClkVfOffset,
    handle_nvmlDeviceGetSupportedPerformanceStates,
    handle_nvmlGpmSampleFree,
    handle_nvmlGpmSampleGet,
    handle_nvmlGpmMigSampleGet,
    handle_cuInit,
    handle_cuDeviceGetName,
    handle_cuDeviceGetUuid,
    handle_cuDeviceGetUuid_v2,
    handle_cuDeviceSetMemPool,
    handle_cuFlushGPUDirectRDMAWrites,
    handle_cuDevicePrimaryCtxRelease_v2,
    handle_cuDevicePrimaryCtxSetFlags_v2,
    handle_cuDevicePrimaryCtxReset_v2,
    handle_cuCtxDestroy_v2,
    handle_cuCtxPushCurrent_v2,
    handle_cuCtxSetCurrent,
    handle_cuCtxSynchronize,
    handle_cuCtxSetLimit,
    handle_cuCtxSetCacheConfig,
    handle_cuCtxSetSharedMemConfig,
    handle_cuCtxResetPersistingL2Cache,
    handle_cuCtxDetach,
    handle_cuModuleUnload,
    handle_cuLinkAddData_v2,
    handle_cuLinkAddFile_v2,
    handle_cuLinkDestroy,
    handle_cuLibraryUnload,
    handle_cuLibraryGetUnifiedFunction,
    handle_cuKernelSetAttribute,
    handle_cuKernelSetCacheConfig,
    handle_cuMemFree_v2,
    handle_cuMemAllocHost_v2,
    handle_cuMemFreeHost,
    handle_cuMemHostAlloc,
    handle_cuDeviceGetPCIBusId,
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
    handle_cuArrayDestroy,
    handle_cuMipmappedArrayDestroy,
    handle_cuMemGetHandleForAddressRange,
    handle_cuMemAddressFree,
    handle_cuMemRelease,
    handle_cuMemMap,
    handle_cuMemUnmap,
    handle_cuMemExportToShareableHandle,
    handle_cuMemFreeAsync,
    handle_cuMemPoolTrimTo,
    handle_cuMemPoolSetAttribute,
    handle_cuMemPoolGetAttribute,
    handle_cuMemPoolDestroy,
    handle_cuMemPoolExportToShareableHandle,
    handle_cuPointerGetAttribute,
    handle_cuMemPrefetchAsync,
    handle_cuMemAdvise,
    handle_cuMemRangeGetAttribute,
    handle_cuStreamWaitEvent,
    handle_cuStreamAddCallback,
    handle_cuStreamBeginCapture_v2,
    handle_cuStreamAttachMemAsync,
    handle_cuStreamQuery,
    handle_cuStreamSynchronize,
    handle_cuStreamDestroy_v2,
    handle_cuStreamCopyAttributes,
    handle_cuEventRecord,
    handle_cuEventRecordWithFlags,
    handle_cuEventQuery,
    handle_cuEventSynchronize,
    handle_cuEventDestroy_v2,
    handle_cuDestroyExternalMemory,
    handle_cuDestroyExternalSemaphore,
    handle_cuStreamWaitValue32_v2,
    handle_cuStreamWaitValue64_v2,
    handle_cuStreamWriteValue32_v2,
    handle_cuStreamWriteValue64_v2,
    handle_cuFuncSetAttribute,
    handle_cuFuncSetCacheConfig,
    handle_cuFuncSetSharedMemConfig,
    handle_cuLaunchKernel,
    handle_cuLaunchCooperativeKernel,
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
    handle_cuGraphEventRecordNodeSetEvent,
    handle_cuGraphEventWaitNodeSetEvent,
    handle_cuDeviceGraphMemTrim,
    handle_cuDeviceGetGraphMemAttribute,
    handle_cuDeviceSetGraphMemAttribute,
    handle_cuGraphDestroyNode,
    handle_cuGraphExecChildGraphNodeSetParams,
    handle_cuGraphExecEventRecordNodeSetEvent,
    handle_cuGraphExecEventWaitNodeSetEvent,
    handle_cuGraphNodeSetEnabled,
    handle_cuGraphUpload,
    handle_cuGraphLaunch,
    handle_cuGraphExecDestroy,
    handle_cuGraphDestroy,
    handle_cuGraphKernelNodeCopyAttributes,
    handle_cuUserObjectRetain,
    handle_cuUserObjectRelease,
    handle_cuGraphRetainUserObject,
    handle_cuGraphReleaseUserObject,
    handle_cuTexRefSetArray,
    handle_cuTexRefSetMipmappedArray,
    handle_cuTexRefSetFormat,
    handle_cuTexRefSetAddressMode,
    handle_cuTexRefSetFilterMode,
    handle_cuTexRefSetMipmapFilterMode,
    handle_cuTexRefSetMipmapLevelBias,
    handle_cuTexRefSetMipmapLevelClamp,
    handle_cuTexRefSetMaxAnisotropy,
    handle_cuTexRefSetFlags,
    handle_cuTexRefDestroy,
    handle_cuSurfRefSetArray,
    handle_cuTexObjectDestroy,
    handle_cuSurfObjectDestroy,
    handle_cuCtxEnablePeerAccess,
    handle_cuCtxDisablePeerAccess,
    handle_cuGraphicsUnregisterResource,
    handle_cuGraphicsResourceSetMapFlags_v2,
    handle_cudaDeviceReset,
};

RequestHandler get_handler(const int op)
{
    return opHandlers[op];
}
