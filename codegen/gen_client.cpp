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
        rpc_write(&options, sizeof(CUjit_option*)) < 0 ||
        rpc_write(&optionValues, sizeof(void**)) < 0 ||
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
        rpc_write(&options, sizeof(CUjit_option*)) < 0 ||
        rpc_write(&optionValues, sizeof(void**)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
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

std::unordered_map<std::string, void *> functionMap = {
    {"nvmlInit_v2", (void *)nvmlInit_v2},
    {"nvmlInitWithFlags", (void *)nvmlInitWithFlags},
    {"nvmlShutdown", (void *)nvmlShutdown},
    {"nvmlSystemGetDriverVersion", (void *)nvmlSystemGetDriverVersion},
    {"nvmlSystemGetNVMLVersion", (void *)nvmlSystemGetNVMLVersion},
    {"nvmlSystemGetProcessName", (void *)nvmlSystemGetProcessName},
    {"nvmlDeviceGetName", (void *)nvmlDeviceGetName},
    {"nvmlDeviceGetSerial", (void *)nvmlDeviceGetSerial},
    {"nvmlDeviceGetMemoryAffinity", (void *)nvmlDeviceGetMemoryAffinity},
    {"nvmlDeviceGetCpuAffinityWithinScope", (void *)nvmlDeviceGetCpuAffinityWithinScope},
    {"nvmlDeviceGetCpuAffinity", (void *)nvmlDeviceGetCpuAffinity},
    {"nvmlDeviceSetCpuAffinity", (void *)nvmlDeviceSetCpuAffinity},
    {"nvmlDeviceClearCpuAffinity", (void *)nvmlDeviceClearCpuAffinity},
    {"nvmlDeviceGetUUID", (void *)nvmlDeviceGetUUID},
    {"nvmlVgpuInstanceGetMdevUUID", (void *)nvmlVgpuInstanceGetMdevUUID},
    {"nvmlDeviceGetBoardPartNumber", (void *)nvmlDeviceGetBoardPartNumber},
    {"nvmlDeviceGetInforomVersion", (void *)nvmlDeviceGetInforomVersion},
    {"nvmlDeviceGetInforomImageVersion", (void *)nvmlDeviceGetInforomImageVersion},
    {"nvmlDeviceValidateInforom", (void *)nvmlDeviceValidateInforom},
    {"nvmlDeviceResetApplicationsClocks", (void *)nvmlDeviceResetApplicationsClocks},
    {"nvmlDeviceSetAutoBoostedClocksEnabled", (void *)nvmlDeviceSetAutoBoostedClocksEnabled},
    {"nvmlDeviceSetDefaultAutoBoostedClocksEnabled", (void *)nvmlDeviceSetDefaultAutoBoostedClocksEnabled},
    {"nvmlDeviceSetDefaultFanSpeed_v2", (void *)nvmlDeviceSetDefaultFanSpeed_v2},
    {"nvmlDeviceSetFanControlPolicy", (void *)nvmlDeviceSetFanControlPolicy},
    {"nvmlDeviceGetVbiosVersion", (void *)nvmlDeviceGetVbiosVersion},
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
    {"nvmlDeviceSetPowerManagementLimit", (void *)nvmlDeviceSetPowerManagementLimit},
    {"nvmlDeviceSetGpuOperationMode", (void *)nvmlDeviceSetGpuOperationMode},
    {"nvmlDeviceSetAPIRestriction", (void *)nvmlDeviceSetAPIRestriction},
    {"nvmlDeviceSetAccountingMode", (void *)nvmlDeviceSetAccountingMode},
    {"nvmlDeviceClearAccountingPids", (void *)nvmlDeviceClearAccountingPids},
    {"nvmlDeviceResetNvLinkErrorCounters", (void *)nvmlDeviceResetNvLinkErrorCounters},
    {"nvmlDeviceFreezeNvLinkUtilizationCounter", (void *)nvmlDeviceFreezeNvLinkUtilizationCounter},
    {"nvmlDeviceResetNvLinkUtilizationCounter", (void *)nvmlDeviceResetNvLinkUtilizationCounter},
    {"nvmlDeviceRegisterEvents", (void *)nvmlDeviceRegisterEvents},
    {"nvmlEventSetFree", (void *)nvmlEventSetFree},
    {"nvmlDeviceGetFieldValues", (void *)nvmlDeviceGetFieldValues},
    {"nvmlDeviceClearFieldValues", (void *)nvmlDeviceClearFieldValues},
    {"nvmlDeviceSetVirtualizationMode", (void *)nvmlDeviceSetVirtualizationMode},
    {"nvmlVgpuTypeGetLicense", (void *)nvmlVgpuTypeGetLicense},
    {"nvmlVgpuInstanceGetUUID", (void *)nvmlVgpuInstanceGetUUID},
    {"nvmlVgpuInstanceGetVmDriverVersion", (void *)nvmlVgpuInstanceGetVmDriverVersion},
    {"nvmlVgpuInstanceSetEncoderCapacity", (void *)nvmlVgpuInstanceSetEncoderCapacity},
    {"nvmlVgpuInstanceClearAccountingPids", (void *)nvmlVgpuInstanceClearAccountingPids},
    {"nvmlGpuInstanceDestroy", (void *)nvmlGpuInstanceDestroy},
    {"nvmlComputeInstanceDestroy", (void *)nvmlComputeInstanceDestroy},
    {"nvmlDeviceSetFanSpeed_v2", (void *)nvmlDeviceSetFanSpeed_v2},
    {"nvmlDeviceSetGpcClkVfOffset", (void *)nvmlDeviceSetGpcClkVfOffset},
    {"nvmlDeviceSetMemClkVfOffset", (void *)nvmlDeviceSetMemClkVfOffset},
    {"nvmlDeviceGetSupportedPerformanceStates", (void *)nvmlDeviceGetSupportedPerformanceStates},
    {"nvmlGpmSampleFree", (void *)nvmlGpmSampleFree},
    {"nvmlGpmSampleGet", (void *)nvmlGpmSampleGet},
    {"nvmlGpmMigSampleGet", (void *)nvmlGpmMigSampleGet},
    {"cuInit", (void *)cuInit},
    {"cuDeviceGetName", (void *)cuDeviceGetName},
    {"cuDeviceGetUuid", (void *)cuDeviceGetUuid},
    {"cuDeviceGetUuid_v2", (void *)cuDeviceGetUuid_v2},
    {"cuDeviceSetMemPool", (void *)cuDeviceSetMemPool},
    {"cuFlushGPUDirectRDMAWrites", (void *)cuFlushGPUDirectRDMAWrites},
    {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxReset_v2", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuCtxDestroy_v2", (void *)cuCtxDestroy_v2},
    {"cuCtxPushCurrent_v2", (void *)cuCtxPushCurrent_v2},
    {"cuCtxSetCurrent", (void *)cuCtxSetCurrent},
    {"cuCtxSynchronize", (void *)cuCtxSynchronize},
    {"cuCtxSetLimit", (void *)cuCtxSetLimit},
    {"cuCtxSetCacheConfig", (void *)cuCtxSetCacheConfig},
    {"cuCtxSetSharedMemConfig", (void *)cuCtxSetSharedMemConfig},
    {"cuCtxResetPersistingL2Cache", (void *)cuCtxResetPersistingL2Cache},
    {"cuCtxDetach", (void *)cuCtxDetach},
    {"cuModuleUnload", (void *)cuModuleUnload},
    {"cuLinkAddData_v2", (void *)cuLinkAddData_v2},
    {"cuLinkAddFile_v2", (void *)cuLinkAddFile_v2},
    {"cuLinkDestroy", (void *)cuLinkDestroy},
    {"cuLibraryUnload", (void *)cuLibraryUnload},
    {"cuLibraryGetUnifiedFunction", (void *)cuLibraryGetUnifiedFunction},
    {"cuKernelSetAttribute", (void *)cuKernelSetAttribute},
    {"cuKernelSetCacheConfig", (void *)cuKernelSetCacheConfig},
    {"cuMemFree_v2", (void *)cuMemFree_v2},
    {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
    {"cuMemFreeHost", (void *)cuMemFreeHost},
    {"cuMemHostAlloc", (void *)cuMemHostAlloc},
    {"cuDeviceGetPCIBusId", (void *)cuDeviceGetPCIBusId},
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
    {"cuArrayDestroy", (void *)cuArrayDestroy},
    {"cuMipmappedArrayDestroy", (void *)cuMipmappedArrayDestroy},
    {"cuMemGetHandleForAddressRange", (void *)cuMemGetHandleForAddressRange},
    {"cuMemAddressFree", (void *)cuMemAddressFree},
    {"cuMemRelease", (void *)cuMemRelease},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemUnmap", (void *)cuMemUnmap},
    {"cuMemExportToShareableHandle", (void *)cuMemExportToShareableHandle},
    {"cuMemFreeAsync", (void *)cuMemFreeAsync},
    {"cuMemPoolTrimTo", (void *)cuMemPoolTrimTo},
    {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
    {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
    {"cuMemPoolDestroy", (void *)cuMemPoolDestroy},
    {"cuMemPoolExportToShareableHandle", (void *)cuMemPoolExportToShareableHandle},
    {"cuPointerGetAttribute", (void *)cuPointerGetAttribute},
    {"cuMemPrefetchAsync", (void *)cuMemPrefetchAsync},
    {"cuMemAdvise", (void *)cuMemAdvise},
    {"cuMemRangeGetAttribute", (void *)cuMemRangeGetAttribute},
    {"cuStreamWaitEvent", (void *)cuStreamWaitEvent},
    {"cuStreamAddCallback", (void *)cuStreamAddCallback},
    {"cuStreamBeginCapture_v2", (void *)cuStreamBeginCapture_v2},
    {"cuStreamAttachMemAsync", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery", (void *)cuStreamQuery},
    {"cuStreamSynchronize", (void *)cuStreamSynchronize},
    {"cuStreamDestroy_v2", (void *)cuStreamDestroy_v2},
    {"cuStreamCopyAttributes", (void *)cuStreamCopyAttributes},
    {"cuEventRecord", (void *)cuEventRecord},
    {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
    {"cuEventQuery", (void *)cuEventQuery},
    {"cuEventSynchronize", (void *)cuEventSynchronize},
    {"cuEventDestroy_v2", (void *)cuEventDestroy_v2},
    {"cuDestroyExternalMemory", (void *)cuDestroyExternalMemory},
    {"cuDestroyExternalSemaphore", (void *)cuDestroyExternalSemaphore},
    {"cuStreamWaitValue32_v2", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWaitValue64_v2", (void *)cuStreamWaitValue64_v2},
    {"cuStreamWriteValue32_v2", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWriteValue64_v2", (void *)cuStreamWriteValue64_v2},
    {"cuFuncSetAttribute", (void *)cuFuncSetAttribute},
    {"cuFuncSetCacheConfig", (void *)cuFuncSetCacheConfig},
    {"cuFuncSetSharedMemConfig", (void *)cuFuncSetSharedMemConfig},
    {"cuLaunchKernel", (void *)cuLaunchKernel},
    {"cuLaunchCooperativeKernel", (void *)cuLaunchCooperativeKernel},
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
    {"cuGraphEventRecordNodeSetEvent", (void *)cuGraphEventRecordNodeSetEvent},
    {"cuGraphEventWaitNodeSetEvent", (void *)cuGraphEventWaitNodeSetEvent},
    {"cuDeviceGraphMemTrim", (void *)cuDeviceGraphMemTrim},
    {"cuDeviceGetGraphMemAttribute", (void *)cuDeviceGetGraphMemAttribute},
    {"cuDeviceSetGraphMemAttribute", (void *)cuDeviceSetGraphMemAttribute},
    {"cuGraphDestroyNode", (void *)cuGraphDestroyNode},
    {"cuGraphExecChildGraphNodeSetParams", (void *)cuGraphExecChildGraphNodeSetParams},
    {"cuGraphExecEventRecordNodeSetEvent", (void *)cuGraphExecEventRecordNodeSetEvent},
    {"cuGraphExecEventWaitNodeSetEvent", (void *)cuGraphExecEventWaitNodeSetEvent},
    {"cuGraphNodeSetEnabled", (void *)cuGraphNodeSetEnabled},
    {"cuGraphUpload", (void *)cuGraphUpload},
    {"cuGraphLaunch", (void *)cuGraphLaunch},
    {"cuGraphExecDestroy", (void *)cuGraphExecDestroy},
    {"cuGraphDestroy", (void *)cuGraphDestroy},
    {"cuGraphKernelNodeCopyAttributes", (void *)cuGraphKernelNodeCopyAttributes},
    {"cuUserObjectRetain", (void *)cuUserObjectRetain},
    {"cuUserObjectRelease", (void *)cuUserObjectRelease},
    {"cuGraphRetainUserObject", (void *)cuGraphRetainUserObject},
    {"cuGraphReleaseUserObject", (void *)cuGraphReleaseUserObject},
    {"cuTexRefSetArray", (void *)cuTexRefSetArray},
    {"cuTexRefSetMipmappedArray", (void *)cuTexRefSetMipmappedArray},
    {"cuTexRefSetFormat", (void *)cuTexRefSetFormat},
    {"cuTexRefSetAddressMode", (void *)cuTexRefSetAddressMode},
    {"cuTexRefSetFilterMode", (void *)cuTexRefSetFilterMode},
    {"cuTexRefSetMipmapFilterMode", (void *)cuTexRefSetMipmapFilterMode},
    {"cuTexRefSetMipmapLevelBias", (void *)cuTexRefSetMipmapLevelBias},
    {"cuTexRefSetMipmapLevelClamp", (void *)cuTexRefSetMipmapLevelClamp},
    {"cuTexRefSetMaxAnisotropy", (void *)cuTexRefSetMaxAnisotropy},
    {"cuTexRefSetFlags", (void *)cuTexRefSetFlags},
    {"cuTexRefDestroy", (void *)cuTexRefDestroy},
    {"cuSurfRefSetArray", (void *)cuSurfRefSetArray},
    {"cuTexObjectDestroy", (void *)cuTexObjectDestroy},
    {"cuSurfObjectDestroy", (void *)cuSurfObjectDestroy},
    {"cuCtxEnablePeerAccess", (void *)cuCtxEnablePeerAccess},
    {"cuCtxDisablePeerAccess", (void *)cuCtxDisablePeerAccess},
    {"cuGraphicsUnregisterResource", (void *)cuGraphicsUnregisterResource},
    {"cuGraphicsResourceSetMapFlags_v2", (void *)cuGraphicsResourceSetMapFlags_v2},
    {"cudaDeviceReset", (void *)cudaDeviceReset},
};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it == functionMap.end())
        return nullptr;
    return it->second;
}
