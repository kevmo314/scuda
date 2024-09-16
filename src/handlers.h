// handlers.h
#ifndef HANDLERS_H
#define HANDLERS_H

#include <nvml.h>

// Function declarations (prototypes)

// 4.11 Initialization and Cleanup
nvmlReturn_t nvmlInitWithFlags(unsigned int flags);
nvmlReturn_t nvmlInit_v2();
nvmlReturn_t nvmlShutdown();

// 4.14 System Queries
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length);
nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries);
nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length);
nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length);
nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray);

// 4.15 Unit Queries
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount);
nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices);
nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds);
nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit);
nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state);
nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu);
nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp);
nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info);

// Device Queries
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);

// Function to get the function by its name
void *getFunctionByName(const char *name);

int rpc_start_request(const unsigned int op);
int rpc_write(const void *data, size_t size);
int rpc_read(void *data, size_t size);
int rpc_wait_for_response(int request_id);
int open_rpc_client();
void close_rpc_client();
nvmlReturn_t rpc_get_return(int request_id);

#endif // HANDLERS_H
