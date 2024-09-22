#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <nvml.h>
#include <cuda.h>
#include <future>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "api.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

typedef int (*RequestHandler)(int connfd);

int handle_nvmlInitWithFlags(int connfd) {
    unsigned int flags;
    if (read(connfd, &flags, sizeof(unsigned int)) < 0)
        return -1;
    return nvmlInitWithFlags(flags);
}

int handle_nvmlInit_v2(int connfd) {
    return nvmlInit_v2();
}

int handle_nvmlShutdown(int connfd) {
    return nvmlShutdown();
}

int handle_nvmlSystemGetDriverVersion(int connfd) {
    unsigned int length;
    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char version[length];
    nvmlReturn_t result = nvmlSystemGetDriverVersion(version, length);
    if (write(connfd, version, length) < 0)
        return -1;
    return result;
}

int handle_nvmlSystemGetHicVersion(int connfd) {
    unsigned int hwbcCount;
    if (read(connfd, &hwbcCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlHwbcEntry_t *hwbcEntries = (nvmlHwbcEntry_t *)malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));
    nvmlReturn_t result = nvmlSystemGetHicVersion(&hwbcCount, hwbcEntries);
    if (write(connfd, &hwbcCount, sizeof(unsigned int)) < 0 ||
        write(connfd, hwbcEntries, hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
        return -1;
    free(hwbcEntries);
    return result;
}

int handle_nvmlSystemGetNVMLVersion(int connfd) {
    unsigned int length;
    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char version[length];
    nvmlReturn_t result = nvmlSystemGetNVMLVersion(version, length);
    if (write(connfd, version, length) < 0)
        return -1;
    return result;
}

int handle_nvmlSystemGetProcessName(int connfd) {
    unsigned int pid;
    unsigned int length;
    if (read(connfd, &pid, sizeof(unsigned int)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char name[length];
    nvmlReturn_t result = nvmlSystemGetProcessName(pid, name, length);
    if (write(connfd, name, length) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetCount(int connfd) {
    unsigned int unitCount;
    nvmlReturn_t result = nvmlUnitGetCount(&unitCount);
    if (write(connfd, &unitCount, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetDevices(int connfd) {
    nvmlUnit_t unit;
    unsigned int deviceCount;
    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlDevice_t *devices = (nvmlDevice_t *)malloc(deviceCount * sizeof(nvmlDevice_t));
    nvmlReturn_t result = nvmlUnitGetDevices(unit, &deviceCount, devices);
    if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0 ||
        write(connfd, devices, deviceCount * sizeof(nvmlDevice_t)) < 0)
        return -1;
    free(devices);
    return result;
}

int handle_nvmlUnitGetFanSpeedInfo(int connfd) {
    nvmlUnit_t unit;
    nvmlUnitFanSpeeds_t fanSpeeds;
    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds);
    if (write(connfd, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetHandleByIndex(int connfd) {
    unsigned int index;
    nvmlUnit_t unit;
    if (read(connfd, &index, sizeof(unsigned int)) < 0)
        return -1;
    nvmlReturn_t result = nvmlUnitGetHandleByIndex(index, &unit);
    if (write(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetLedState(int connfd) {
    nvmlUnit_t unit;
    nvmlLedState_t state;
    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlUnitGetLedState(unit, &state);
    if (write(connfd, &state, sizeof(nvmlLedState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetPsuInfo(int connfd) {
    nvmlUnit_t unit;
    nvmlPSUInfo_t psu;
    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlUnitGetPsuInfo(unit, &psu);
    if (write(connfd, &psu, sizeof(nvmlPSUInfo_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetTemperature(int connfd) {
    nvmlUnit_t unit;
    unsigned int type;
    unsigned int temp;
    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &type, sizeof(unsigned int)) < 0)
        return -1;
    nvmlReturn_t result = nvmlUnitGetTemperature(unit, type, &temp);
    if (write(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlUnitGetUnitInfo(int connfd) {
    nvmlUnit_t unit;
    nvmlUnitInfo_t info;
    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlUnitGetUnitInfo(unit, &info);
    if (write(connfd, &info, sizeof(nvmlUnitInfo_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetAPIRestriction(int connfd) {
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);
    if (write(connfd, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetAdaptiveClockInfoStatus(int connfd) {
    nvmlDevice_t device;
    unsigned int adaptiveClockStatus;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);
    if (write(connfd, &adaptiveClockStatus, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetApplicationsClock(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);
    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetArchitecture(int connfd) {
    nvmlDevice_t device;
    nvmlDeviceArchitecture_t arch;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetArchitecture(device, &arch);
    if (write(connfd, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetAttributes_v2(int connfd) {
    nvmlDevice_t device;
    nvmlDeviceAttributes_t attributes;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetAttributes_v2(device, &attributes);
    if (write(connfd, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t isEnabled, defaultIsEnabled;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);
    if (write(connfd, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        write(connfd, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(int connfd) {
    nvmlDevice_t device;
    nvmlBAR1Memory_t bar1Memory;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory);
    if (write(connfd, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetBoardId(int connfd) {
    nvmlDevice_t device;
    unsigned int boardId;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetBoardId(device, &boardId);
    if (write(connfd, &boardId, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetBoardPartNumber(int connfd) {
    nvmlDevice_t device;
    unsigned int length;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char partNumber[length];
    nvmlReturn_t result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);
    if (write(connfd, partNumber, length) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetBrand(int connfd) {
    nvmlDevice_t device;
    nvmlBrandType_t type;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetBrand(device, &type);
    if (write(connfd, &type, sizeof(nvmlBrandType_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetBridgeChipInfo(int connfd) {
    nvmlDevice_t device;
    nvmlBridgeChipHierarchy_t bridgeHierarchy;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy);
    if (write(connfd, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetBusType(int connfd) {
    nvmlDevice_t device;
    nvmlBusType_t type;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetBusType(device, &type);
    if (write(connfd, &type, sizeof(nvmlBusType_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetClkMonStatus(int connfd) {
    nvmlDevice_t device;
    nvmlClkMonStatus_t status;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetClkMonStatus(device, &status);
    if (write(connfd, &status, sizeof(nvmlClkMonStatus_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetClock(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    nvmlClockId_t clockId;
    unsigned int clockMHz;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clockId, sizeof(nvmlClockId_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);
    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetClockInfo(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int clock;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetClockInfo(device, type, &clock);
    if (write(connfd, &clock, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetComputeMode(int connfd) {
    nvmlDevice_t device;
    nvmlComputeMode_t mode;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetComputeMode(device, &mode);
    if (write(connfd, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(int connfd) {
    nvmlDevice_t device;
    unsigned int infoCount;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);
    if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 || write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;
    free(infos);
    return result;
}

int handle_nvmlDeviceGetCount_v2(int connfd) {
    unsigned int deviceCount;
    nvmlReturn_t result = nvmlDeviceGetCount_v2(&deviceCount);
    if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetCudaComputeCapability(int connfd) {
    nvmlDevice_t device;
    int major, minor;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
    if (write(connfd, &major, sizeof(int)) < 0 || write(connfd, &minor, sizeof(int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(int connfd) {
    nvmlDevice_t device;
    unsigned int currLinkGen;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);
    if (write(connfd, &currLinkGen, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(int connfd) {
    nvmlDevice_t device;
    unsigned int currLinkWidth;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);
    if (write(connfd, &currLinkWidth, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetCurrentClocksThrottleReasons(int connfd) {
    nvmlDevice_t device;
    unsigned long long clocksThrottleReasons;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clocksThrottleReasons);
    if (write(connfd, &clocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDecoderUtilization(int connfd) {
    nvmlDevice_t device;
    unsigned int utilization, samplingPeriodUs;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);
    if (write(connfd, &utilization, sizeof(unsigned int)) < 0 || write(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDefaultApplicationsClock(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);
    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDefaultEccMode(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t defaultMode;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDefaultEccMode(device, &defaultMode);
    if (write(connfd, &defaultMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDetailedEccErrors(int connfd) {
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlEccErrorCounts_t eccCounts;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 || read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, &eccCounts);
    if (write(connfd, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDisplayActive(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t isActive;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDisplayActive(device, &isActive);
    if (write(connfd, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDisplayMode(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t display;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDisplayMode(device, &display);
    if (write(connfd, &display, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetDynamicPstatesInfo(int connfd) {
    nvmlDevice_t device;
    nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);
    if (write(connfd, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetEccMode(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t current, pending;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetEccMode(device, &current, &pending);
    if (write(connfd, &current, sizeof(nvmlEnableState_t)) < 0 || write(connfd, &pending, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetEncoderCapacity(int connfd) {
    nvmlDevice_t device;
    nvmlEncoderType_t encoderQueryType;
    unsigned int encoderCapacity;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, &encoderCapacity);
    if (write(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetEncoderSessions(int connfd) {
    nvmlDevice_t device;
    unsigned int sessionCount;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlEncoderSessionInfo_t *sessionInfos = (nvmlEncoderSessionInfo_t *)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));
    nvmlReturn_t result = nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);
    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 || write(connfd, sessionInfos, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;
    free(sessionInfos);
    return result;
}

int handle_nvmlDeviceGetEncoderStats(int connfd) {
    nvmlDevice_t device;
    unsigned int sessionCount, averageFps, averageLatency;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);
    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 || write(connfd, &averageFps, sizeof(unsigned int)) < 0 || write(connfd, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetEncoderUtilization(int connfd) {
    nvmlDevice_t device;
    unsigned int utilization, samplingPeriodUs;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetEncoderUtilization(device, &utilization, &samplingPeriodUs);
    if (write(connfd, &utilization, sizeof(unsigned int)) < 0 || write(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetEnforcedPowerLimit(int connfd) {
    nvmlDevice_t device;
    unsigned int limit;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetEnforcedPowerLimit(device, &limit);
    if (write(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetFBCSessions(int connfd) {
    nvmlDevice_t device;
    unsigned int sessionCount;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlFBCSessionInfo_t *sessionInfo = (nvmlFBCSessionInfo_t *)malloc(sessionCount * sizeof(nvmlFBCSessionInfo_t));
    nvmlReturn_t result = nvmlDeviceGetFBCSessions(device, &sessionCount, sessionInfo);
    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 || write(connfd, sessionInfo, sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;
    free(sessionInfo);
    return result;
}

int handle_nvmlDeviceGetFBCStats(int connfd) {
    nvmlDevice_t device;
    nvmlFBCStats_t fbcStats;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetFBCStats(device, &fbcStats);
    if (write(connfd, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(int connfd) {
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &fan, sizeof(unsigned int)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetFanControlPolicy_v2(device, fan, &policy);
    if (write(connfd, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetFanSpeed(int connfd) {
    nvmlDevice_t device;
    unsigned int speed;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetFanSpeed(device, &speed);
    if (write(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetFanSpeed_v2(int connfd) {
    nvmlDevice_t device;
    unsigned int fan, speed;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &fan, sizeof(unsigned int)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetFanSpeed_v2(device, fan, &speed);
    if (write(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(int connfd) {
    nvmlDevice_t device;
    int minOffset, maxOffset;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);
    if (write(connfd, &minOffset, sizeof(int)) < 0 || write(connfd, &maxOffset, sizeof(int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGpcClkVfOffset(int connfd) {
    nvmlDevice_t device;
    int offset;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetGpcClkVfOffset(device, &offset);
    if (write(connfd, &offset, sizeof(int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGpuFabricInfo(int connfd) {
    nvmlDevice_t device;
    nvmlGpuFabricInfo_t gpuFabricInfo;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo);
    if (write(connfd, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGpuMaxPcieLinkGeneration(int connfd) {
    nvmlDevice_t device;
    unsigned int maxLinkGenDevice;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &maxLinkGenDevice);
    if (write(connfd, &maxLinkGenDevice, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGpuOperationMode(int connfd) {
    nvmlDevice_t device;
    nvmlGpuOperationMode_t current, pending;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetGpuOperationMode(device, &current, &pending);
    if (write(connfd, &current, sizeof(nvmlGpuOperationMode_t)) < 0 || write(connfd, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(int connfd) {
    nvmlDevice_t device;
    unsigned int infoCount;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
    nvmlReturn_t result = nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, infos);
    if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 || write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;
    free(infos);
    return result;
}

int handle_nvmlDeviceGetGspFirmwareMode(int connfd) {
    nvmlDevice_t device;
    unsigned int isEnabled, defaultMode;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);
    if (write(connfd, &isEnabled, sizeof(unsigned int)) < 0 || write(connfd, &defaultMode, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetGspFirmwareVersion(int connfd) {
    nvmlDevice_t device;
    unsigned int length;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char version[length];
    nvmlReturn_t result = nvmlDeviceGetGspFirmwareVersion(device, version);
    if (write(connfd, version, length) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetHandleByIndex_v2(int connfd) {
    unsigned int index;
    nvmlDevice_t device;
    if (read(connfd, &index, sizeof(unsigned int)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(index, &device);
    if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(int connfd) {
    unsigned int length;
    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char pciBusId[length];
    nvmlDevice_t device;
    if (read(connfd, pciBusId, length) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device);
    if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetHandleBySerial(int connfd) {
    unsigned int length;
    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char serial[length];
    nvmlDevice_t device;
    if (read(connfd, serial, length) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetHandleBySerial(serial, &device);
    if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetHandleByUUID(int connfd) {
    unsigned int length;
    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char uuid[length];
    nvmlDevice_t device;
    if (read(connfd, uuid, length) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetHandleByUUID(uuid, &device);
    if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetIndex(int connfd) {
    nvmlDevice_t device;
    unsigned int index;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetIndex(device, &index);
    if (write(connfd, &index, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetInforomConfigurationChecksum(int connfd) {
    nvmlDevice_t device;
    unsigned int checksum;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetInforomConfigurationChecksum(device, &checksum);
    if (write(connfd, &checksum, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetInforomImageVersion(int connfd) {
    nvmlDevice_t device;
    unsigned int length;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char version[length];
    nvmlReturn_t result = nvmlDeviceGetInforomImageVersion(device, version, length);
    if (write(connfd, version, length) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetInforomVersion(int connfd) {
    nvmlDevice_t device;
    nvmlInforomObject_t object;
    unsigned int length;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &object, sizeof(nvmlInforomObject_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char version[length];
    nvmlReturn_t result = nvmlDeviceGetInforomVersion(device, object, version, length);
    if (write(connfd, version, length) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetIrqNum(int connfd) {
    nvmlDevice_t device;
    unsigned int irqNum;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetIrqNum(device, &irqNum);
    if (write(connfd, &irqNum, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(int connfd) {
    nvmlDevice_t device;
    unsigned int infoCount;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &infoCount, sizeof(unsigned int)) < 0)
        return -1;
    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
    nvmlReturn_t result = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, infos);
    if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 || write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;
    free(infos);
    return result;
}

int handle_nvmlDeviceGetMaxClockInfo(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int clock;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &type, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMaxClockInfo(device, type, &clock);
    if (write(connfd, &clock, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMaxCustomerBoostClock(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, &clockMHz);
    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(int connfd) {
    nvmlDevice_t device;
    unsigned int maxLinkGen;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkGeneration(device, &maxLinkGen);
    if (write(connfd, &maxLinkGen, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(int connfd) {
    nvmlDevice_t device;
    unsigned int maxLinkWidth;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkWidth(device, &maxLinkWidth);
    if (write(connfd, &maxLinkWidth, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMemoryInfo(int connfd) {
    nvmlDevice_t device;
    nvmlMemory_t memory;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (write(connfd, &memory, sizeof(nvmlMemory_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMemoryInfo_v2(int connfd) {
    nvmlDevice_t device;
    nvmlMemory_v2_t memoryInfo;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    memoryInfo.version = nvmlMemory_v2;
    nvmlReturn_t res = nvmlDeviceGetMemoryInfo_v2(device, &memoryInfo);
    if (res != NVML_SUCCESS)
        return -1;
    if (write(connfd, &memoryInfo, sizeof(nvmlMemory_v2_t)) < 0)
        return -1;
    return NVML_SUCCESS;
}

int handle_nvmlDeviceGetMinMaxClockOfPState(int connfd) {
    nvmlDevice_t device;
    nvmlClockType_t type;
    nvmlPstates_t pstate;
    unsigned int minClockMHz;
    unsigned int maxClockMHz;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &type, sizeof(nvmlClockType_t)) < 0 || read(connfd, &pstate, sizeof(nvmlPstates_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);
    if (write(connfd, &minClockMHz, sizeof(unsigned int)) < 0 || write(connfd, &maxClockMHz, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMinMaxFanSpeed(int connfd) {
    nvmlDevice_t device;
    unsigned int minSpeed;
    unsigned int maxSpeed;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);
    if (write(connfd, &minSpeed, sizeof(unsigned int)) < 0 || write(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMinorNumber(int connfd) {
    nvmlDevice_t device;
    unsigned int minorNumber;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMinorNumber(device, &minorNumber);
    if (write(connfd, &minorNumber, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetMultiGpuBoard(int connfd) {
    nvmlDevice_t device;
    unsigned int multiGpuBool;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetMultiGpuBoard(device, &multiGpuBool);
    if (write(connfd, &multiGpuBool, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetName(int connfd) {
    nvmlDevice_t device;
    unsigned int length;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;
    char name[length];
    nvmlReturn_t result = nvmlDeviceGetName(device, name, length);
    if (write(connfd, name, length) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetNumFans(int connfd) {
    nvmlDevice_t device;
    unsigned int numFans;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetNumFans(device, &numFans);
    if (write(connfd, &numFans, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetNumGpuCores(int connfd) {
    nvmlDevice_t device;
    unsigned int numCores;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetNumGpuCores(device, &numCores);
    if (write(connfd, &numCores, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetP2PStatus(int connfd) {
    nvmlDevice_t device1, device2;
    nvmlGpuP2PCapsIndex_t p2pIndex;
    nvmlGpuP2PStatus_t p2pStatus;
    if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 || read(connfd, &device2, sizeof(nvmlDevice_t)) < 0 || read(connfd, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, &p2pStatus);
    if (write(connfd, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPciInfo_v3(int connfd) {
    nvmlDevice_t device;
    nvmlPciInfo_t pci;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPciInfo_v3(device, &pci);
    if (write(connfd, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPcieLinkMaxSpeed(int connfd) {
    nvmlDevice_t device;
    unsigned int maxSpeed;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPcieLinkMaxSpeed(device, &maxSpeed);
    if (write(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPcieReplayCounter(int connfd) {
    nvmlDevice_t device;
    unsigned int value;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPcieReplayCounter(device, &value);
    if (write(connfd, &value, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPcieSpeed(int connfd) {
    nvmlDevice_t device;
    unsigned int pcieSpeed;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed);
    if (write(connfd, &pcieSpeed, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPcieThroughput(int connfd) {
    nvmlDevice_t device;
    nvmlPcieUtilCounter_t counter;
    unsigned int value;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPcieThroughput(device, counter, &value);
    if (write(connfd, &value, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPerformanceState(int connfd) {
    nvmlDevice_t device;
    nvmlPstates_t pState;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPerformanceState(device, &pState);
    if (write(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPersistenceMode(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPersistenceMode(device, &mode);
    if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(int connfd) {
    nvmlDevice_t device;
    unsigned int defaultLimit;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);
    if (write(connfd, &defaultLimit, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerManagementLimit(int connfd) {
    nvmlDevice_t device;
    unsigned int limit;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimit(device, &limit);
    if (write(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerManagementLimitConstraints(int connfd) {
    nvmlDevice_t device;
    unsigned int minLimit;
    unsigned int maxLimit;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);
    if (write(connfd, &minLimit, sizeof(unsigned int)) < 0 || write(connfd, &maxLimit, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerManagementMode(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t mode;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerManagementMode(device, &mode);
    if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerSource(int connfd) {
    nvmlDevice_t device;
    nvmlPowerSource_t powerSource;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerSource(device, &powerSource);
    if (write(connfd, &powerSource, sizeof(nvmlPowerSource_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerState(int connfd) {
    nvmlDevice_t device;
    nvmlPstates_t pState;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerState(device, &pState);
    if (write(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetPowerUsage(int connfd) {
    nvmlDevice_t device;
    unsigned int power;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
    if (write(connfd, &power, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetProcessUtilization(int connfd) {
    nvmlDevice_t device;
    unsigned int processSamplesCount;
    unsigned long long lastSeenTimeStamp;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &processSamplesCount, sizeof(unsigned int)) < 0 || read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;
    nvmlProcessUtilizationSample_t utilization[processSamplesCount];
    nvmlReturn_t result = nvmlDeviceGetProcessUtilization(device, utilization, &processSamplesCount, lastSeenTimeStamp);
    if (write(connfd, &processSamplesCount, sizeof(unsigned int)) < 0 || write(connfd, utilization, processSamplesCount * sizeof(nvmlProcessUtilizationSample_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetRemappedRows(int connfd) {
    nvmlDevice_t device;
    unsigned int corrRows, uncRows, isPending, failureOccurred;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;
    nvmlReturn_t result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);
    if (write(connfd, &corrRows, sizeof(unsigned int)) < 0 || write(connfd, &uncRows, sizeof(unsigned int)) < 0 || write(connfd, &isPending, sizeof(unsigned int)) < 0 || write(connfd, &failureOccurred, sizeof(unsigned int)) < 0)
        return -1;
    return result;
}

int handle_nvmlDeviceGetRetiredPages(int connfd) {
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, NULL);
    unsigned long long addresses[pageCount];
    result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, addresses);

    if (write(connfd, &pageCount, sizeof(unsigned int)) < 0 || write(connfd, addresses, pageCount * sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(int connfd) {
    nvmlDevice_t device;
    nvmlEnableState_t isPending;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPagesPendingStatus(device, &isPending);

    if (write(connfd, &isPending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPages_v2(int connfd) {
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, NULL, NULL);
    unsigned long long addresses[pageCount], timestamps[pageCount];
    result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, addresses, timestamps);

    if (write(connfd, &pageCount, sizeof(unsigned int)) < 0 || write(connfd, addresses, pageCount * sizeof(unsigned long long)) < 0 || write(connfd, timestamps, pageCount * sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRowRemapperHistogram(int connfd) {
    nvmlDevice_t device;
    nvmlRowRemapperHistogramValues_t values;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRowRemapperHistogram(device, &values);

    if (write(connfd, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSamples(int connfd) {
    nvmlDevice_t device;
    nvmlSamplingType_t type;
    unsigned long long lastSeenTimeStamp;
    unsigned int sampleCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &type, sizeof(nvmlSamplingType_t)) < 0 || read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 || read(connfd, &sampleCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlSample_t samples[sampleCount];
    nvmlValueType_t sampleValType;

    nvmlReturn_t result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);

    if (write(connfd, &sampleValType, sizeof(nvmlValueType_t)) < 0 || write(connfd, &sampleCount, sizeof(unsigned int)) < 0 || write(connfd, samples, sampleCount * sizeof(nvmlSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSerial(int connfd) {
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char serial[length];
    nvmlReturn_t result = nvmlDeviceGetSerial(device, serial, length);

    if (write(connfd, serial, length) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedClocksThrottleReasons(int connfd) {
    nvmlDevice_t device;
    unsigned long long supportedClocksThrottleReasons;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedClocksThrottleReasons(device, &supportedClocksThrottleReasons);

    if (write(connfd, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(int connfd) {
    nvmlDevice_t device;
    unsigned int memoryClockMHz;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &memoryClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, NULL);

    unsigned int clocksMHz[count];
    result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, clocksMHz);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 || write(connfd, clocksMHz, count * sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedMemoryClocks(int connfd) {
    nvmlDevice_t device;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedMemoryClocks(device, &count, NULL);

    unsigned int clocksMHz[count];
    result = nvmlDeviceGetSupportedMemoryClocks(device, &count, clocksMHz);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 || write(connfd, clocksMHz, count * sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(int connfd) {
    nvmlDevice_t device;
    unsigned int size;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    nvmlPstates_t pstates[size];
    nvmlReturn_t result = nvmlDeviceGetSupportedPerformanceStates(device, pstates, size);

    if (write(connfd, pstates, size * sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTargetFanSpeed(int connfd) {
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int targetSpeed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &fan, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTargetFanSpeed(device, fan, &targetSpeed);

    if (write(connfd, &targetSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTemperature(int connfd) {
    nvmlDevice_t device;
    nvmlTemperatureSensors_t sensorType;
    unsigned int temp;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTemperature(device, sensorType, &temp);

    if (write(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTemperatureThreshold(int connfd) {
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    unsigned int temp;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, &temp);

    if (write(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetThermalSettings(int connfd) {
    nvmlDevice_t device;
    unsigned int sensorIndex;
    nvmlGpuThermalSettings_t pThermalSettings;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &sensorIndex, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetThermalSettings(device, sensorIndex, &pThermalSettings);

    if (write(connfd, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTopologyCommonAncestor(int connfd) {
    nvmlDevice_t device1, device2;
    nvmlGpuTopologyLevel_t pathInfo;

    if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 || read(connfd, &device2, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &pathInfo);

    if (write(connfd, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTopologyNearestGpus(int connfd) {
    nvmlDevice_t device;
    nvmlGpuTopologyLevel_t level;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, NULL);

    nvmlDevice_t deviceArray[count];
    result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, deviceArray);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 || write(connfd, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTotalEccErrors(int connfd) {
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    unsigned long long eccCounts;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 || read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, &eccCounts);

    if (write(connfd, &eccCounts, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTotalEnergyConsumption(int connfd) {
    nvmlDevice_t device;
    unsigned long long energy;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);

    if (write(connfd, &energy, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetUUID(int connfd) {
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char uuid[length];
    nvmlReturn_t result = nvmlDeviceGetUUID(device, uuid, length);

    if (write(connfd, uuid, length) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetUtilizationRates(int connfd) {
    nvmlDevice_t device;
    nvmlUtilization_t utilization;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);

    if (write(connfd, &utilization, sizeof(nvmlUtilization_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVbiosVersion(int connfd) {
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char version[length];
    nvmlReturn_t result = nvmlDeviceGetVbiosVersion(device, version, length);

    if (write(connfd, version, length) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetViolationStatus(int connfd) {
    nvmlDevice_t device;
    nvmlPerfPolicyType_t perfPolicyType;
    nvmlViolationTime_t violTime;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetViolationStatus(device, perfPolicyType, &violTime);

    if (write(connfd, &violTime, sizeof(nvmlViolationTime_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceOnSameBoard(int connfd) {
    nvmlDevice_t device1, device2;
    int onSameBoard;

    if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 || read(connfd, &device2, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceOnSameBoard(device1, device2, &onSameBoard);

    if (write(connfd, &onSameBoard, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceValidateInforom(int connfd) {
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceValidateInforom(device);

    return result;
}

int handle_nvmlUnitSetLedState(int connfd) {
    nvmlUnit_t unit;
    nvmlLedColor_t color;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 || read(connfd, &color, sizeof(nvmlLedColor_t)) < 0)
        return -1;

    return nvmlUnitSetLedState(unit, color);
}

int handle_nvmlDeviceGetSupportedEventTypes(int connfd) {
    nvmlDevice_t device;
    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    unsigned long long eventTypes;
    nvmlReturn_t result = nvmlDeviceGetSupportedEventTypes(device, &eventTypes);

    if (write(connfd, &eventTypes, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}


int handle_nvmlDeviceRegisterEvents(int connfd) {
    nvmlDevice_t device;
    unsigned long long eventTypes;
    nvmlEventSet_t set;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 || read(connfd, &eventTypes, sizeof(unsigned long long)) < 0 || read(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceRegisterEvents(device, eventTypes, set);
    return result;
}

int handle_nvmlEventSetCreate(int connfd) {
    nvmlEventSet_t set;
    nvmlReturn_t result = nvmlEventSetCreate(&set);
    if (write(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;
    return result;
}

int handle_nvmlEventSetFree(int connfd) {
    nvmlEventSet_t set;

    if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    return nvmlEventSetFree(set);
}

int handle_cuDriverGetVersion(int connfd) {
    nvmlEventSet_t set;
    nvmlEventData_t data;
    unsigned int timeoutms;

    if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0 || read(connfd, &timeoutms, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetWait_v2(set, &data, timeoutms);

    if (write(connfd, &data, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}


int handle_cuGetProcAddress_v2(int connfd) {
    int symbol_length;
    char symbol[256];
    void *func = nullptr;  // pointer to be sent back to the client
    int cudaVersion;
    cuuint64_t flags;
    CUdriverProcAddressQueryResult result;

    std::cerr << ">>>> Server: Receiving parameters" << std::endl;

    if (read(connfd, &symbol_length, sizeof(int)) < 0) {
        std::cerr << "Failed to read symbol length" << std::endl;
        return -1;
    }

    if (symbol_length <= 0 || symbol_length > sizeof(symbol)) {
        std::cerr << "Invalid symbol length" << std::endl;
        return -1;
    }

    if (read(connfd, symbol, symbol_length) < 0) {
        std::cerr << "Failed to read symbol string" << std::endl;
        return -1;
    }

    if (read(connfd, &cudaVersion, sizeof(int)) < 0) {
        std::cerr << "Failed to read CUDA version" << std::endl;
        return -1;
    }

    if (read(connfd, &func, sizeof(void *)) < 0) {
        std::cerr << "Failed to read function pointer" << std::endl;
        return -1;
    }

    if (read(connfd, &flags, sizeof(cuuint64_t)) < 0) {
        std::cerr << "Failed to read flags" << std::endl;
        return -1;
    }

    std::cout << "Server: Symbol - " << symbol << ", CUDA Version - " << cudaVersion << ", Flags - " << flags << std::endl;

    // Call the actual cuGetProcAddress_v2 function with the correct parameters
    CUresult result_code = cuGetProcAddress_v2(symbol, &func, cudaVersion, flags, &result);
    if (result_code != CUDA_SUCCESS) {
        const char *errorStr = nullptr;
        cuGetErrorString(result_code, &errorStr);
        std::cerr << "cuGetProcAddress_v2 failed with error code: " << result_code 
                << " (" << (errorStr ? errorStr : "Unknown error") << ")" << std::endl;
        return -1;
    }

    std::cout << "v2 result: " << &result << std::endl;

    if (write(connfd, &func, sizeof(void *)) < 0) {
        std::cerr << "Failed to write function pointer to client. Error: " << strerror(errno) << std::endl;
        return -1;
    }

    if (write(connfd, &result, sizeof(CUdriverProcAddressQueryResult)) < 0) {
        std::cerr << "Failed to write result to client. Error: " << strerror(errno) << std::endl;
        return -1;
    }

    std::cout << "Server: Sent function pointer and result to client successfully." << std::endl;

    return result;
}

static RequestHandler opHandlers[] = {
    /* 0  */ handle_nvmlInitWithFlags,
    /* 1  */ handle_nvmlInit_v2,
    /* 2  */ handle_nvmlShutdown,
    /* 3  */ nullptr,  // No handler for RPC_nvmlErrorString, as an example
    /* 4  */ handle_nvmlSystemGetDriverVersion,
    /* 5  */ nullptr,  // RPC_nvmlSystemGetDriverVersion_v2 is commented
    /* 6  */ nullptr,  // RPC_nvmlSystemGetDriverBranch is commented
    /* 7  */ handle_nvmlSystemGetHicVersion,
    /* 8  */ handle_nvmlSystemGetNVMLVersion,
    /* 9  */ handle_nvmlSystemGetProcessName,
    /* 10 */ nullptr, // handle_nvmlSystemGetTopologyGpuSet,
    /* 11 */ handle_nvmlUnitGetCount,
    /* 12 */ handle_nvmlUnitGetDevices,
    /* 13 */ handle_nvmlUnitGetFanSpeedInfo,
    /* 14 */ handle_nvmlUnitGetHandleByIndex,
    /* 15 */ handle_nvmlUnitGetLedState,
    /* 16 */ handle_nvmlUnitGetPsuInfo,
    /* 17 */ handle_nvmlUnitGetTemperature,
    /* 18 */ handle_nvmlUnitGetUnitInfo,
    /* 19 */ handle_nvmlDeviceGetAPIRestriction,
    /* 20 */ handle_nvmlDeviceGetAdaptiveClockInfoStatus,
    /* 21 */ handle_nvmlDeviceGetApplicationsClock,
    /* 22 */ handle_nvmlDeviceGetArchitecture,
    /* 23 */ handle_nvmlDeviceGetAttributes_v2,
    /* 24 */ handle_nvmlDeviceGetAutoBoostedClocksEnabled,
    /* 25 */ handle_nvmlDeviceGetBAR1MemoryInfo,
    /* 26 */ handle_nvmlDeviceGetBoardId,
    /* 27 */ handle_nvmlDeviceGetBoardPartNumber,
    /* 28 */ handle_nvmlDeviceGetBrand,
    /* 29 */ handle_nvmlDeviceGetBridgeChipInfo,
    /* 30 */ handle_nvmlDeviceGetBusType,
    /* 31 */ nullptr, // handle_nvmlDeviceGetC2cModeInfoV (missing)
    /* 32 */ handle_nvmlDeviceGetClkMonStatus,
    /* 33 */ handle_nvmlDeviceGetClock,
    /* 34 */ handle_nvmlDeviceGetClockInfo,
    /* 35 */ nullptr,  // handle_nvmlDeviceGetClockOffsets
    /* 36 */ handle_nvmlDeviceGetComputeMode,
    /* 37 */ handle_nvmlDeviceGetComputeRunningProcesses_v3,
    /* 38 */ nullptr,  // handle_nvmlDeviceGetConfComputeGpuAttestationReport
    /* 39 */ nullptr,  // handle_nvmlDeviceGetConfComputeGpuCertificate
    /* 40 */ nullptr,  // handle_nvmlDeviceGetConfComputeMemSizeInfo
    /* 41 */ nullptr,  // handle_nvmlDeviceGetConfComputeProtectedMemoryUsage
    /* 42 */ handle_nvmlDeviceGetCount_v2,
    /* 43 */ handle_nvmlDeviceGetCudaComputeCapability,
    /* 44 */ handle_nvmlDeviceGetCurrPcieLinkGeneration,
    /* 45 */ handle_nvmlDeviceGetCurrPcieLinkWidth,
    /* 46 */ nullptr,  // handle_nvmlDeviceGetCurrentClocksEventReasons
    /* 47 */ handle_nvmlDeviceGetCurrentClocksThrottleReasons,
    /* 48 */ handle_nvmlDeviceGetDecoderUtilization,
    /* 49 */ handle_nvmlDeviceGetDefaultApplicationsClock,
    /* 50 */ handle_nvmlDeviceGetDefaultEccMode,
    /* 51 */ handle_nvmlDeviceGetDetailedEccErrors,
    /* 52 */ handle_nvmlDeviceGetDisplayActive,
    /* 53 */ handle_nvmlDeviceGetDisplayMode,
    /* 54 */ nullptr,  // handle_nvmlDeviceGetDriverModel_v2
    /* 55 */ handle_nvmlDeviceGetDynamicPstatesInfo,
    /* 56 */ handle_nvmlDeviceGetEccMode,
    /* 57 */ handle_nvmlDeviceGetEncoderCapacity,
    /* 58 */ handle_nvmlDeviceGetEncoderSessions,
    /* 59 */ handle_nvmlDeviceGetEncoderStats,
    /* 60 */ handle_nvmlDeviceGetEncoderUtilization,
    /* 61 */ handle_nvmlDeviceGetEnforcedPowerLimit,
    /* 62 */ handle_nvmlDeviceGetFBCSessions,
    /* 63 */ handle_nvmlDeviceGetFBCStats,
    /* 64 */ handle_nvmlDeviceGetFanControlPolicy_v2,
    /* 65 */ handle_nvmlDeviceGetFanSpeed,
    /* 66 */ handle_nvmlDeviceGetFanSpeed_v2,
    /* 67 */ handle_nvmlDeviceGetGpcClkMinMaxVfOffset,
    /* 68 */ handle_nvmlDeviceGetGpcClkVfOffset,
    /* 69 */ handle_nvmlDeviceGetGpuFabricInfo,
    /* 70 */ nullptr,  // handle_nvmlDeviceGetGpuFabricInfoV
    /* 71 */ handle_nvmlDeviceGetGpuMaxPcieLinkGeneration,
    /* 72 */ handle_nvmlDeviceGetGpuOperationMode,
    /* 73 */ handle_nvmlDeviceGetGraphicsRunningProcesses_v3,
    /* 74 */ handle_nvmlDeviceGetGspFirmwareMode,
    /* 75 */ handle_nvmlDeviceGetGspFirmwareVersion,
    /* 76 */ handle_nvmlDeviceGetHandleByIndex_v2,
    /* 77 */ handle_nvmlDeviceGetHandleByPciBusId_v2,
    /* 78 */ handle_nvmlDeviceGetHandleBySerial,
    /* 79 */ handle_nvmlDeviceGetHandleByUUID,
    /* 80 */ handle_nvmlDeviceGetIndex,
    /* 81 */ handle_nvmlDeviceGetInforomConfigurationChecksum,
    /* 82 */ handle_nvmlDeviceGetInforomImageVersion,
    /* 83 */ handle_nvmlDeviceGetInforomVersion,
    /* 84 */ handle_nvmlDeviceGetIrqNum,
    /* 85 */ nullptr,  // handle_nvmlDeviceGetJpgUtilization
    /* 86 */ nullptr,  // handle_nvmlDeviceGetLastBBXFlushTime
    /* 87 */ handle_nvmlDeviceGetMPSComputeRunningProcesses_v3,
    /* 88 */ handle_nvmlDeviceGetMaxClockInfo,
    /* 89 */ handle_nvmlDeviceGetMaxCustomerBoostClock,
    /* 90 */ handle_nvmlDeviceGetMaxPcieLinkGeneration,
    /* 91 */ handle_nvmlDeviceGetMaxPcieLinkWidth,
    /* 92 */ nullptr,  // handle_nvmlDeviceGetMemClkMinMaxVfOffset,
    /* 93 */ nullptr,  // handle_nvmlDeviceGetMemClkVfOffset,
    /* 94 */ nullptr,  // handle_nvmlDeviceGetMemoryBusWidth,
    /* 95 */ nullptr,  // handle_nvmlDeviceGetMemoryErrorCounter,
    /* 96 */ handle_nvmlDeviceGetMemoryInfo,
    /* 97 */ handle_nvmlDeviceGetMemoryInfo_v2,
    /* 98 */ handle_nvmlDeviceGetMinMaxClockOfPState,
    /* 99 */ handle_nvmlDeviceGetMinMaxFanSpeed,
    /* 100 */ handle_nvmlDeviceGetMinorNumber,
    /* 101 */ nullptr,  // handle_nvmlDeviceGetModuleId,
    /* 102 */ handle_nvmlDeviceGetMultiGpuBoard,
    /* 103 */ handle_nvmlDeviceGetName,
    /* 104 */ handle_nvmlDeviceGetNumFans,
    /* 105 */ handle_nvmlDeviceGetNumGpuCores,
    /* 106 */ nullptr,  // handle_nvmlDeviceGetOfaUtilization,
    /* 107 */ handle_nvmlDeviceGetP2PStatus,
    /* 108 */ nullptr,  // handle_nvmlDeviceGetPciInfoExt,
    /* 109 */ handle_nvmlDeviceGetPciInfo_v3,
    /* 110 */ handle_nvmlDeviceGetPcieLinkMaxSpeed,
    /* 111 */ handle_nvmlDeviceGetPcieReplayCounter,
    /* 112 */ handle_nvmlDeviceGetPcieSpeed,
    /* 113 */ handle_nvmlDeviceGetPcieThroughput,
    /* 114 */ handle_nvmlDeviceGetPerformanceState,
    /* 115 */ handle_nvmlDeviceGetPersistenceMode,
    /* 116 */ handle_nvmlDeviceGetPowerManagementDefaultLimit,
    /* 117 */ handle_nvmlDeviceGetPowerManagementLimit,
    /* 118 */ handle_nvmlDeviceGetPowerManagementLimitConstraints,
    /* 119 */ handle_nvmlDeviceGetPowerManagementMode,
    /* 120 */ handle_nvmlDeviceGetPowerSource,
    /* 121 */ handle_nvmlDeviceGetPowerState,
    /* 122 */ handle_nvmlDeviceGetPowerUsage,
    /* 123 */ handle_nvmlDeviceGetProcessUtilization,
    /* 124 */ nullptr,  // handle_nvmlDeviceGetProcessesUtilizationInfo,
    /* 125 */ handle_nvmlDeviceGetRemappedRows,
    /* 126 */ handle_nvmlDeviceGetRetiredPages,
    /* 127 */ handle_nvmlDeviceGetRetiredPagesPendingStatus,
    /* 128 */ handle_nvmlDeviceGetRetiredPages_v2,
    /* 129 */ handle_nvmlDeviceGetRowRemapperHistogram,
    /* 130 */ nullptr,  // handle_nvmlDeviceGetRunningProcessDetailList,
    /* 131 */ handle_nvmlDeviceGetSamples,
    /* 132 */ handle_nvmlDeviceGetSerial,
    /* 133 */ nullptr,  // handle_nvmlDeviceGetSramEccErrorStatus,
    /* 134 */ nullptr, // handle_nvmlDeviceGetSupportedClocksEventReasons
    /* 135 */ handle_nvmlDeviceGetSupportedClocksThrottleReasons,
    /* 136 */ handle_nvmlDeviceGetSupportedGraphicsClocks,
    /* 137 */ handle_nvmlDeviceGetSupportedMemoryClocks,
    /* 138 */ handle_nvmlDeviceGetSupportedPerformanceStates,
    /* 139 */ handle_nvmlDeviceGetTargetFanSpeed,
    /* 140 */ handle_nvmlDeviceGetTemperature,
    /* 141 */ handle_nvmlDeviceGetTemperatureThreshold,
    /* 142 */ handle_nvmlDeviceGetThermalSettings,
    /* 143 */ handle_nvmlDeviceGetTopologyCommonAncestor,
    /* 144 */ handle_nvmlDeviceGetTopologyNearestGpus,
    /* 145 */ handle_nvmlDeviceGetTotalEccErrors,
    /* 146 */ handle_nvmlDeviceGetTotalEnergyConsumption,
    /* 147 */ handle_nvmlDeviceGetUUID,
    /* 148 */ handle_nvmlDeviceGetUtilizationRates,
    /* 149 */ handle_nvmlDeviceGetVbiosVersion,
    /* 150 */ handle_nvmlDeviceGetViolationStatus,
    /* 151 */ handle_nvmlDeviceOnSameBoard,
    /* 152 */ nullptr,  // handle_nvmlDeviceSetClockOffsets,
    /* 153 */ nullptr,  // handle_nvmlDeviceSetConfComputeUnprotectedMemSize,
    /* 154 */ nullptr,  // handle_nvmlDeviceSetPowerManagementLimit_v2,
    /* 155 */ handle_nvmlDeviceValidateInforom,
    /* 156 */ nullptr,  // handle_nvmlSystemGetConfComputeCapabilities,
    /* 157 */ nullptr,  // handle_nvmlSystemGetConfComputeGpusReadyState,
    /* 158 */ nullptr,  // handle_nvmlSystemGetConfComputeKeyRotationThresholdInfo,
    /* 159 */ nullptr,  // handle_nvmlSystemGetConfComputeSettings,
    /* 160 */ nullptr,  // handle_nvmlSystemGetConfComputeState,
    /* 161 */ nullptr,  // handle_nvmlSystemSetConfComputeGpusReadyState,
    /* 162 */ handle_nvmlUnitSetLedState,
    /* 163 */ nullptr,  // handle_nvmlDeviceClearEccErrorCounts,
    /* 164 */ nullptr,  // handle_nvmlDeviceResetApplicationsClocks,
    /* 165 */ nullptr,  // handle_nvmlDeviceResetGpuLockedClocks,
    /* 166 */ nullptr,  // handle_nvmlDeviceResetMemoryLockedClocks,
    /* 167 */ nullptr,  // handle_nvmlDeviceSetAPIRestriction,
    /* 168 */ nullptr,  // handle_nvmlDeviceSetApplicationsClocks,
    /* 169 */ nullptr,  // handle_nvmlDeviceSetAutoBoostedClocksEnabled,
    /* 170 */ nullptr,  // handle_nvmlDeviceSetComputeMode,
    /* 171 */ nullptr,  // handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled,
    /* 172 */ nullptr,  // handle_nvmlDeviceSetDefaultFanSpeed_v2,
    /* 173 */ nullptr,  // handle_nvmlDeviceSetDriverModel,
    /* 174 */ nullptr,  // handle_nvmlDeviceSetEccMode,
    /* 175 */ nullptr,  // handle_nvmlDeviceSetFanControlPolicy,
    /* 176 */ nullptr,  // handle_nvmlDeviceSetFanSpeed_v2,
    /* 177 */ nullptr,  // handle_nvmlDeviceSetGpcClkVfOffset,
    /* 178 */ nullptr,  // handle_nvmlDeviceSetGpuLockedClocks,
    /* 179 */ nullptr,  // handle_nvmlDeviceSetGpuOperationMode,
    /* 180 */ nullptr,  // handle_nvmlDeviceSetMemClkVfOffset,
    /* 181 */ nullptr,  // handle_nvmlDeviceSetMemoryLockedClocks,
    /* 182 */ nullptr,  // handle_nvmlDeviceSetPersistenceMode,
    /* 183 */ nullptr,  // handle_nvmlDeviceSetPowerManagementLimit,
    /* 184 */ nullptr,  // handle_nvmlDeviceSetTemperatureThreshold,
    /* 185 */ nullptr,  // handle_nvmlDeviceFreezeNvLinkUtilizationCounter,
    /* 186 */ nullptr,  // handle_nvmlDeviceGetNvLinkCapability,
    /* 187 */ nullptr,  // handle_nvmlDeviceGetNvLinkErrorCounter,
    /* 188 */ nullptr,  // handle_nvmlDeviceGetNvLinkRemoteDeviceType,
    /* 189 */ nullptr,  // handle_nvmlDeviceGetNvLinkRemotePciInfo_v2,
    /* 190 */ nullptr,  // handle_nvmlDeviceGetNvLinkState,
    /* 191 */ nullptr,  // handle_nvmlDeviceGetNvLinkUtilizationControl,
    /* 192 */ nullptr,  // handle_nvmlDeviceGetNvLinkUtilizationCounter,
    /* 193 */ nullptr,  // handle_nvmlDeviceGetNvLinkVersion,
    /* 194 */ nullptr,  // handle_nvmlDeviceResetNvLinkErrorCounters,
    /* 195 */ nullptr,  // handle_nvmlDeviceResetNvLinkUtilizationCounter,
    /* 196 */ nullptr,  // handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold,
    /* 197 */ nullptr,  // handle_nvmlDeviceSetNvLinkUtilizationControl,
    /* 198 */ nullptr,  // handle_nvmlSystemGetNvlinkBwMode,
    /* 199 */ nullptr,  // handle_nvmlSystemSetNvlinkBwMode,
    handle_nvmlDeviceGetSupportedEventTypes,       // RPC_nvmlDeviceGetSupportedEventTypes (200)
    handle_nvmlDeviceRegisterEvents,               // RPC_nvmlDeviceRegisterEvents (201)
    handle_nvmlEventSetCreate,                     // RPC_nvmlEventSetCreate (202)
    handle_nvmlEventSetFree,                       // RPC_nvmlEventSetFree (203)
    nullptr, // 204
    // cuda
    nullptr,                 // RPC_cuDriverGetVersion (205)
    nullptr,                    // RPC_cuLinkCreate_v2 (206)
    nullptr,                   // RPC_cuLinkAddData_v2 (207)
    nullptr,                     // RPC_cuLinkComplete (208)
    nullptr,                   // RPC_cuModuleLoadData (209)
    nullptr,                     // RPC_cuModuleUnload (210)
    nullptr,                   // RPC_cuGetErrorString (211)
    nullptr,                      // RPC_cuLinkDestroy (212)
    nullptr,                // RPC_cuModuleGetFunction (213)
    nullptr,                 // RPC_cuFuncSetAttribute (214)
    nullptr,                     // RPC_cuLaunchKernel (215)
    nullptr,                     // RPC_cuGetErrorName (216)
    nullptr,              // RPC_cuModuleLoadFatBinary (217)
    nullptr,                 // RPC_cuModuleLoadDataEx (218)
    nullptr,                   // RPC_cuLinkAddFile_v2 (219)
    nullptr,                             // RPC_cuInit (220)
    nullptr,                 // RPC_cuFuncGetAttribute (221)
    nullptr,                   // RPC_cuCtxPushCurrent (222)
    nullptr,                    // RPC_cuCtxPopCurrent (223)
    nullptr,                     // RPC_cuCtxGetDevice (224)
    nullptr,           // RPC_cuDevicePrimaryCtxRetain (225)
    nullptr,          // RPC_cuDevicePrimaryCtxRelease (226)
    nullptr,            // RPC_cuDevicePrimaryCtxReset (227)
    nullptr,                        // RPC_cuDeviceGet (228)
    nullptr,               // RPC_cuDeviceGetAttribute (229)
    nullptr,                // RPC_cuStreamSynchronize (230)
    nullptr, // RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor (231)
    nullptr,                   // RPC_cuLaunchKernelEx (232)
    nullptr,                    // RPC_cuMemcpyDtoH_v2 (233)
    nullptr,               // RPC_cuModuleGetGlobal_v2 (234)
    handle_cuGetProcAddress_v2,
};

int request_handler(int connfd) {
    unsigned int op;

    // Attempt to read the operation code from the client
    if (read(connfd, &op, sizeof(unsigned int)) < 0) {
        std::cout << "Error reading opcode from client" << std::endl;
        return -1;
    }

    if (opHandlers[op] == NULL) {
        std::cout << "Unknown or unsupported operation: " << op << std::endl;
        return -1;
    }

    return opHandlers[op](connfd);
}

void client_handler(int connfd)
{
    int request_id;

    while (1)
    {
        int n = read(connfd, &request_id, sizeof(int));
        if (n == 0)
        {
            printf("client disconnected, loop continuing. \n");
            break;
        }
        else if (n < 0)
        {
            printf("error reading from client.\n");
            break;
        }

        if (write(connfd, &request_id, sizeof(int)) < 0)
        {
            printf("error writing to client.\n");
            break;
        }

        // run our request handler in a separate thread
        std::future<int> request_future = std::async(std::launch::async, [connfd]()
                                                     {
            std::cout << "request handled by thread: " << std::this_thread::get_id() << std::endl;

            return request_handler(connfd); });

        // wait for result
        int res = request_future.get();

        if (write(connfd, &res, sizeof(int)) < 0)
        {
            printf("error writing result to client.\n");
            break;
        }
    }

    close(connfd);
}

int main()
{
    int port = DEFAULT_PORT;
    struct sockaddr_in servaddr, cli;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("Socket creation failed.\n");
        exit(EXIT_FAILURE);
    }

    char *p = getenv("SCUDA_PORT");

    if (p == NULL)
    {
        port = DEFAULT_PORT;
    }
    else
    {
        port = atoi(p);
        // << "Using SCUDA_PORT: " << port << std::endl;
    }

    // Bind the socket
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(port);

    const int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
    {
        printf("Socket bind failed.\n");
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0)
    {
        printf("Socket bind failed.\n");
        exit(EXIT_FAILURE);
    }

    if (listen(sockfd, MAX_CLIENTS) != 0)
    {
        printf("Listen failed.\n");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", port);

    // Server loop
    while (1)
    {
        socklen_t len = sizeof(cli);
        int connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

        if (connfd < 0)
        {
            std::cerr << "Server accept failed." << std::endl;
            continue;
        }

        std::cout << "Client connected, spawning thread." << std::endl;

        std::thread client_thread(client_handler, connfd);

        // detach the thread so it runs independently
        client_thread.detach();
    }

    close(sockfd);
    return 0;
}
