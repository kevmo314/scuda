#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <nvml.h>
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

using RequestHandler = std::function<int(int)>;

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

int handle_nvmlEventSetWait_v2(int connfd) {
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

static std::unordered_map<unsigned int, RequestHandler> opHandlers = {
    // 4.11 Initialization and Cleanup
    {RPC_nvmlInitWithFlags, handle_nvmlInitWithFlags},
    {RPC_nvmlInit_v2, handle_nvmlInit_v2},
    {RPC_nvmlShutdown, handle_nvmlShutdown},

    // 4.14 System Queries
    {RPC_nvmlSystemGetDriverVersion, handle_nvmlSystemGetDriverVersion},
    {RPC_nvmlSystemGetHicVersion, handle_nvmlSystemGetHicVersion},
    {RPC_nvmlSystemGetNVMLVersion, handle_nvmlSystemGetNVMLVersion},
    {RPC_nvmlSystemGetProcessName, handle_nvmlSystemGetProcessName},

    // 4.15 Unit Queries
    {RPC_nvmlUnitGetCount, handle_nvmlUnitGetCount},
    {RPC_nvmlUnitGetDevices, handle_nvmlUnitGetDevices},
    {RPC_nvmlUnitGetFanSpeedInfo, handle_nvmlUnitGetFanSpeedInfo},
    {RPC_nvmlUnitGetHandleByIndex, handle_nvmlUnitGetHandleByIndex},
    {RPC_nvmlUnitGetLedState, handle_nvmlUnitGetLedState},
    {RPC_nvmlUnitGetPsuInfo, handle_nvmlUnitGetPsuInfo},
    {RPC_nvmlUnitGetTemperature, handle_nvmlUnitGetTemperature},
    {RPC_nvmlUnitGetUnitInfo, handle_nvmlUnitGetUnitInfo},

    // 4.16 Device Queries
    {RPC_nvmlDeviceGetAPIRestriction, handle_nvmlDeviceGetAPIRestriction},
    {RPC_nvmlDeviceGetAdaptiveClockInfoStatus, handle_nvmlDeviceGetAdaptiveClockInfoStatus},
    {RPC_nvmlDeviceGetApplicationsClock, handle_nvmlDeviceGetApplicationsClock},
    {RPC_nvmlDeviceGetArchitecture, handle_nvmlDeviceGetArchitecture},
    {RPC_nvmlDeviceGetAttributes_v2, handle_nvmlDeviceGetAttributes_v2},
    {RPC_nvmlDeviceGetAutoBoostedClocksEnabled, handle_nvmlDeviceGetAutoBoostedClocksEnabled},
    {RPC_nvmlDeviceGetBAR1MemoryInfo, handle_nvmlDeviceGetBAR1MemoryInfo},
    {RPC_nvmlDeviceGetBoardId, handle_nvmlDeviceGetBoardId},
    {RPC_nvmlDeviceGetBoardPartNumber, handle_nvmlDeviceGetBoardPartNumber},
    {RPC_nvmlDeviceGetBrand, handle_nvmlDeviceGetBrand},
    {RPC_nvmlDeviceGetBridgeChipInfo, handle_nvmlDeviceGetBridgeChipInfo},
    {RPC_nvmlDeviceGetBusType, handle_nvmlDeviceGetBusType},
    {RPC_nvmlDeviceGetClkMonStatus, handle_nvmlDeviceGetClkMonStatus},
    {RPC_nvmlDeviceGetClock, handle_nvmlDeviceGetClock},
    {RPC_nvmlDeviceGetClockInfo, handle_nvmlDeviceGetClockInfo},
    {RPC_nvmlDeviceGetComputeMode, handle_nvmlDeviceGetComputeMode},
    {RPC_nvmlDeviceGetComputeRunningProcesses_v3, handle_nvmlDeviceGetComputeRunningProcesses_v3},
    {RPC_nvmlDeviceGetCount_v2, handle_nvmlDeviceGetCount_v2},
    {RPC_nvmlDeviceGetCudaComputeCapability, handle_nvmlDeviceGetCudaComputeCapability},
    {RPC_nvmlDeviceGetCurrPcieLinkGeneration, handle_nvmlDeviceGetCurrPcieLinkGeneration},
    {RPC_nvmlDeviceGetCurrPcieLinkWidth, handle_nvmlDeviceGetCurrPcieLinkWidth},
    {RPC_nvmlDeviceGetCurrentClocksThrottleReasons, handle_nvmlDeviceGetCurrentClocksThrottleReasons},
    {RPC_nvmlDeviceGetDecoderUtilization, handle_nvmlDeviceGetDecoderUtilization},
    {RPC_nvmlDeviceGetDefaultApplicationsClock, handle_nvmlDeviceGetDefaultApplicationsClock},
    {RPC_nvmlDeviceGetDefaultEccMode, handle_nvmlDeviceGetDefaultEccMode},
    {RPC_nvmlDeviceGetDetailedEccErrors, handle_nvmlDeviceGetDetailedEccErrors},
    {RPC_nvmlDeviceGetDisplayActive, handle_nvmlDeviceGetDisplayActive},
    {RPC_nvmlDeviceGetDisplayMode, handle_nvmlDeviceGetDisplayMode},
    {RPC_nvmlDeviceGetDynamicPstatesInfo, handle_nvmlDeviceGetDynamicPstatesInfo},
    {RPC_nvmlDeviceGetEccMode, handle_nvmlDeviceGetEccMode},
    {RPC_nvmlDeviceGetEncoderCapacity, handle_nvmlDeviceGetEncoderCapacity},
    {RPC_nvmlDeviceGetEncoderSessions, handle_nvmlDeviceGetEncoderSessions},
    {RPC_nvmlDeviceGetEncoderStats, handle_nvmlDeviceGetEncoderStats},
    {RPC_nvmlDeviceGetEncoderUtilization, handle_nvmlDeviceGetEncoderUtilization},
    {RPC_nvmlDeviceGetEnforcedPowerLimit, handle_nvmlDeviceGetEnforcedPowerLimit},
    {RPC_nvmlDeviceGetFBCSessions, handle_nvmlDeviceGetFBCSessions},
    {RPC_nvmlDeviceGetFBCStats, handle_nvmlDeviceGetFBCStats},
    {RPC_nvmlDeviceGetFanControlPolicy_v2, handle_nvmlDeviceGetFanControlPolicy_v2},
    {RPC_nvmlDeviceGetFanSpeed, handle_nvmlDeviceGetFanSpeed},
    {RPC_nvmlDeviceGetFanSpeed_v2, handle_nvmlDeviceGetFanSpeed_v2},
    {RPC_nvmlDeviceGetGpcClkMinMaxVfOffset, handle_nvmlDeviceGetGpcClkMinMaxVfOffset},
    {RPC_nvmlDeviceGetGpcClkVfOffset, handle_nvmlDeviceGetGpcClkVfOffset},
    {RPC_nvmlDeviceGetGpuFabricInfo, handle_nvmlDeviceGetGpuFabricInfo},
    {RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration, handle_nvmlDeviceGetGpuMaxPcieLinkGeneration},
    {RPC_nvmlDeviceGetGpuOperationMode, handle_nvmlDeviceGetGpuOperationMode},
    {RPC_nvmlDeviceGetGraphicsRunningProcesses_v3, handle_nvmlDeviceGetGraphicsRunningProcesses_v3},
    {RPC_nvmlDeviceGetGspFirmwareMode, handle_nvmlDeviceGetGspFirmwareMode},
    {RPC_nvmlDeviceGetGspFirmwareVersion, handle_nvmlDeviceGetGspFirmwareVersion},
    {RPC_nvmlDeviceGetHandleByIndex_v2, handle_nvmlDeviceGetHandleByIndex_v2},
    {RPC_nvmlDeviceGetHandleByPciBusId_v2, handle_nvmlDeviceGetHandleByPciBusId_v2},
    {RPC_nvmlDeviceGetHandleBySerial, handle_nvmlDeviceGetHandleBySerial},
    {RPC_nvmlDeviceGetHandleByUUID, handle_nvmlDeviceGetHandleByUUID},
    {RPC_nvmlDeviceGetIndex, handle_nvmlDeviceGetIndex},
    {RPC_nvmlDeviceGetInforomConfigurationChecksum, handle_nvmlDeviceGetInforomConfigurationChecksum},
    {RPC_nvmlDeviceGetInforomImageVersion, handle_nvmlDeviceGetInforomImageVersion},
    {RPC_nvmlDeviceGetInforomVersion, handle_nvmlDeviceGetInforomVersion},
    {RPC_nvmlDeviceGetIrqNum, handle_nvmlDeviceGetIrqNum},
    {RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3, handle_nvmlDeviceGetMPSComputeRunningProcesses_v3},
    {RPC_nvmlDeviceGetMaxClockInfo, handle_nvmlDeviceGetMaxClockInfo},
    {RPC_nvmlDeviceGetMaxCustomerBoostClock, handle_nvmlDeviceGetMaxCustomerBoostClock},
    {RPC_nvmlDeviceGetMaxPcieLinkGeneration, handle_nvmlDeviceGetMaxPcieLinkGeneration},
    {RPC_nvmlDeviceGetMaxPcieLinkWidth, handle_nvmlDeviceGetMaxPcieLinkWidth},
    {RPC_nvmlDeviceGetMemoryInfo, handle_nvmlDeviceGetMemoryInfo},
    {RPC_nvmlDeviceGetMemoryInfo_v2, handle_nvmlDeviceGetMemoryInfo_v2},
    {RPC_nvmlDeviceGetMinMaxClockOfPState, handle_nvmlDeviceGetMinMaxClockOfPState},
    {RPC_nvmlDeviceGetMinMaxFanSpeed, handle_nvmlDeviceGetMinMaxFanSpeed},
    {RPC_nvmlDeviceGetMinorNumber, handle_nvmlDeviceGetMinorNumber},
    {RPC_nvmlDeviceGetMultiGpuBoard, handle_nvmlDeviceGetMultiGpuBoard},
    {RPC_nvmlDeviceGetName, handle_nvmlDeviceGetName},
    {RPC_nvmlDeviceGetNumFans, handle_nvmlDeviceGetNumFans},
    {RPC_nvmlDeviceGetNumGpuCores, handle_nvmlDeviceGetNumGpuCores},
    {RPC_nvmlDeviceGetP2PStatus, handle_nvmlDeviceGetP2PStatus},
    {RPC_nvmlDeviceGetPciInfo_v3, handle_nvmlDeviceGetPciInfo_v3},
    {RPC_nvmlDeviceGetPcieLinkMaxSpeed, handle_nvmlDeviceGetPcieLinkMaxSpeed},
    {RPC_nvmlDeviceGetPcieReplayCounter, handle_nvmlDeviceGetPcieReplayCounter},
    {RPC_nvmlDeviceGetPcieSpeed, handle_nvmlDeviceGetPcieSpeed},
    {RPC_nvmlDeviceGetPcieThroughput, handle_nvmlDeviceGetPcieThroughput},
    {RPC_nvmlDeviceGetPerformanceState, handle_nvmlDeviceGetPerformanceState},
    {RPC_nvmlDeviceGetPersistenceMode, handle_nvmlDeviceGetPersistenceMode},
    {RPC_nvmlDeviceGetPowerManagementDefaultLimit, handle_nvmlDeviceGetPowerManagementDefaultLimit},
    {RPC_nvmlDeviceGetPowerManagementLimit, handle_nvmlDeviceGetPowerManagementLimit},
    {RPC_nvmlDeviceGetPowerManagementLimitConstraints, handle_nvmlDeviceGetPowerManagementLimitConstraints},
    {RPC_nvmlDeviceGetPowerManagementMode, handle_nvmlDeviceGetPowerManagementMode},
    {RPC_nvmlDeviceGetPowerSource, handle_nvmlDeviceGetPowerSource},
    {RPC_nvmlDeviceGetPowerState, handle_nvmlDeviceGetPowerState},
    {RPC_nvmlDeviceGetPowerUsage, handle_nvmlDeviceGetPowerUsage},
    {RPC_nvmlDeviceGetProcessUtilization, handle_nvmlDeviceGetProcessUtilization},
    {RPC_nvmlDeviceGetRemappedRows, handle_nvmlDeviceGetRemappedRows},
    {RPC_nvmlDeviceGetRetiredPages, handle_nvmlDeviceGetRetiredPages},
    {RPC_nvmlDeviceGetRetiredPagesPendingStatus, handle_nvmlDeviceGetRetiredPagesPendingStatus},
    {RPC_nvmlDeviceGetRetiredPages_v2, handle_nvmlDeviceGetRetiredPages_v2},
    {RPC_nvmlDeviceGetRowRemapperHistogram, handle_nvmlDeviceGetRowRemapperHistogram},
    {RPC_nvmlDeviceGetSamples, handle_nvmlDeviceGetSamples},
    {RPC_nvmlDeviceGetSerial, handle_nvmlDeviceGetSerial},
    {RPC_nvmlDeviceGetSupportedClocksThrottleReasons, handle_nvmlDeviceGetSupportedClocksThrottleReasons},
    {RPC_nvmlDeviceGetSupportedGraphicsClocks, handle_nvmlDeviceGetSupportedGraphicsClocks},
    {RPC_nvmlDeviceGetSupportedMemoryClocks, handle_nvmlDeviceGetSupportedMemoryClocks},
    {RPC_nvmlDeviceGetSupportedPerformanceStates, handle_nvmlDeviceGetSupportedPerformanceStates},
    {RPC_nvmlDeviceGetTargetFanSpeed, handle_nvmlDeviceGetTargetFanSpeed},
    {RPC_nvmlDeviceGetTemperature, handle_nvmlDeviceGetTemperature},
    {RPC_nvmlDeviceGetTemperatureThreshold, handle_nvmlDeviceGetTemperatureThreshold},
    {RPC_nvmlDeviceGetThermalSettings, handle_nvmlDeviceGetThermalSettings},
    {RPC_nvmlDeviceGetTopologyCommonAncestor, handle_nvmlDeviceGetTopologyCommonAncestor},
    {RPC_nvmlDeviceGetTopologyNearestGpus, handle_nvmlDeviceGetTopologyNearestGpus},
    {RPC_nvmlDeviceGetTotalEccErrors, handle_nvmlDeviceGetTotalEccErrors},
    {RPC_nvmlDeviceGetTotalEnergyConsumption, handle_nvmlDeviceGetTotalEnergyConsumption},
    {RPC_nvmlDeviceGetUUID, handle_nvmlDeviceGetUUID},
    {RPC_nvmlDeviceGetUtilizationRates, handle_nvmlDeviceGetUtilizationRates},
    {RPC_nvmlDeviceGetVbiosVersion, handle_nvmlDeviceGetVbiosVersion},
    {RPC_nvmlDeviceGetViolationStatus, handle_nvmlDeviceGetViolationStatus},
    {RPC_nvmlDeviceOnSameBoard, handle_nvmlDeviceOnSameBoard},
    {RPC_nvmlDeviceValidateInforom, handle_nvmlDeviceValidateInforom},

    // 4.17 Unit Commands
    {RPC_nvmlUnitSetLedState, handle_nvmlUnitSetLedState},

    // 4.20 Event Handling Methods
    {RPC_nvmlDeviceGetSupportedEventTypes, handle_nvmlDeviceGetSupportedEventTypes},
    {RPC_nvmlDeviceRegisterEvents, handle_nvmlDeviceRegisterEvents},
    {RPC_nvmlEventSetCreate, handle_nvmlEventSetCreate},
    {RPC_nvmlEventSetFree, handle_nvmlEventSetFree},
    {RPC_nvmlEventSetWait_v2, handle_nvmlEventSetWait_v2}
};

int request_handler(int connfd)
{
    unsigned int op;
    if (read(connfd, &op, sizeof(unsigned int)) < 0)
        return -1;

    auto it = opHandlers.find(op);
    if (it != opHandlers.end()) {
        std::cout << "found operation!!!" << op << std::endl;
        return it->second(connfd);
    } else {
        std::cerr << "Unknown operation: " << op << std::endl;
        return -1;
    }
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
            // << "request handled by thread: " << std::this_thread::get_id() << std::endl;

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

        // << "Client connected, spawning thread." << std::endl;

        std::thread client_thread(client_handler, connfd);

        // detach the thread so it runs independently
        client_thread.detach();
    }

    close(sockfd);
    return 0;
}
