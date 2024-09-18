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

int request_handler(int connfd)
{
    unsigned int op;
    if (read(connfd, &op, sizeof(unsigned int)) < 0)
        return -1;

    switch (op)
    {
    // 4.11 Initialization and Cleanup
    case RPC_nvmlInitWithFlags:
    {
        unsigned int flags;
        if (read(connfd, &flags, sizeof(unsigned int)) < 0)
            return -1;
        return nvmlInitWithFlags(flags);
    }
    case RPC_nvmlInit_v2:
        return nvmlInit_v2();
    case RPC_nvmlShutdown:
        return nvmlShutdown();
    // 4.14 System Queries
    case RPC_nvmlSystemGetDriverVersion:
    {
        unsigned int length;
        if (read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        char version[length];
        nvmlReturn_t result = nvmlSystemGetDriverVersion(version, length);
        if (write(connfd, version, length) < 0)
            return -1;
        return result;
    }
    case RPC_nvmlSystemGetHicVersion:
    {
        unsigned int hwbcCount;
        if (read(connfd, &hwbcCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlHwbcEntry_t *hwbcEntries =
            (nvmlHwbcEntry_t *)malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));
        nvmlReturn_t result = nvmlSystemGetHicVersion(&hwbcCount, hwbcEntries);
        if (write(connfd, &hwbcCount, sizeof(unsigned int)) < 0 ||
            write(connfd, hwbcEntries, hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
            return -1;
        return result;
    }
    case RPC_nvmlSystemGetNVMLVersion:
    {
        unsigned int length;
        if (read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        char version[length];
        nvmlReturn_t result = nvmlSystemGetNVMLVersion(version, length);
        if (write(connfd, version, length) < 0)
            return -1;
        return result;
    }
    case RPC_nvmlSystemGetProcessName:
    {
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
    case RPC_nvmlSystemGetTopologyGpuSet:
    {
        unsigned int cpuNumber;
        unsigned int count;
        if (read(connfd, &cpuNumber, sizeof(unsigned int)) < 0 ||
            read(connfd, &count, sizeof(unsigned int)) < 0)
            return -1;
        nvmlDevice_t *deviceArray =
            (nvmlDevice_t *)malloc(count * sizeof(nvmlDevice_t));
        nvmlReturn_t result =
            nvmlSystemGetTopologyGpuSet(cpuNumber, &count, deviceArray);
        if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
            write(connfd, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
            return -1;
        return result;
    }

    // 4.15 Unit Queries
    case RPC_nvmlUnitGetCount:
    {
        unsigned int unitCount;
        nvmlReturn_t result = nvmlUnitGetCount(&unitCount);
        if (write(connfd, &unitCount, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlUnitGetDevices:
    {
        nvmlUnit_t unit;
        unsigned int deviceCount;
        if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
            read(connfd, &deviceCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlDevice_t *devices =
            (nvmlDevice_t *)malloc(deviceCount * sizeof(nvmlDevice_t));
        nvmlReturn_t result = nvmlUnitGetDevices(unit, &deviceCount, devices);
        if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0 ||
            write(connfd, devices, deviceCount * sizeof(nvmlDevice_t)) < 0)
            return -1;
        free(devices);
        return result;
    }

    case RPC_nvmlUnitGetFanSpeedInfo:
    {
        nvmlUnit_t unit;
        nvmlUnitFanSpeeds_t fanSpeeds;
        if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds);
        if (write(connfd, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlUnitGetHandleByIndex:
    {
        unsigned int index;
        nvmlUnit_t unit;
        if (read(connfd, &index, sizeof(unsigned int)) < 0)
            return -1;
        nvmlReturn_t result = nvmlUnitGetHandleByIndex(index, &unit);
        if (write(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlUnitGetLedState:
    {
        nvmlUnit_t unit;
        nvmlLedState_t state;
        if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlUnitGetLedState(unit, &state);
        if (write(connfd, &state, sizeof(nvmlLedState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlUnitGetPsuInfo:
    {
        nvmlUnit_t unit;
        nvmlPSUInfo_t psu;
        if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlUnitGetPsuInfo(unit, &psu);
        if (write(connfd, &psu, sizeof(nvmlPSUInfo_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlUnitGetTemperature:
    {
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

    case RPC_nvmlUnitGetUnitInfo:
    {
        nvmlUnit_t unit;
        nvmlUnitInfo_t info;
        if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlUnitGetUnitInfo(unit, &info);
        if (write(connfd, &info, sizeof(nvmlUnitInfo_t)) < 0)
            return -1;
        return result;
    }

    // 4.16 Device Queries
    case RPC_nvmlDeviceGetAPIRestriction:
    {
        nvmlDevice_t device;
        nvmlRestrictedAPI_t apiType;
        nvmlEnableState_t isRestricted;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);
        if (write(connfd, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetAdaptiveClockInfoStatus:
    {
        nvmlDevice_t device;
        unsigned int adaptiveClockStatus;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);
        if (write(connfd, &adaptiveClockStatus, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetApplicationsClock:
    {
        nvmlDevice_t device;
        nvmlClockType_t clockType;
        unsigned int clockMHz;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);
        if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetArchitecture:
    {
        nvmlDevice_t device;
        nvmlDeviceArchitecture_t arch;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetArchitecture(device, &arch);
        if (write(connfd, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetAttributes_v2:
    {
        nvmlDevice_t device;
        nvmlDeviceAttributes_t attributes;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetAttributes_v2(device, &attributes);
        if (write(connfd, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetAutoBoostedClocksEnabled:
    {
        nvmlDevice_t device;
        nvmlEnableState_t isEnabled, defaultIsEnabled;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetAutoBoostedClocksEnabled(
            device, &isEnabled, &defaultIsEnabled);
        if (write(connfd, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
            write(connfd, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetBAR1MemoryInfo:
    {
        nvmlDevice_t device;
        nvmlBAR1Memory_t bar1Memory;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory);
        if (write(connfd, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetBoardId:
    {
        nvmlDevice_t device;
        unsigned int boardId;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetBoardId(device, &boardId);
        if (write(connfd, &boardId, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetBoardPartNumber:
    {
        nvmlDevice_t device;
        unsigned int length;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        char partNumber[length];
        nvmlReturn_t result =
            nvmlDeviceGetBoardPartNumber(device, partNumber, length);
        if (write(connfd, partNumber, length) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetBrand:
    {
        nvmlDevice_t device;
        nvmlBrandType_t type;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetBrand(device, &type);
        if (write(connfd, &type, sizeof(nvmlBrandType_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetBridgeChipInfo:
    {
        nvmlDevice_t device;
        nvmlBridgeChipHierarchy_t bridgeHierarchy;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy);
        if (write(connfd, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetBusType:
    {
        nvmlDevice_t device;
        nvmlBusType_t type;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetBusType(device, &type);
        if (write(connfd, &type, sizeof(nvmlBusType_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetClkMonStatus:
    {
        nvmlDevice_t device;
        nvmlClkMonStatus_t status;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetClkMonStatus(device, &status);
        if (write(connfd, &status, sizeof(nvmlClkMonStatus_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetClock:
    {
        nvmlDevice_t device;
        nvmlClockType_t clockType;
        nvmlClockId_t clockId;
        unsigned int clockMHz;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0 ||
            read(connfd, &clockId, sizeof(nvmlClockId_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);
        if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetClockInfo:
    {
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

    case RPC_nvmlDeviceGetComputeMode:
    {
        nvmlDevice_t device;
        nvmlComputeMode_t mode;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetComputeMode(device, &mode);
        if (write(connfd, &mode, sizeof(nvmlComputeMode_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetComputeRunningProcesses_v3:
    {
        nvmlDevice_t device;
        unsigned int infoCount;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &infoCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlProcessInfo_t *infos =
            (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
        nvmlReturn_t result =
            nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);
        if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
            write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
            return -1;
        free(infos);
        return result;
    }

    case RPC_nvmlDeviceGetCount_v2:
    {
        unsigned int deviceCount;
        nvmlReturn_t result = nvmlDeviceGetCount_v2(&deviceCount);
        if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetCudaComputeCapability:
    {
        nvmlDevice_t device;
        int major, minor;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
        if (write(connfd, &major, sizeof(int)) < 0 ||
            write(connfd, &minor, sizeof(int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetCurrPcieLinkGeneration:
    {
        nvmlDevice_t device;
        unsigned int currLinkGen;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);
        if (write(connfd, &currLinkGen, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetCurrPcieLinkWidth:
    {
        nvmlDevice_t device;
        unsigned int currLinkWidth;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);
        if (write(connfd, &currLinkWidth, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetCurrentClocksThrottleReasons:
    {
        nvmlDevice_t device;
        unsigned long long clocksThrottleReasons;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetCurrentClocksThrottleReasons(
            device, &clocksThrottleReasons);
        if (write(connfd, &clocksThrottleReasons, sizeof(unsigned long long)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDecoderUtilization:
    {
        nvmlDevice_t device;
        unsigned int utilization, samplingPeriodUs;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDecoderUtilization(device, &utilization,
                                                              &samplingPeriodUs);
        if (write(connfd, &utilization, sizeof(unsigned int)) < 0 ||
            write(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDefaultApplicationsClock:
    {
        nvmlDevice_t device;
        nvmlClockType_t clockType;
        unsigned int clockMHz;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);
        if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDefaultEccMode:
    {
        nvmlDevice_t device;
        nvmlEnableState_t defaultMode;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDefaultEccMode(device, &defaultMode);
        if (write(connfd, &defaultMode, sizeof(nvmlEnableState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDetailedEccErrors:
    {
        nvmlDevice_t device;
        nvmlMemoryErrorType_t errorType;
        nvmlEccCounterType_t counterType;
        nvmlEccErrorCounts_t eccCounts;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
            read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDetailedEccErrors(
            device, errorType, counterType, &eccCounts);
        if (write(connfd, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDisplayActive:
    {
        nvmlDevice_t device;
        nvmlEnableState_t isActive;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDisplayActive(device, &isActive);
        if (write(connfd, &isActive, sizeof(nvmlEnableState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDisplayMode:
    {
        nvmlDevice_t device;
        nvmlEnableState_t display;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDisplayMode(device, &display);
        if (write(connfd, &display, sizeof(nvmlEnableState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDynamicPstatesInfo:
    {
        nvmlDevice_t device;
        nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);
        if (write(connfd, &pDynamicPstatesInfo,
                  sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetEccMode:
    {
        nvmlDevice_t device;
        nvmlEnableState_t current, pending;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetEccMode(device, &current, &pending);
        if (write(connfd, &current, sizeof(nvmlEnableState_t)) < 0 ||
            write(connfd, &pending, sizeof(nvmlEnableState_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetEncoderCapacity:
    {
        nvmlDevice_t device;
        nvmlEncoderType_t encoderQueryType;
        unsigned int encoderCapacity;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType,
                                                           &encoderCapacity);
        if (write(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetEncoderSessions:
    {
        nvmlDevice_t device;
        unsigned int sessionCount;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &sessionCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlEncoderSessionInfo_t *sessionInfos = (nvmlEncoderSessionInfo_t *)malloc(
            sessionCount * sizeof(nvmlEncoderSessionInfo_t));
        nvmlReturn_t result =
            nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);
        if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
            write(connfd, sessionInfos,
                  sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
            return -1;
        free(sessionInfos);
        return result;
    }

    case RPC_nvmlDeviceGetEncoderStats:
    {
        nvmlDevice_t device;
        unsigned int sessionCount, averageFps, averageLatency;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetEncoderStats(
            device, &sessionCount, &averageFps, &averageLatency);
        if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
            write(connfd, &averageFps, sizeof(unsigned int)) < 0 ||
            write(connfd, &averageLatency, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetEncoderUtilization:
    {
        nvmlDevice_t device;
        unsigned int utilization, samplingPeriodUs;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetEncoderUtilization(device, &utilization,
                                                              &samplingPeriodUs);
        if (write(connfd, &utilization, sizeof(unsigned int)) < 0 ||
            write(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetEnforcedPowerLimit:
    {
        nvmlDevice_t device;
        unsigned int limit;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetEnforcedPowerLimit(device, &limit);
        if (write(connfd, &limit, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetFBCSessions:
    {
        nvmlDevice_t device;
        unsigned int sessionCount;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &sessionCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlFBCSessionInfo_t *sessionInfo = (nvmlFBCSessionInfo_t *)malloc(
            sessionCount * sizeof(nvmlFBCSessionInfo_t));
        nvmlReturn_t result =
            nvmlDeviceGetFBCSessions(device, &sessionCount, sessionInfo);
        if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
            write(connfd, sessionInfo,
                  sessionCount * sizeof(nvmlFBCSessionInfo_t)) < 0)
            return -1;
        free(sessionInfo);
        return result;
    }

    case RPC_nvmlDeviceGetFBCStats:
    {
        nvmlDevice_t device;
        nvmlFBCStats_t fbcStats;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetFBCStats(device, &fbcStats);
        if (write(connfd, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetFanControlPolicy_v2:
    {
        nvmlDevice_t device;
        unsigned int fan;
        nvmlFanControlPolicy_t policy;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &fan, sizeof(unsigned int)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetFanControlPolicy_v2(device, fan, &policy);
        if (write(connfd, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetFanSpeed:
    {
        nvmlDevice_t device;
        unsigned int speed;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetFanSpeed(device, &speed);
        if (write(connfd, &speed, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetFanSpeed_v2:
    {
        nvmlDevice_t device;
        unsigned int fan, speed;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &fan, sizeof(unsigned int)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetFanSpeed_v2(device, fan, &speed);
        if (write(connfd, &speed, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGpcClkMinMaxVfOffset:
    {
        nvmlDevice_t device;
        int minOffset, maxOffset;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);
        if (write(connfd, &minOffset, sizeof(int)) < 0 ||
            write(connfd, &maxOffset, sizeof(int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGpcClkVfOffset:
    {
        nvmlDevice_t device;
        int offset;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetGpcClkVfOffset(device, &offset);
        if (write(connfd, &offset, sizeof(int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGpuFabricInfo:
    {
        nvmlDevice_t device;
        nvmlGpuFabricInfo_t gpuFabricInfo;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo);
        if (write(connfd, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration:
    {
        nvmlDevice_t device;
        unsigned int maxLinkGenDevice;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &maxLinkGenDevice);
        if (write(connfd, &maxLinkGenDevice, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGpuOperationMode:
    {
        nvmlDevice_t device;
        nvmlGpuOperationMode_t current, pending;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetGpuOperationMode(device, &current, &pending);
        if (write(connfd, &current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
            write(connfd, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGraphicsRunningProcesses_v3:
    {
        nvmlDevice_t device;
        unsigned int infoCount;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &infoCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlProcessInfo_t *infos =
            (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
        nvmlReturn_t result =
            nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, infos);
        if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
            write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
            return -1;
        free(infos);
        return result;
    }

    case RPC_nvmlDeviceGetGspFirmwareMode:
    {
        nvmlDevice_t device;
        unsigned int isEnabled, defaultMode;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);
        if (write(connfd, &isEnabled, sizeof(unsigned int)) < 0 ||
            write(connfd, &defaultMode, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetGspFirmwareVersion:
    {
        nvmlDevice_t device;
        unsigned int length;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        char version[length];
        nvmlReturn_t result = nvmlDeviceGetGspFirmwareVersion(device, version);
        if (write(connfd, version, length) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetHandleByIndex_v2:
    {
        unsigned int index;
        nvmlDevice_t device;
        if (read(connfd, &index, sizeof(unsigned int)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(index, &device);
        if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetHandleByPciBusId_v2:
    {
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

    case RPC_nvmlDeviceGetHandleBySerial:
    {
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

    case RPC_nvmlDeviceGetHandleByUUID:
    {
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

    case RPC_nvmlDeviceGetIndex:
    {
        nvmlDevice_t device;
        unsigned int index;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetIndex(device, &index);
        if (write(connfd, &index, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetInforomConfigurationChecksum:
    {
        nvmlDevice_t device;
        unsigned int checksum;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result =
            nvmlDeviceGetInforomConfigurationChecksum(device, &checksum);
        if (write(connfd, &checksum, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetInforomImageVersion:
    {
        nvmlDevice_t device;
        unsigned int length;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        char version[length];
        nvmlReturn_t result =
            nvmlDeviceGetInforomImageVersion(device, version, length);
        if (write(connfd, version, length) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetInforomVersion:
    {
        nvmlDevice_t device;
        nvmlInforomObject_t object;
        unsigned int length;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &object, sizeof(nvmlInforomObject_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        char version[length];
        nvmlReturn_t result =
            nvmlDeviceGetInforomVersion(device, object, version, length);
        if (write(connfd, version, length) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetIrqNum:
    {
        nvmlDevice_t device;
        unsigned int irqNum;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetIrqNum(device, &irqNum);
        if (write(connfd, &irqNum, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3:
    {
        nvmlDevice_t device;
        unsigned int infoCount;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &infoCount, sizeof(unsigned int)) < 0)
            return -1;
        nvmlProcessInfo_t *infos =
            (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
        nvmlReturn_t result =
            nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, infos);
        if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
            write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
            return -1;
        free(infos);
        return result;
    }

    case RPC_nvmlDeviceGetMaxClockInfo:
    {
        nvmlDevice_t device;
        nvmlClockType_t type;
        unsigned int clock;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &type, sizeof(nvmlClockType_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMaxClockInfo(device, type, &clock);

        if (write(connfd, &clock, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMaxCustomerBoostClock:
    {
        nvmlDevice_t device;
        nvmlClockType_t clockType;
        unsigned int clockMHz;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, &clockMHz);

        if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMaxPcieLinkGeneration:
    {
        nvmlDevice_t device;
        unsigned int maxLinkGen;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkGeneration(device, &maxLinkGen);

        if (write(connfd, &maxLinkGen, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMaxPcieLinkWidth:
    {
        nvmlDevice_t device;
        unsigned int maxLinkWidth;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkWidth(device, &maxLinkWidth);

        if (write(connfd, &maxLinkWidth, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMemoryInfo:
    {
        nvmlDevice_t device;
        nvmlMemory_t memory;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);

        if (write(connfd, &memory, sizeof(nvmlMemory_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMemoryInfo_v2:
    {
        nvmlDevice_t device;
        nvmlMemory_v2_t memoryInfo;

        // Read the device handle from the client
        std::cout << "Reading device handle from client" << std::endl;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        {
            std::cerr << "Failed to read device handle" << std::endl;
            return -1;
        }

        memoryInfo.version = nvmlMemory_v2;

        // Get memory info for the provided device
        std::cout << "Calling nvmlDeviceGetMemoryInfo_v2" << std::endl;
        nvmlReturn_t res = nvmlDeviceGetMemoryInfo_v2(device, &memoryInfo);
        if (res != NVML_SUCCESS)
        {
            std::cerr << "Failed to get memory info: " << nvmlErrorString(res)
                      << std::endl;
            return -1;
        }

        // Send the memory info back to the client
        std::cout << "Sending memory info to client" << std::endl;
        if (write(connfd, &memoryInfo, sizeof(nvmlMemory_v2_t)) < 0)
        {
            std::cerr << "Failed to send memory info to client" << std::endl;
            return -1;
        }

        return NVML_SUCCESS;
    }

    case RPC_nvmlDeviceGetMinMaxClockOfPState:
    {
        nvmlDevice_t device;
        nvmlClockType_t type;
        nvmlPstates_t pstate;
        unsigned int minClockMHz;
        unsigned int maxClockMHz;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &type, sizeof(nvmlClockType_t)) < 0 ||
            read(connfd, &pstate, sizeof(nvmlPstates_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);

        if (write(connfd, &minClockMHz, sizeof(unsigned int)) < 0 ||
            write(connfd, &maxClockMHz, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMinMaxFanSpeed:
    {
        nvmlDevice_t device;
        unsigned int minSpeed;
        unsigned int maxSpeed;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);

        if (write(connfd, &minSpeed, sizeof(unsigned int)) < 0 ||
            write(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMinorNumber:
    {
        nvmlDevice_t device;
        unsigned int minorNumber;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMinorNumber(device, &minorNumber);

        if (write(connfd, &minorNumber, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetMultiGpuBoard:
    {
        nvmlDevice_t device;
        unsigned int multiGpuBool;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetMultiGpuBoard(device, &multiGpuBool);

        if (write(connfd, &multiGpuBool, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetName:
    {
        nvmlDevice_t device;
        unsigned int length;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;

        char name[length];
        nvmlReturn_t result = nvmlDeviceGetName(device, name, length);

        if (write(connfd, name, length) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetNumFans:
    {
        nvmlDevice_t device;
        unsigned int numFans;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetNumFans(device, &numFans);

        if (write(connfd, &numFans, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetNumGpuCores:
    {
        nvmlDevice_t device;
        unsigned int numCores;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetNumGpuCores(device, &numCores);

        if (write(connfd, &numCores, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetP2PStatus:
    {
        nvmlDevice_t device1, device2;
        nvmlGpuP2PCapsIndex_t p2pIndex;
        nvmlGpuP2PStatus_t p2pStatus;

        if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &device2, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, &p2pStatus);

        if (write(connfd, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPciInfo_v3:
    {
        nvmlDevice_t device;
        nvmlPciInfo_t pci;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPciInfo_v3(device, &pci);

        if (write(connfd, &pci, sizeof(nvmlPciInfo_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPcieLinkMaxSpeed:
    {
        nvmlDevice_t device;
        unsigned int maxSpeed;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPcieLinkMaxSpeed(device, &maxSpeed);

        if (write(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPcieReplayCounter:
    {
        nvmlDevice_t device;
        unsigned int value;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPcieReplayCounter(device, &value);

        if (write(connfd, &value, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPcieSpeed:
    {
        nvmlDevice_t device;
        unsigned int pcieSpeed;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed);

        if (write(connfd, &pcieSpeed, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPcieThroughput:
    {
        nvmlDevice_t device;
        nvmlPcieUtilCounter_t counter;
        unsigned int value;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPcieThroughput(device, counter, &value);

        if (write(connfd, &value, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPerformanceState:
    {
        nvmlDevice_t device;
        nvmlPstates_t pState;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPerformanceState(device, &pState);

        if (write(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPersistenceMode:
    {
        nvmlDevice_t device;
        nvmlEnableState_t mode;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPersistenceMode(device, &mode);

        if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerManagementDefaultLimit:
    {
        nvmlDevice_t device;
        unsigned int defaultLimit;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);

        if (write(connfd, &defaultLimit, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerManagementLimit:
    {
        nvmlDevice_t device;
        unsigned int limit;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerManagementLimit(device, &limit);

        if (write(connfd, &limit, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerManagementLimitConstraints:
    {
        nvmlDevice_t device;
        unsigned int minLimit;
        unsigned int maxLimit;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);

        if (write(connfd, &minLimit, sizeof(unsigned int)) < 0 ||
            write(connfd, &maxLimit, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerManagementMode:
    {
        nvmlDevice_t device;
        nvmlEnableState_t mode;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerManagementMode(device, &mode);

        if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerSource:
    {
        nvmlDevice_t device;
        nvmlPowerSource_t powerSource;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerSource(device, &powerSource);

        if (write(connfd, &powerSource, sizeof(nvmlPowerSource_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerState:
    {
        nvmlDevice_t device;
        nvmlPstates_t pState;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerState(device, &pState);

        if (write(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetPowerUsage:
    {
        nvmlDevice_t device;
        unsigned int power;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);

        if (write(connfd, &power, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetProcessUtilization:
    {
        nvmlDevice_t device;
        unsigned int processSamplesCount;
        unsigned long long lastSeenTimeStamp;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &processSamplesCount, sizeof(unsigned int)) < 0 ||
            read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
            return -1;

        nvmlProcessUtilizationSample_t utilization[processSamplesCount];
        nvmlReturn_t result = nvmlDeviceGetProcessUtilization(device, utilization, &processSamplesCount, lastSeenTimeStamp);

        if (write(connfd, &processSamplesCount, sizeof(unsigned int)) < 0 ||
            write(connfd, utilization, processSamplesCount * sizeof(nvmlProcessUtilizationSample_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetRemappedRows:
    {
        nvmlDevice_t device;
        unsigned int corrRows, uncRows, isPending, failureOccurred;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);

        if (write(connfd, &corrRows, sizeof(unsigned int)) < 0 ||
            write(connfd, &uncRows, sizeof(unsigned int)) < 0 ||
            write(connfd, &isPending, sizeof(unsigned int)) < 0 ||
            write(connfd, &failureOccurred, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetRetiredPages:
    {
        nvmlDevice_t device;
        nvmlPageRetirementCause_t cause;
        unsigned int pageCount;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, NULL);

        unsigned long long addresses[pageCount];
        result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, addresses);

        if (write(connfd, &pageCount, sizeof(unsigned int)) < 0 ||
            write(connfd, addresses, pageCount * sizeof(unsigned long long)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetRetiredPagesPendingStatus:
    {
        nvmlDevice_t device;
        nvmlEnableState_t isPending;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetRetiredPagesPendingStatus(device, &isPending);

        if (write(connfd, &isPending, sizeof(nvmlEnableState_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetRetiredPages_v2:
    {
        nvmlDevice_t device;
        nvmlPageRetirementCause_t cause;
        unsigned int pageCount;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &cause, sizeof(nvmlPageRetirementCause_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, NULL, NULL);

        unsigned long long addresses[pageCount], timestamps[pageCount];
        result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, addresses, timestamps);

        if (write(connfd, &pageCount, sizeof(unsigned int)) < 0 ||
            write(connfd, addresses, pageCount * sizeof(unsigned long long)) < 0 ||
            write(connfd, timestamps, pageCount * sizeof(unsigned long long)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetRowRemapperHistogram:
    {
        nvmlDevice_t device;
        nvmlRowRemapperHistogramValues_t values;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetRowRemapperHistogram(device, &values);

        if (write(connfd, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetSamples:
    {
        nvmlDevice_t device;
        nvmlSamplingType_t type;
        unsigned long long lastSeenTimeStamp;
        unsigned int sampleCount;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &type, sizeof(nvmlSamplingType_t)) < 0 ||
            read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
            read(connfd, &sampleCount, sizeof(unsigned int)) < 0)
            return -1;

        nvmlSample_t samples[sampleCount];
        nvmlValueType_t sampleValType;

        nvmlReturn_t result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);

        if (write(connfd, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
            write(connfd, &sampleCount, sizeof(unsigned int)) < 0 ||
            write(connfd, samples, sampleCount * sizeof(nvmlSample_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetSerial:
    {
        nvmlDevice_t device;
        unsigned int length;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;

        char serial[length];
        nvmlReturn_t result = nvmlDeviceGetSerial(device, serial, length);

        if (write(connfd, serial, length) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetSupportedClocksThrottleReasons:
    {
        nvmlDevice_t device;
        unsigned long long supportedClocksThrottleReasons;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetSupportedClocksThrottleReasons(device, &supportedClocksThrottleReasons);

        if (write(connfd, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetSupportedGraphicsClocks:
    {
        nvmlDevice_t device;
        unsigned int memoryClockMHz;
        unsigned int count;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &memoryClockMHz, sizeof(unsigned int)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, NULL);

        unsigned int clocksMHz[count];
        result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, clocksMHz);

        if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
            write(connfd, clocksMHz, count * sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetSupportedMemoryClocks:
    {
        nvmlDevice_t device;
        unsigned int count;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetSupportedMemoryClocks(device, &count, NULL);

        unsigned int clocksMHz[count];
        result = nvmlDeviceGetSupportedMemoryClocks(device, &count, clocksMHz);

        if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
            write(connfd, clocksMHz, count * sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetSupportedPerformanceStates:
    {
        nvmlDevice_t device;
        unsigned int size;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &size, sizeof(unsigned int)) < 0)
            return -1;

        nvmlPstates_t pstates[size];
        nvmlReturn_t result = nvmlDeviceGetSupportedPerformanceStates(device, pstates, size);

        if (write(connfd, pstates, size * sizeof(nvmlPstates_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTargetFanSpeed:
    {
        nvmlDevice_t device;
        unsigned int fan;
        unsigned int targetSpeed;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &fan, sizeof(unsigned int)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTargetFanSpeed(device, fan, &targetSpeed);

        if (write(connfd, &targetSpeed, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTemperature:
    {
        nvmlDevice_t device;
        nvmlTemperatureSensors_t sensorType;
        unsigned int temp;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTemperature(device, sensorType, &temp);

        if (write(connfd, &temp, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTemperatureThreshold:
    {
        nvmlDevice_t device;
        nvmlTemperatureThresholds_t thresholdType;
        unsigned int temp;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, &temp);

        if (write(connfd, &temp, sizeof(unsigned int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetThermalSettings:
    {
        nvmlDevice_t device;
        unsigned int sensorIndex;
        nvmlGpuThermalSettings_t pThermalSettings;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &sensorIndex, sizeof(unsigned int)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetThermalSettings(device, sensorIndex, &pThermalSettings);

        if (write(connfd, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTopologyCommonAncestor:
    {
        nvmlDevice_t device1, device2;
        nvmlGpuTopologyLevel_t pathInfo;

        if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &device2, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &pathInfo);

        if (write(connfd, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTopologyNearestGpus:
    {
        nvmlDevice_t device;
        nvmlGpuTopologyLevel_t level;
        unsigned int count;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, NULL);

        nvmlDevice_t deviceArray[count];
        result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, deviceArray);

        if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
            write(connfd, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTotalEccErrors:
    {
        nvmlDevice_t device;
        nvmlMemoryErrorType_t errorType;
        nvmlEccCounterType_t counterType;
        unsigned long long eccCounts;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
            read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, &eccCounts);

        if (write(connfd, &eccCounts, sizeof(unsigned long long)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetTotalEnergyConsumption:
    {
        nvmlDevice_t device;
        unsigned long long energy;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);

        if (write(connfd, &energy, sizeof(unsigned long long)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetUUID:
    {
        nvmlDevice_t device;
        unsigned int length;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;

        char uuid[length];
        nvmlReturn_t result = nvmlDeviceGetUUID(device, uuid, length);

        if (write(connfd, uuid, length) < 0)
            return -1;

        return result;
    }
    case RPC_nvmlDeviceGetUtilizationRates:
    {
        nvmlDevice_t device;
        nvmlUtilization_t utilization;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);

        if (write(connfd, &utilization, sizeof(nvmlUtilization_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetVbiosVersion:
    {
        nvmlDevice_t device;
        unsigned int length;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;

        char version[length];
        nvmlReturn_t result = nvmlDeviceGetVbiosVersion(device, version, length);

        if (write(connfd, version, length) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceGetViolationStatus:
    {
        nvmlDevice_t device;
        nvmlPerfPolicyType_t perfPolicyType;
        nvmlViolationTime_t violTime;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceGetViolationStatus(device, perfPolicyType, &violTime);

        if (write(connfd, &violTime, sizeof(nvmlViolationTime_t)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceOnSameBoard:
    {
        nvmlDevice_t device1, device2;
        int onSameBoard;

        if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &device2, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceOnSameBoard(device1, device2, &onSameBoard);

        if (write(connfd, &onSameBoard, sizeof(int)) < 0)
            return -1;

        return result;
    }

    case RPC_nvmlDeviceValidateInforom:
    {
        nvmlDevice_t device;

        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;

        nvmlReturn_t result = nvmlDeviceValidateInforom(device);

        return result;
    }

    // 4.17 Unit Commands
    case RPC_nvmlUnitSetLedState:
    {
        nvmlUnit_t unit;
        nvmlLedColor_t color;
        if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
            read(connfd, &color, sizeof(nvmlLedColor_t)) < 0)
            return -1;
        return nvmlUnitSetLedState(unit, color);
    }
    // 4.20 Event Handling Methods
    case RPC_nvmlDeviceGetSupportedEventTypes:
    {
        nvmlDevice_t device;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        unsigned long long eventTypes;
        nvmlReturn_t result = nvmlDeviceGetSupportedEventTypes(device, &eventTypes);
        if (write(connfd, &eventTypes, sizeof(unsigned long long)) < 0)
            return -1;
        return result;
    }
    case RPC_nvmlDeviceRegisterEvents:
    {
        nvmlDevice_t device;
        unsigned long long eventTypes;
        nvmlEventSet_t set;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &eventTypes, sizeof(unsigned long long)) < 0 ||
            read(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceRegisterEvents(device, eventTypes, set);
        return result;
    }
    case RPC_nvmlEventSetCreate:
    {
        nvmlEventSet_t set;
        nvmlReturn_t result = nvmlEventSetCreate(&set);
        if (write(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
            return -1;
        return result;
    }
    case RPC_nvmlEventSetFree:
    {
        nvmlEventSet_t set;
        if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
            return -1;
        return nvmlEventSetFree(set);
    }
    case RPC_nvmlEventSetWait_v2:
    {
        nvmlEventSet_t set;
        nvmlEventData_t data;
        unsigned int timeoutms;
        if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0 ||
            read(connfd, &timeoutms, sizeof(unsigned int)) < 0)
            return -1;
        nvmlReturn_t result = nvmlEventSetWait_v2(set, &data, timeoutms);
        if (write(connfd, &data, sizeof(unsigned long long)) < 0)
            return -1;
        return result;
    }

    default:
        printf("Unknown operation %d\n", op);
        break;
    }
    return -1;
}

void client_handler(int connfd)
{
    std::cout << "handling client in thread: " << std::this_thread::get_id()
              << std::endl;

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
        std::cout << "SCUDA_PORT not defined, defaulting to: " << "14833"
                  << std::endl;
        port = DEFAULT_PORT;
    }
    else
    {
        port = atoi(p);
        std::cout << "Using SCUDA_PORT: " << port << std::endl;
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
