#include <arpa/inet.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <string>
#include <cstring>
#include <nvml.h>
#include <future>
#include <sys/socket.h>

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
        nvmlHwbcEntry_t *hwbcEntries = (nvmlHwbcEntry_t *)malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));
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
        nvmlDevice_t *deviceArray = (nvmlDevice_t *)malloc(count * sizeof(nvmlDevice_t));
        nvmlReturn_t result = nvmlSystemGetTopologyGpuSet(cpuNumber, &count, deviceArray);
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
        nvmlDevice_t *devices = (nvmlDevice_t *)malloc(deviceCount * sizeof(nvmlDevice_t));
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
        nvmlReturn_t result = nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);
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
        nvmlReturn_t result = nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);
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
        nvmlReturn_t result = nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);
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
        nvmlReturn_t result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);
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
        nvmlReturn_t result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);
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

    case RPC_nvmlDeviceGetC2cModeInfoV:
    {
        nvmlDevice_t device;
        nvmlC2cModeInfo_v1_t c2cModeInfo;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetC2cModeInfoV(device, &c2cModeInfo);
        if (write(connfd, &c2cModeInfo, sizeof(nvmlC2cModeInfo_v1_t)) < 0)
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
        nvmlReturn_t result = nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);
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

    case RPC_nvmlDeviceGetClockOffsets:
    {
        nvmlDevice_t device;
        nvmlClockOffset_t info;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetClockOffsets(device, &info);
        if (write(connfd, &info, sizeof(nvmlClockOffset_t)) < 0)
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
        nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *)malloc(infoCount * sizeof(nvmlProcessInfo_t));
        nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);
        if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
            write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
            return -1;
        free(infos);
        return result;
    }

    case RPC_nvmlDeviceGetConfComputeGpuAttestationReport:
    {
        nvmlDevice_t device;
        nvmlConfComputeGpuAttestationReport_t gpuAtstReport;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetConfComputeGpuAttestationReport(device, &gpuAtstReport);
        if (write(connfd, &gpuAtstReport, sizeof(nvmlConfComputeGpuAttestationReport_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetConfComputeGpuCertificate:
    {
        nvmlDevice_t device;
        nvmlConfComputeGpuCertificate_t gpuCert;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetConfComputeGpuCertificate(device, &gpuCert);
        if (write(connfd, &gpuCert, sizeof(nvmlConfComputeGpuCertificate_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetConfComputeMemSizeInfo:
    {
        nvmlDevice_t device;
        nvmlConfComputeMemSizeInfo_t memInfo;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetConfComputeMemSizeInfo(device, &memInfo);
        if (write(connfd, &memInfo, sizeof(nvmlConfComputeMemSizeInfo_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetConfComputeProtectedMemoryUsage:
    {
        nvmlDevice_t device;
        nvmlMemory_t memory;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetConfComputeProtectedMemoryUsage(device, &memory);
        if (write(connfd, &memory, sizeof(nvmlMemory_t)) < 0)
            return -1;
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
        nvmlReturn_t result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
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
        nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);
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
        nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);
        if (write(connfd, &currLinkWidth, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetCurrentClocksEventReasons:
    {
        nvmlDevice_t device;
        unsigned long long clocksEventReasons;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetCurrentClocksEventReasons(device, &clocksEventReasons);
        if (write(connfd, &clocksEventReasons, sizeof(unsigned long long)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetCurrentClocksThrottleReasons:
    {
        nvmlDevice_t device;
        unsigned long long clocksThrottleReasons;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clocksThrottleReasons);
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
        nvmlReturn_t result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);
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
        nvmlReturn_t result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);
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
        nvmlReturn_t result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, &eccCounts);
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

    case RPC_nvmlDeviceGetDriverModel_v2:
    {
        nvmlDevice_t device;
        nvmlDriverModel_t current, pending;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDriverModel_v2(device, &current, &pending);
        if (write(connfd, &current, sizeof(nvmlDriverModel_t)) < 0 ||
            write(connfd, &pending, sizeof(nvmlDriverModel_t)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetDynamicPstatesInfo:
    {
        nvmlDevice_t device;
        nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);
        if (write(connfd, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
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
        nvmlReturn_t result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, &encoderCapacity);
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
        nvmlEncoderSessionInfo_t *sessionInfos = (nvmlEncoderSessionInfo_t *)malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));
        nvmlReturn_t result = nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);
        if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
            write(connfd, sessionInfos, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
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
        nvmlReturn_t result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);
        if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
            write(connfd, &averageFps, sizeof(unsigned int)) < 0 ||
            write(connfd, &averageLatency, sizeof(unsigned int)) < 0)
            return -1;
        return result;
    }

    case RPC_nvmlDeviceGetName:
    {
        nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        unsigned int length;
        if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
            read(connfd, &length, sizeof(unsigned int)) < 0)
            return -1;
        nvmlReturn_t result = nvmlDeviceGetName(device, name, length);

        printf("received device name response: %s\n", name);

        if (write(connfd, name, length) < 0)
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
            std::cerr << "Failed to get memory info: " << nvmlErrorString(res) << std::endl;
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
    default:
        printf("Unknown operation %d\n", op);
        break;
    }
    return -1;
}

void client_handler(int connfd)
{
    std::cout << "handling client in thread: " << std::this_thread::get_id() << std::endl;

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
        std::future<int> request_future = std::async(std::launch::async, [connfd]() {
            std::cout << "request handled by thread: " << std::this_thread::get_id() << std::endl;

            return request_handler(connfd);
        });

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
        std::cout << "SCUDA_PORT not defined, defaulting to: " << "14833" << std::endl;
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
