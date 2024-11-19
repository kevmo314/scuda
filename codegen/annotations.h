#include <nvml.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 */
nvmlReturn_t nvmlInit_v2();
/**
 * @param flags SEND_ONLY
 */
nvmlReturn_t nvmlInitWithFlags(unsigned int flags);
/**
 */
nvmlReturn_t nvmlShutdown();
/**
 * @disabled
 * @param result SEND_ONLY
 */
const char *nvmlErrorString(nvmlReturn_t result);
/**
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length);
/**
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length);
/**
 * @param cudaDriverVersion RECV_ONLY
 */
nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion);
/**
 * @param cudaDriverVersion RECV_ONLY
 */
nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion);
/**
 * @param pid SEND_ONLY
 * @param length SEND_ONLY
 * @param name RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length);
/**
 * @param unitCount RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount);
/**
 * @param index SEND_ONLY
 * @param unit RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit);
/**
 * @param unit SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info);
/**
 * @param unit SEND_ONLY
 * @param state RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state);
/**
 * @param unit SEND_ONLY
 * @param psu RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu);
/**
 * @param unit SEND_ONLY
 * @param type SEND_ONLY
 * @param temp RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp);
/**
 * @param unit SEND_ONLY
 * @param fanSpeeds RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds);
/**
 * @param unit SEND_ONLY
 * @param deviceCount SEND_RECV
 * @param devices RECV_ONLY LENGTH:deviceCount
 */
nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices);
/**
 * @param hwbcCount SEND_RECV
 * @param hwbcEntries RECV_ONLY LENGTH:hwbcCount
 */
nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries);
/**
 * @param deviceCount RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);
/**
 * @param device SEND_ONLY
 * @param attributes RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t *attributes);
/**
 * @param index SEND_ONLY
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);
/**
 * @param serial SEND_ONLY NULL_TERMINATED
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device);
/**
 * @param uuid SEND_ONLY NULL_TERMINATED
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device);
/**
 * @param pciBusId SEND_ONLY NULL_TERMINATED
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param name RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param type RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type);
/**
 * @param device SEND_ONLY
 * @param index RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param serial RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param nodeSetSize SEND_ONLY
 * @param nodeSet RECV_ONLY LENGTH:nodeSetSize
 * @param scope SEND_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, nvmlAffinityScope_t scope);
/**
 * @param device SEND_ONLY
 * @param cpuSetSize SEND_ONLY
 * @param cpuSet RECV_ONLY LENGTH:cpuSetSize
 * @param scope SEND_ONLY
 */
nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, nvmlAffinityScope_t scope);
/**
 * @param device SEND_ONLY
 * @param cpuSetSize SEND_ONLY
 * @param cpuSet RECV_ONLY LENGTH:cpuSetSize
 */
nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param pathInfo RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo);
/**
 * @param device SEND_ONLY
 * @param level SEND_ONLY
 * @param count SEND_RECV
 * @param deviceArray RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray);
/**
 * @param cpuNumber SEND_ONLY
 * @param count SEND_RECV
 * @param deviceArray RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param p2pIndex SEND_ONLY
 * @param p2pStatus RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t *p2pStatus);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param uuid RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length);
/**
 * @param vgpuInstance SEND_ONLY
 * @param size SEND_ONLY
 * @param mdevUuid RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size);
/**
 * @param device SEND_ONLY
 * @param minorNumber RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param partNumber RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param object SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char *version, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param checksum RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int *checksum);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param display RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display);
/**
 * @param device SEND_ONLY
 * @param isActive RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode);
/**
 * @param device SEND_ONLY
 * @param pci RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci);
/**
 * @param device SEND_ONLY
 * @param maxLinkGen RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGen);
/**
 * @param device SEND_ONLY
 * @param maxLinkGenDevice RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int *maxLinkGenDevice);
/**
 * @param device SEND_ONLY
 * @param maxLinkWidth RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int *maxLinkWidth);
/**
 * @param device SEND_ONLY
 * @param currLinkGen RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *currLinkGen);
/**
 * @param device SEND_ONLY
 * @param currLinkWidth RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *currLinkWidth);
/**
 * @param device SEND_ONLY
 * @param counter SEND_ONLY
 * @param value RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value);
/**
 * @param device SEND_ONLY
 * @param value RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param clock RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param clock RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockId SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 * @param count SEND_RECV
 * @param clocksMHz RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count, unsigned int *clocksMHz);
/**
 * @param device SEND_ONLY
 * @param memoryClockMHz SEND_ONLY
 * @param count SEND_RECV
 * @param clocksMHz RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz);
/**
 * @param device SEND_ONLY
 * @param isEnabled RECV_ONLY
 * @param defaultIsEnabled RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t *isEnabled, nvmlEnableState_t *defaultIsEnabled);
/**
 * @param device SEND_ONLY
 * @param enabled SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled);
/**
 * @param device SEND_ONLY
 * @param enabled SEND_ONLY
 * @param flags SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags);
/**
 * @param device SEND_ONLY
 * @param speed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param speed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int *speed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param targetSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int *targetSpeed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan);
/**
 * @param device SEND_ONLY
 * @param minSpeed RECV_ONLY
 * @param maxSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int *minSpeed, unsigned int *maxSpeed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param policy RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t *policy);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param policy SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy);
/**
 * @param device SEND_ONLY
 * @param numFans RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans);
/**
 * @param device SEND_ONLY
 * @param sensorType SEND_ONLY
 * @param temp RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp);
/**
 * @param device SEND_ONLY
 * @param thresholdType SEND_ONLY
 * @param temp RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp);
/**
 * @param device SEND_ONLY
 * @param thresholdType SEND_ONLY
 * @param temp SEND_RECV
 */
nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp);
/**
 * @param device SEND_ONLY
 * @param sensorIndex SEND_ONLY
 * @param pThermalSettings RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings);
/**
 * @param device SEND_ONLY
 * @param pState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState);
/**
 * @param device SEND_ONLY
 * @param clocksThrottleReasons RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons);
/**
 * @param device SEND_ONLY
 * @param supportedClocksThrottleReasons RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons);
/**
 * @param device SEND_ONLY
 * @param pState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t *mode);
/**
 * @param device SEND_ONLY
 * @param limit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit);
/**
 * @param device SEND_ONLY
 * @param minLimit RECV_ONLY
 * @param maxLimit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit);
/**
 * @param device SEND_ONLY
 * @param defaultLimit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int *defaultLimit);
/**
 * @param device SEND_ONLY
 * @param power RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power);
/**
 * @param device SEND_ONLY
 * @param energy RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy);
/**
 * @param device SEND_ONLY
 * @param limit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit);
/**
 * @param device SEND_ONLY
 * @param current RECV_ONLY
 * @param pending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t *current, nvmlGpuOperationMode_t *pending);
/**
 * @param device SEND_ONLY
 * @param memory RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
/**
 * @param device SEND_ONLY
 * @param memory RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode);
/**
 * @param device SEND_ONLY
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor);
/**
 * @param device SEND_ONLY
 * @param current RECV_ONLY
 * @param pending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending);
/**
 * @param device SEND_ONLY
 * @param defaultMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t *defaultMode);
/**
 * @param device SEND_ONLY
 * @param boardId RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId);
/**
 * @param device SEND_ONLY
 * @param multiGpuBool RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool);
/**
 * @param device SEND_ONLY
 * @param errorType SEND_ONLY
 * @param counterType SEND_ONLY
 * @param eccCounts RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long *eccCounts);
/**
 * @param device SEND_ONLY
 * @param errorType SEND_ONLY
 * @param counterType SEND_ONLY
 * @param eccCounts RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t *eccCounts);
/**
 * @param device SEND_ONLY
 * @param errorType SEND_ONLY
 * @param counterType SEND_ONLY
 * @param locationType SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long *count);
/**
 * @param device SEND_ONLY
 * @param utilization RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization);
/**
 * @param device SEND_ONLY
 * @param utilization RECV_ONLY
 * @param samplingPeriodUs RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
/**
 * @param device SEND_ONLY
 * @param encoderQueryType SEND_ONLY
 * @param encoderCapacity RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity);
/**
 * @param device SEND_ONLY
 * @param sessionCount RECV_ONLY
 * @param averageFps RECV_ONLY
 * @param averageLatency RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency);
/**
 * @param device SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfos RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos);
/**
 * @param device SEND_ONLY
 * @param utilization RECV_ONLY
 * @param samplingPeriodUs RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);
/**
 * @param device SEND_ONLY
 * @param fbcStats RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats);
/**
 * @param device SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfo RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo);
/**
 * @param device SEND_ONLY
 * @param current RECV_ONLY
 * @param pending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param bridgeHierarchy RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t *bridgeHierarchy);
/**
 * @param device SEND_ONLY
 * @param infoCount SEND_RECV
 * @param infos RECV_ONLY LENGTH:infoCount
 */
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
/**
 * @param device SEND_ONLY
 * @param infoCount SEND_RECV
 * @param infos RECV_ONLY LENGTH:infoCount
 */
nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
/**
 * @param device SEND_ONLY
 * @param infoCount SEND_RECV
 * @param infos RECV_ONLY LENGTH:infoCount
 */
nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param onSameBoard RECV_ONLY
 */
nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int *onSameBoard);
/**
 * @param device SEND_ONLY
 * @param apiType SEND_ONLY
 * @param isRestricted RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t *isRestricted);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param lastSeenTimeStamp SEND_ONLY
 * @param sampleValType RECV_ONLY
 * @param sampleCount SEND_RECV
 * @param samples RECV_ONLY LENGTH:sampleCount
 */
nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *sampleCount, nvmlSample_t *samples);
/**
 * @param device SEND_ONLY
 * @param bar1Memory RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t *bar1Memory);
/**
 * @param device SEND_ONLY
 * @param perfPolicyType SEND_ONLY
 * @param violTime RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t *violTime);
/**
 * @param device SEND_ONLY
 * @param irqNum RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int *irqNum);
/**
 * @param device SEND_ONLY
 * @param numCores RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int *numCores);
/**
 * @param device SEND_ONLY
 * @param powerSource RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t *powerSource);
/**
 * @param device SEND_ONLY
 * @param busWidth RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int *busWidth);
/**
 * @param device SEND_ONLY
 * @param maxSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int *maxSpeed);
/**
 * @param device SEND_ONLY
 * @param pcieSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int *pcieSpeed);
/**
 * @param device SEND_ONLY
 * @param adaptiveClockStatus RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int *adaptiveClockStatus);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode);
/**
 * @param device SEND_ONLY
 * @param pid SEND_ONLY
 * @param stats RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t *stats);
/**
 * @param device SEND_ONLY
 * @param count SEND_RECV
 * @param pids RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count, unsigned int *pids);
/**
 * @param device SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int *bufferSize);
/**
 * @param device SEND_ONLY
 * @param cause SEND_ONLY
 * @param pageCount SEND_RECV
 * @param addresses RECV_ONLY LENGTH:pageCount
 */
nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses);
/**
 * @param device SEND_ONLY
 * @param cause SEND_ONLY
 * @param pageCount SEND_RECV
 * @param addresses RECV_ONLY LENGTH:pageCount
 * @param timestamps RECV_ONLY LENGTH:pageCount
 */
nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps);
/**
 * @param device SEND_ONLY
 * @param isPending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t *isPending);
/**
 * @param device SEND_ONLY
 * @param corrRows RECV_ONLY
 * @param uncRows RECV_ONLY
 * @param isPending RECV_ONLY
 * @param failureOccurred RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows, unsigned int *isPending, unsigned int *failureOccurred);
/**
 * @param device SEND_ONLY
 * @param values RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t *values);
/**
 * @param device SEND_ONLY
 * @param arch RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t *arch);
/**
 * @param unit SEND_ONLY
 * @param color SEND_ONLY
 */
nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode);
/**
 * @param device SEND_ONLY
 * @param ecc SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc);
/**
 * @param device SEND_ONLY
 * @param counterType SEND_ONLY
 */
nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType);
/**
 * @param device SEND_ONLY
 * @param driverModel SEND_ONLY
 * @param flags SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags);
/**
 * @param device SEND_ONLY
 * @param minGpuClockMHz SEND_ONLY
 * @param maxGpuClockMHz SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param minMemClockMHz SEND_ONLY
 * @param maxMemClockMHz SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param memClockMHz SEND_ONLY
 * @param graphicsClockMHz SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz);
/**
 * @param device SEND_ONLY
 * @param status RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status);
/**
 * @param device SEND_ONLY
 * @param limit SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode);
/**
 * @param device SEND_ONLY
 * @param apiType SEND_ONLY
 * @param isRestricted SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param isActive RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param version RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int *version);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int *capResult);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param pci RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param counterValue RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long *counterValue);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param control SEND_ONLY DEREFERENCE
 * @param reset SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control, unsigned int reset);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param control RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param rxcounter RECV_ONLY
 * @param txcounter RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long *rxcounter, unsigned long long *txcounter);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param freeze SEND_ONLY
 */
nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param pNvLinkDeviceType RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType);
/**
 * @param set RECV_ONLY
 */
nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set);
/**
 * @param device SEND_ONLY
 * @param eventTypes SEND_ONLY
 * @param set SEND_ONLY
 */
nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set);
/**
 * @param device SEND_ONLY
 * @param eventTypes RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes);
/**
 * @param set SEND_ONLY
 * @param data RECV_ONLY
 * @param timeoutms SEND_ONLY
 */
nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms);
/**
 * @param set SEND_ONLY
 */
nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set);
/**
 * @param pciInfo SEND_RECV
 * @param newState SEND_ONLY
 */
nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t newState);
/**
 * @param pciInfo SEND_RECV
 * @param currentState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo, nvmlEnableState_t *currentState);
/**
 * @param pciInfo SEND_RECV
 * @param gpuState SEND_ONLY
 * @param linkState SEND_ONLY
 */
nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState);
/**
 * @param pciInfo SEND_RECV
 */
nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo);
/**
 * @param device SEND_ONLY
 * @param valuesCount SEND_ONLY
 * @param values RECV_ONLY LENGTH:valuesCount
 */
nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values);
/**
 * @param device SEND_ONLY
 * @param valuesCount SEND_ONLY
 * @param values RECV_ONLY LENGTH:valuesCount
 */
nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t *values);
/**
 * @param device SEND_ONLY
 * @param pVirtualMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t *pVirtualMode);
/**
 * @param device SEND_ONLY
 * @param pHostVgpuMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t *pHostVgpuMode);
/**
 * @param device SEND_ONLY
 * @param virtualMode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode);
/**
 * @param device SEND_ONLY
 * @param pGridLicensableFeatures RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);
/**
 * @param device SEND_ONLY
 * @param processSamplesCount SEND_RECV
 * @param utilization RECV_ONLY LENGTH:processSamplesCount
 * @param lastSeenTimeStamp SEND_ONLY
 */
nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp);
/**
 * @param device SEND_ONLY
 * @param version RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char *version);
/**
 * @param device SEND_ONLY
 * @param isEnabled RECV_ONLY
 * @param defaultMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int *isEnabled, unsigned int *defaultMode);
/**
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int *capResult);
/**
 * @param device SEND_ONLY
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int *capResult);
/**
 * @param device SEND_ONLY
 * @param vgpuCount SEND_RECV
 * @param vgpuTypeIds RECV_ONLY LENGTH:vgpuCount
 */
nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds);
/**
 * @param device SEND_ONLY
 * @param vgpuCount SEND_RECV
 * @param vgpuTypeIds RECV_ONLY LENGTH:vgpuCount
 */
nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuTypeId_t *vgpuTypeIds);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param size RECV_ONLY
 * @param vgpuTypeClass RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param size SEND_RECV
 * @param vgpuTypeName RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param gpuInstanceProfileId RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param deviceID RECV_ONLY
 * @param subsystemID RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param fbSize RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param numDisplayHeads RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param displayIndex SEND_ONLY
 * @param xdim RECV_ONLY
 * @param ydim RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param size SEND_ONLY
 * @param vgpuTypeLicenseString RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param frameRateLimit RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit);
/**
 * @param device SEND_ONLY
 * @param vgpuTypeId SEND_ONLY
 * @param vgpuInstanceCount RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param vgpuInstanceCountPerVm RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm);
/**
 * @param device SEND_ONLY
 * @param vgpuCount SEND_RECV
 * @param vgpuInstances RECV_ONLY LENGTH:vgpuCount
 */
nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount, nvmlVgpuInstance_t *vgpuInstances);
/**
 * @param vgpuInstance SEND_ONLY
 * @param size SEND_ONLY
 * @param vmId RECV_ONLY LENGTH:size
 * @param vmIdType RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, nvmlVgpuVmIdType_t *vmIdType);
/**
 * @param vgpuInstance SEND_ONLY
 * @param size SEND_ONLY
 * @param uuid RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size);
/**
 * @param vgpuInstance SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char *version, unsigned int length);
/**
 * @param vgpuInstance SEND_ONLY
 * @param fbUsage RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage);
/**
 * @param vgpuInstance SEND_ONLY
 * @param licensed RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int *licensed);
/**
 * @param vgpuInstance SEND_ONLY
 * @param vgpuTypeId RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t *vgpuTypeId);
/**
 * @param vgpuInstance SEND_ONLY
 * @param frameRateLimit RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit);
/**
 * @param vgpuInstance SEND_ONLY
 * @param eccMode RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *eccMode);
/**
 * @param vgpuInstance SEND_ONLY
 * @param encoderCapacity RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity);
/**
 * @param vgpuInstance SEND_ONLY
 * @param encoderCapacity SEND_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity);
/**
 * @param vgpuInstance SEND_ONLY
 * @param sessionCount RECV_ONLY
 * @param averageFps RECV_ONLY
 * @param averageLatency RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency);
/**
 * @param vgpuInstance SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfo RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfo);
/**
 * @param vgpuInstance SEND_ONLY
 * @param fbcStats RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t *fbcStats);
/**
 * @param vgpuInstance SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfo RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, nvmlFBCSessionInfo_t *sessionInfo);
/**
 * @param vgpuInstance SEND_ONLY
 * @param gpuInstanceId RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId);
/**
 * @param vgpuInstance SEND_ONLY
 * @param length SEND_RECV
 * @param vgpuPciId RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char *vgpuPciId, unsigned int *length);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int *capResult);
/**
 * @param vgpuInstance SEND_ONLY
 * @param bufferSize SEND_RECV
 * @param vgpuMetadata RECV_ONLY LENGTH:bufferSize
 */
nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize);
/**
 * @param device SEND_ONLY
 * @param bufferSize SEND_RECV
 * @param pgpuMetadata RECV_ONLY LENGTH:bufferSize
 */
nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize);
/**
 * @param vgpuMetadata SEND_RECV
 * @param pgpuMetadata RECV_ONLY
 * @param compatibilityInfo RECV_ONLY
 */
nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata, nvmlVgpuPgpuMetadata_t *pgpuMetadata, nvmlVgpuPgpuCompatibility_t *compatibilityInfo);
/**
 * @param device SEND_ONLY
 * @param bufferSize SEND_RECV
 * @param pgpuMetadata RECV_ONLY LENGTH:bufferSize
 */
nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize);
/**
 * @param device SEND_ONLY
 * @param pSchedulerLog RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t *pSchedulerLog);
/**
 * @param device SEND_ONLY
 * @param pSchedulerState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t *pSchedulerState);
/**
 * @param device SEND_ONLY
 * @param pCapabilities RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t *pCapabilities);
/**
 * @param supported RECV_ONLY
 * @param current RECV_ONLY
 */
nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current);
/**
 * @param vgpuVersion RECV_ONLY
 */
nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion);
/**
 * @param device SEND_ONLY
 * @param lastSeenTimeStamp SEND_ONLY
 * @param sampleValType SEND_RECV
 * @param vgpuInstanceSamplesCount SEND_RECV
 * @param utilizationSamples RECV_ONLY LENGTH:vgpuInstanceSamplesCount
 */
nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples);
/**
 * @param device SEND_ONLY
 * @param lastSeenTimeStamp SEND_ONLY
 * @param vgpuProcessSamplesCount SEND_RECV
 * @param utilizationSamples RECV_ONLY LENGTH:vgpuProcessSamplesCount
 */
nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t *utilizationSamples);
/**
 * @param vgpuInstance SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t *mode);
/**
 * @param vgpuInstance SEND_ONLY
 * @param count SEND_RECV
 * @param pids RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids);
/**
 * @param vgpuInstance SEND_ONLY
 * @param pid SEND_ONLY
 * @param stats RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t *stats);
/**
 * @param vgpuInstance SEND_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance);
/**
 * @param vgpuInstance SEND_ONLY
 * @param licenseInfo RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t *licenseInfo);
/**
 * @param deviceCount RECV_ONLY
 */
nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount);
/**
 * @param index SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t *info);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 * @param activationStatus RECV_ONLY
 */
nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t *activationStatus);
/**
 * @param device SEND_ONLY
 * @param currentMode RECV_ONLY
 * @param pendingMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode, unsigned int *pendingMode);
/**
 * @param device SEND_ONLY
 * @param profile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t *info);
/**
 * @param device SEND_ONLY
 * @param profile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t *info);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param placements RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param gpuInstance RECV_ONLY
 */
nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstance);
/**
 * @disabled
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param placement SEND_ONLY
 * @param gpuInstance RECV_ONLY
 */
nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement, nvmlGpuInstance_t *gpuInstance);
/**
 * @param gpuInstance SEND_ONLY
 */
nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param gpuInstances RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t *gpuInstances, unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param id SEND_ONLY
 * @param gpuInstance RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t *gpuInstance);
/**
 * @param gpuInstance SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t *info);
/**
 * @param gpuInstance SEND_ONLY
 * @param profile SEND_ONLY
 * @param engProfile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info);
/**
 * @param gpuInstance SEND_ONLY
 * @param profile SEND_ONLY
 * @param engProfile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param placements RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t *placements, unsigned int *count);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param computeInstance RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstance);
/**
 * @disabled
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param placement SEND_ONLY DEREFERENCE
 * @param computeInstance RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t *placement, nvmlComputeInstance_t *computeInstance);
/**
 * @param computeInstance SEND_ONLY
 */
nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param computeInstances RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t *computeInstances, unsigned int *count);
/**
 * @param gpuInstance SEND_ONLY
 * @param id SEND_ONLY
 * @param computeInstance RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t *computeInstance);
/**
 * @param computeInstance SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t *info);
/**
 * @param device SEND_ONLY
 * @param isMigDevice RECV_ONLY
 */
nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice);
/**
 * @param device SEND_ONLY
 * @param id RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id);
/**
 * @param device SEND_ONLY
 * @param id RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id);
/**
 * @param device SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param index SEND_ONLY
 * @param migDevice RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t *migDevice);
/**
 * @param migDevice SEND_ONLY
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t *device);
/**
 * @param device SEND_ONLY
 * @param type RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t *type);
/**
 * @param device SEND_ONLY
 * @param pDynamicPstatesInfo RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param speed SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed);
/**
 * @param device SEND_ONLY
 * @param offset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int *offset);
/**
 * @param device SEND_ONLY
 * @param offset SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset);
/**
 * @param device SEND_ONLY
 * @param offset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int *offset);
/**
 * @param device SEND_ONLY
 * @param offset SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param pstate SEND_ONLY
 * @param minClockMHz RECV_ONLY
 * @param maxClockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int *minClockMHz, unsigned int *maxClockMHz);
/**
 * @param device SEND_ONLY
 * @param size SEND_ONLY
 * @param pstates RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size);
/**
 * @param device SEND_ONLY
 * @param minOffset RECV_ONLY
 * @param maxOffset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset);
/**
 * @param device SEND_ONLY
 * @param minOffset RECV_ONLY
 * @param maxOffset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int *minOffset, int *maxOffset);
/**
 * @param device SEND_ONLY
 * @param gpuFabricInfo RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t *gpuFabricInfo);
/**
 * @param metricsGet RECV_ONLY
 */
nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t *metricsGet);
/**
 * @param gpmSample SEND_ONLY
 */
nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample);
/**
 * @param gpmSample RECV_ONLY
 */
nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t *gpmSample);
/**
 * @param device SEND_ONLY
 * @param gpmSample SEND_ONLY
 */
nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample);
/**
 * @param device SEND_ONLY
 * @param gpuInstanceId SEND_ONLY
 * @param gpmSample SEND_ONLY
 */
nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample);
/**
 * @param device SEND_ONLY
 * @param gpmSupport RECV_ONLY
 */
nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t *gpmSupport);
/**
 * @disabled
 * @param device SEND_ONLY
 * @param state RECV_ONLY
 */
nvmlReturn_t nvmlDeviceCcuGetStreamState(nvmlDevice_t device, unsigned int *state);
/**
 * @disabled
 * @param device SEND_ONLY
 * @param state SEND_ONLY
 */
nvmlReturn_t nvmlDeviceCcuSetStreamState(nvmlDevice_t device, unsigned int state);
/**
 * @param device SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t *info);
/**
 * @disabled
 * @param error SEND_ONLY
 * @param pStr SEND_RECV
 */
CUresult cuGetErrorString(CUresult error, const char **pStr);
/**
 * @disabled
 * @param error SEND_ONLY
 * @param pStr SEND_RECV
 */
CUresult cuGetErrorName(CUresult error, const char **pStr);
/**
 * @param Flags SEND_ONLY
 */
CUresult cuInit(unsigned int Flags);
/**
 * @param driverVersion RECV_ONLY
 */
CUresult cuDriverGetVersion(int *driverVersion);
/**
 * @param device RECV_ONLY
 * @param ordinal SEND_ONLY
 */
CUresult cuDeviceGet(CUdevice *device, int ordinal);
/**
 * @param count RECV_ONLY
 */
CUresult cuDeviceGetCount(int *count);
/**
 * @param len SEND_ONLY
 * @param name RECV_ONLY LENGTH:len
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
/**
 * @param uuid RECV_ONLY SIZE:16
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
/**
 * @param uuid RECV_ONLY SIZE:16
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
/**
 * @param luid RECV_ONLY NULL_TERMINATED
 * @param deviceNodeMask RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
/**
 * @param bytes RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
/**
 * @param maxWidthInElements RECV_ONLY
 * @param format SEND_ONLY
 * @param numChannels SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev);
/**
 * @param pi RECV_ONLY
 * @param attrib SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
/**
 * @disabled
 * @param nvSciSyncAttrList SEND_RECV
 * @param dev SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags);
/**
 * @param dev SEND_ONLY
 * @param pool SEND_ONLY
 */
CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
/**
 * @param pool RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev);
/**
 * @param pool_out RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev);
/**
 * @param pi RECV_ONLY
 * @param type SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev);
/**
 * @param target SEND_ONLY
 * @param scope SEND_ONLY
 */
CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope);
/**
 * @param prop RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
/**
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
/**
 * @param pctx RECV_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
/**
 * @param dev SEND_ONLY
 */
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev);
/**
 * @param dev SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags);
/**
 * @param dev SEND_ONLY
 * @param flags RECV_ONLY
 * @param active RECV_ONLY
 */
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
/**
 * @param dev SEND_ONLY
 */
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev);
/**
 * @param pctx RECV_ONLY
 * @param flags SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
/**
 * @param pctx RECV_ONLY
 * @param numParams SEND_ONLY
 * @param paramsArray RECV_ONLY LENGTH:numParams
 * @param flags SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams, unsigned int flags, CUdevice dev);
/**
 * @param ctx SEND_ONLY
 */
CUresult cuCtxDestroy_v2(CUcontext ctx);
/**
 * @param ctx SEND_ONLY
 */
CUresult cuCtxPushCurrent_v2(CUcontext ctx);
/**
 * @param pctx RECV_ONLY
 */
CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
/**
 * @param ctx SEND_ONLY
 */
CUresult cuCtxSetCurrent(CUcontext ctx);
/**
 * @param pctx RECV_ONLY
 */
CUresult cuCtxGetCurrent(CUcontext *pctx);
/**
 * @param device RECV_ONLY
 */
CUresult cuCtxGetDevice(CUdevice *device);
/**
 * @param flags RECV_ONLY
 */
CUresult cuCtxGetFlags(unsigned int *flags);
/**
 * @param ctx SEND_ONLY
 * @param ctxId RECV_ONLY
 */
CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId);
/**
 */
CUresult cuCtxSynchronize();
/**
 * @param limit SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuCtxSetLimit(CUlimit limit, size_t value);
/**
 * @param pvalue RECV_ONLY
 * @param limit SEND_ONLY
 */
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit);
/**
 * @param pconfig RECV_ONLY
 */
CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig);
/**
 * @param config SEND_ONLY
 */
CUresult cuCtxSetCacheConfig(CUfunc_cache config);
/**
 * @param pConfig RECV_ONLY
 */
CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig);
/**
 * @param config SEND_ONLY
 */
CUresult cuCtxSetSharedMemConfig(CUsharedconfig config);
/**
 * @param ctx SEND_ONLY
 * @param version RECV_ONLY
 */
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
/**
 * @param leastPriority RECV_ONLY
 * @param greatestPriority RECV_ONLY
 */
CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
/**
 */
CUresult cuCtxResetPersistingL2Cache();
/**
 * @param pExecAffinity RECV_ONLY
 * @param type SEND_ONLY
 */
CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type);
/**
 * @param pctx RECV_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags);
/**
 * @param ctx SEND_ONLY
 */
CUresult cuCtxDetach(CUcontext ctx);
/**
 * @param module RECV_ONLY
 * @param fname SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleLoad(CUmodule *module, const char *fname);
/**
 * @disabled
 * @param module RECV_ONLY
 * @param image SEND_ONLY
 */
CUresult cuModuleLoadData(CUmodule *module, const void *image);
/**
 * @disabled
 * @param module RECV_ONLY
 * @param image SEND_ONLY NULL_TERMINATED
 * @param numOptions SEND_ONLY
 * @param options SEND_ONLY LENGTH:numOptions
 * @param optionValues SEND_RECV
 */
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
/**
 * @disabled
 * @param module SEND_RECV
 * @param fatCubin SEND_RECV
 */
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
/**
 * @param hmod SEND_ONLY
 */
CUresult cuModuleUnload(CUmodule hmod);
/**
 * @param mode SEND_RECV
 */
CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode);
/**
 * @param hfunc RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
/**
 * @param dptr RECV_ONLY
 * @param bytes RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
/**
 * @param numOptions SEND_ONLY
 * @param options SEND_RECV
 * @param optionValues SEND_RECV
 * @param stateOut SEND_RECV
 */
CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut);
/**
 * @param state SEND_ONLY
 * @param type SEND_ONLY
 * @param data SEND_RECV
 * @param size SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 * @param numOptions SEND_ONLY
 * @param options SEND_ONLY LENGTH:numOptions
 * @param optionValues SEND_ONLY LENGTH:numOptions
 */
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);
/**
 * @param state SEND_ONLY
 * @param type SEND_ONLY
 * @param path SEND_ONLY NULL_TERMINATED
 * @param numOptions SEND_ONLY
 * @param options SEND_ONLY LENGTH:numOptions
 * @param optionValues SEND_ONLY LENGTH:numOptions
 */
CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues);
/**
 * @param state SEND_ONLY
 * @param cubinOut RECV_ONLY
 * @param sizeOut RECV_ONLY
 */
CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
/**
 * @param state SEND_ONLY
 */
CUresult cuLinkDestroy(CUlinkState state);
/**
 * @param pTexRef RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
/**
 * @param pSurfRef RECV_ONLY
 * @param hmod SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name);
/**
 * @disabled
 * @param library RECV_ONLY
 * @param code SEND_ONLY
 * @param numJitOptions SEND_ONLY
 * @param jitOptions SEND_ONLY LENGTH:numJitOptions
 * @param jitOptionsValues SEND_ONLY LENGTH:numJitOptions
 * @param numLibraryOptions SEND_ONLY
 * @param libraryOptions SEND_ONLY LENGTH:numLibraryOptions
 * @param libraryOptionValues SEND_ONLY LENGTH:numLibraryOptions
 */
CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
/**
 * @param library RECV_ONLY
 * @param fileName SEND_ONLY NULL_TERMINATED
 * @param numJitOptions SEND_ONLY
 * @param jitOptions SEND_ONLY LENGTH:numJitOptions
 * @param jitOptionsValues SEND_ONLY LENGTH:numJitOptions
 * @param numLibraryOptions SEND_ONLY
 * @param libraryOptions SEND_ONLY LENGTH:numLibraryOptions
 * @param libraryOptionValues SEND_ONLY LENGTH:numLibraryOptions
 */
CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
/**
 * @param library SEND_ONLY
 */
CUresult cuLibraryUnload(CUlibrary library);
/**
 * @param pKernel RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name);
/**
 * @param pMod RECV_ONLY
 * @param library SEND_ONLY
 */
CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library);
/**
 * @param pFunc RECV_ONLY
 * @param kernel SEND_ONLY
 */
CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel);
/**
 * @param dptr RECV_ONLY
 * @param bytes RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name);
/**
 * @param dptr RECV_ONLY
 * @param bytes RECV_ONLY
 * @param library SEND_ONLY
 * @param name SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name);
/**
 * @param fptr RECV_ONLY
 * @param library SEND_ONLY
 * @param symbol SEND_ONLY NULL_TERMINATED
 */
CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library, const char *symbol);
/**
 * @param pi SEND_RECV
 * @param attrib SEND_ONLY
 * @param kernel SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev);
/**
 * @param attrib SEND_ONLY
 * @param val SEND_ONLY
 * @param kernel SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev);
/**
 * @param kernel SEND_ONLY
 * @param config SEND_ONLY
 * @param dev SEND_ONLY
 */
CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev);
/**
 * @param free SEND_RECV
 * @param total SEND_RECV
 */
CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
/**
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
/**
 * @param dptr SEND_RECV
 * @param pPitch SEND_RECV
 * @param WidthInBytes SEND_ONLY
 * @param Height SEND_ONLY
 * @param ElementSizeBytes SEND_ONLY
 */
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
/**
 * @param dptr SEND_ONLY
 */
CUresult cuMemFree_v2(CUdeviceptr dptr);
/**
 * @param pbase SEND_RECV
 * @param psize SEND_RECV
 * @param dptr SEND_ONLY
 */
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
/**
 * @param pp RECV_ONLY
 * @param bytesize SEND_ONLY
 */
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize);
/**
 * @param p SEND_ONLY
 */
CUresult cuMemFreeHost(void *p);
/**
 * @param pp RECV_ONLY
 * @param bytesize SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
/**
 * @param pdptr SEND_RECV
 * @param p SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags);
/**
 * @param pFlags SEND_RECV
 * @param p SEND_ONLY
 */
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p);
/**
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
/**
 * @param dev SEND_RECV
 * @param pciBusId SEND_ONLY NULL_TERMINATED
 */
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);
/**
 * @param len SEND_ONLY
 * @param pciBusId RECV_ONLY LENGTH:len
 * @param dev SEND_ONLY
 */
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
/**
 * @param pHandle SEND_RECV
 * @param event SEND_ONLY
 */
CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event);
/**
 * @param phEvent SEND_RECV
 * @param handle SEND_ONLY
 */
CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle);
/**
 * @param pHandle SEND_RECV
 * @param dptr SEND_ONLY
 */
CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr);
/**
 * @param pdptr SEND_RECV
 * @param handle SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags);
/**
 * @param dptr SEND_ONLY
 */
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr);
/**
 * @param p SEND_RECV
 * @param bytesize SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags);
/**
 * @param p SEND_RECV
 */
CUresult cuMemHostUnregister(void *p);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
/**
 * @param dstDevice SEND_ONLY
 * @param dstContext SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param srcContext SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
/**
 * @param dstDevice SEND_ONLY
 * @param srcHost SEND_RECV
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
/**
 * @param dstHost SEND_RECV
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
/**
 * @param dstDevice SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
/**
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
/**
 * @param dstDevice SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
/**
 * @disabled
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcHost SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
/**
 * @param dstHost SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
/**
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 */
CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 */
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 */
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 */
CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 */
CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param dstContext SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param srcContext SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param srcHost SEND_RECV
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
/**
 * @param dstHost SEND_RECV
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
/**
 * @disabled
 * @param dstArray SEND_ONLY
 * @param dstOffset SEND_ONLY
 * @param srcHost SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
/**
 * @param dstHost SEND_RECV
 * @param srcArray SEND_ONLY
 * @param srcOffset SEND_ONLY
 * @param ByteCount SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
/**
 * @disabled
 * @param pCopy SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param uc SEND_ONLY
 * @param N SEND_ONLY
 */
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
/**
 * @param dstDevice SEND_ONLY
 * @param us SEND_ONLY
 * @param N SEND_ONLY
 */
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N);
/**
 * @param dstDevice SEND_ONLY
 * @param ui SEND_ONLY
 * @param N SEND_ONLY
 */
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
/**
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param uc SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 */
CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
/**
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param us SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 */
CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
/**
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param ui SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 */
CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
/**
 * @param dstDevice SEND_ONLY
 * @param uc SEND_ONLY
 * @param N SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param us SEND_ONLY
 * @param N SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param ui SEND_ONLY
 * @param N SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param uc SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param us SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
/**
 * @param dstDevice SEND_ONLY
 * @param dstPitch SEND_ONLY
 * @param ui SEND_ONLY
 * @param Width SEND_ONLY
 * @param Height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
/**
 * @param pHandle SEND_RECV
 * @param pAllocateArray SEND_RECV
 */
CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
/**
 * @param pArrayDescriptor SEND_RECV
 * @param hArray SEND_ONLY
 */
CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
/**
 * @param sparseProperties SEND_RECV
 * @param array SEND_ONLY
 */
CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUarray array);
/**
 * @param sparseProperties SEND_RECV
 * @param mipmap SEND_ONLY
 */
CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap);
/**
 * @param memoryRequirements SEND_RECV
 * @param array SEND_ONLY
 * @param device SEND_ONLY
 */
CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUarray array, CUdevice device);
/**
 * @param memoryRequirements SEND_RECV
 * @param mipmap SEND_ONLY
 * @param device SEND_ONLY
 */
CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap, CUdevice device);
/**
 * @param pPlaneArray SEND_RECV
 * @param hArray SEND_ONLY
 * @param planeIdx SEND_ONLY
 */
CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx);
/**
 * @param hArray SEND_ONLY
 */
CUresult cuArrayDestroy(CUarray hArray);
/**
 * @param pHandle SEND_RECV
 * @param pAllocateArray SEND_RECV
 */
CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
/**
 * @param pArrayDescriptor SEND_RECV
 * @param hArray SEND_ONLY
 */
CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray);
/**
 * @param pHandle SEND_RECV
 * @param pMipmappedArrayDesc SEND_RECV
 * @param numMipmapLevels SEND_ONLY
 */
CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels);
/**
 * @param pLevelArray SEND_RECV
 * @param hMipmappedArray SEND_ONLY
 * @param level SEND_ONLY
 */
CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
/**
 * @param hMipmappedArray SEND_ONLY
 */
CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
/**
 * @param handle SEND_RECV
 * @param dptr SEND_ONLY
 * @param size SEND_ONLY
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags);
/**
 * @param ptr SEND_RECV
 * @param size SEND_ONLY
 * @param alignment SEND_ONLY
 * @param addr SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 */
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
/**
 * @param handle SEND_RECV
 * @param size SEND_ONLY
 * @param prop SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
/**
 * @param handle SEND_ONLY
 */
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 * @param offset SEND_ONLY
 * @param handle SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
/**
 * @param mapInfoList SEND_RECV
 * @param count SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 */
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
/**
 * @param ptr SEND_ONLY
 * @param size SEND_ONLY
 * @param desc SEND_RECV
 * @param count SEND_ONLY
 */
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
/**
 * @param flags SEND_RECV
 * @param location SEND_RECV
 * @param ptr SEND_ONLY
 */
CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr);
/**
 * @param shareableHandle SEND_RECV
 * @param handle SEND_ONLY
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
/**
 * @param handle SEND_RECV
 * @param osHandle SEND_RECV
 * @param shHandleType SEND_ONLY
 */
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
/**
 * @param granularity SEND_RECV
 * @param prop SEND_RECV
 * @param option SEND_ONLY
 */
CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
/**
 * @param prop SEND_RECV
 * @param handle SEND_ONLY
 */
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle);
/**
 * @param handle SEND_RECV
 * @param addr SEND_RECV
 */
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr);
/**
 * @param dptr SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
/**
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
/**
 * @param pool SEND_ONLY
 * @param minBytesToKeep SEND_ONLY
 */
CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
/**
 * @param pool SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
/**
 * @param pool SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
/**
 * @param pool SEND_ONLY
 * @param map SEND_RECV
 * @param count SEND_ONLY
 */
CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count);
/**
 * @param flags SEND_RECV
 * @param memPool SEND_ONLY
 * @param location SEND_RECV
 */
CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location);
/**
 * @param pool SEND_RECV
 * @param poolProps SEND_RECV
 */
CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
/**
 * @param pool SEND_ONLY
 */
CUresult cuMemPoolDestroy(CUmemoryPool pool);
/**
 * @param dptr SEND_RECV
 * @param bytesize SEND_ONLY
 * @param pool SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
/**
 * @param handle_out SEND_RECV
 * @param pool SEND_ONLY
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
/**
 * @param pool_out SEND_RECV
 * @param handle SEND_RECV
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle, CUmemAllocationHandleType handleType, unsigned long long flags);
/**
 * @param shareData_out SEND_RECV
 * @param ptr SEND_ONLY
 */
CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr);
/**
 * @param ptr_out SEND_RECV
 * @param pool SEND_ONLY
 * @param shareData SEND_RECV
 */
CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData);
/**
 * @param data SEND_RECV
 * @param attribute SEND_ONLY
 * @param ptr SEND_ONLY
 */
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
/**
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param dstDevice SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
/**
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param device SEND_ONLY
 */
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
/**
 * @param data SEND_RECV
 * @param dataSize SEND_ONLY
 * @param attribute SEND_ONLY
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
/**
 * @param data SEND_RECV
 * @param dataSizes SEND_RECV
 * @param attributes SEND_RECV
 * @param numAttributes SEND_ONLY
 * @param devPtr SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
/**
 * @param value SEND_RECV
 * @param attribute SEND_ONLY
 * @param ptr SEND_ONLY
 */
CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr);
/**
 * @param numAttributes SEND_ONLY
 * @param attributes SEND_RECV
 * @param data SEND_RECV
 * @param ptr SEND_ONLY
 */
CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);
/**
 * @param phStream SEND_RECV
 * @param Flags SEND_ONLY
 */
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
/**
 * @param phStream SEND_RECV
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
/**
 * @param hStream SEND_ONLY
 * @param priority SEND_RECV
 */
CUresult cuStreamGetPriority(CUstream hStream, int *priority);
/**
 * @param hStream SEND_ONLY
 * @param flags SEND_RECV
 */
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags);
/**
 * @param hStream SEND_ONLY
 * @param streamId SEND_RECV
 */
CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId);
/**
 * @param hStream SEND_ONLY
 * @param pctx SEND_RECV
 */
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx);
/**
 * @param hStream SEND_ONLY
 * @param hEvent SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
/**
 * @param hStream SEND_ONLY
 * @param callback SEND_ONLY
 * @param userData SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
/**
 * @param hStream SEND_ONLY
 * @param mode SEND_ONLY
 */
CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);
/**
 * @param mode SEND_RECV
 */
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode);
/**
 * @param hStream SEND_ONLY
 * @param phGraph SEND_RECV
 */
CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph);
/**
 * @param hStream SEND_ONLY
 * @param captureStatus SEND_RECV
 */
CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus);
/**
 * @disabled
 * @param hStream SEND_ONLY
 * @param captureStatus_out RECV_ONLY
 * @param id_out RECV_ONLY
 * @param graph_out RECV_ONLY
 * @param numDependencies_out RECV_ONLY
 * @param dependencies_out RECV_ONLY LENGTH:numDependencies_out
 */
CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
/**
 * @param hStream SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
/**
 * @param hStream SEND_ONLY
 * @param dptr SEND_ONLY
 * @param length SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
/**
 * @param hStream SEND_ONLY
 */
CUresult cuStreamQuery(CUstream hStream);
/**
 * @param hStream SEND_ONLY
 */
CUresult cuStreamSynchronize(CUstream hStream);
/**
 * @param hStream SEND_ONLY
 */
CUresult cuStreamDestroy_v2(CUstream hStream);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
CUresult cuStreamCopyAttributes(CUstream dst, CUstream src);
/**
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out SEND_RECV
 */
CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value_out);
/**
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value);
/**
 * @param phEvent SEND_RECV
 * @param Flags SEND_ONLY
 */
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags);
/**
 * @param hEvent SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
/**
 * @param hEvent SEND_ONLY
 * @param hStream SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags);
/**
 * @param hEvent SEND_ONLY
 */
CUresult cuEventQuery(CUevent hEvent);
/**
 * @param hEvent SEND_ONLY
 */
CUresult cuEventSynchronize(CUevent hEvent);
/**
 * @param hEvent SEND_ONLY
 */
CUresult cuEventDestroy_v2(CUevent hEvent);
/**
 * @param pMilliseconds SEND_RECV
 * @param hStart SEND_ONLY
 * @param hEnd SEND_ONLY
 */
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
/**
 * @param extMem_out SEND_RECV
 * @param memHandleDesc SEND_RECV
 */
CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc);
/**
 * @param devPtr SEND_RECV
 * @param extMem SEND_ONLY
 * @param bufferDesc SEND_RECV
 */
CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc);
/**
 * @param mipmap SEND_RECV
 * @param extMem SEND_ONLY
 * @param mipmapDesc SEND_RECV
 */
CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc);
/**
 * @param extMem SEND_ONLY
 */
CUresult cuDestroyExternalMemory(CUexternalMemory extMem);
/**
 * @param extSem_out SEND_RECV
 * @param semHandleDesc SEND_RECV
 */
CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc);
/**
 * @param extSemArray SEND_RECV
 * @param paramsArray SEND_RECV
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
/**
 * @param extSemArray SEND_RECV
 * @param paramsArray SEND_RECV
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
/**
 * @param extSem SEND_ONLY
 */
CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param addr SEND_ONLY
 * @param value SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param count SEND_ONLY
 * @param paramArray SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
/**
 * @param pi SEND_RECV
 * @param attrib SEND_ONLY
 * @param hfunc SEND_ONLY
 */
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
/**
 * @param hfunc SEND_ONLY
 * @param attrib SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
/**
 * @param hfunc SEND_ONLY
 * @param config SEND_ONLY
 */
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
/**
 * @param hfunc SEND_ONLY
 * @param config SEND_ONLY
 */
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config);
/**
 * @param hmod SEND_RECV
 * @param hfunc SEND_ONLY
 */
CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc);

// TODO: Spec from here

/**
 * @param f SEND_ONLY
 * @param gridDimX SEND_ONLY
 * @param gridDimY SEND_ONLY
 * @param gridDimZ SEND_ONLY
 * @param blockDimX SEND_ONLY
 * @param blockDimY SEND_ONLY
 * @param blockDimZ SEND_ONLY
 * @param sharedMemBytes SEND_ONLY
 * @param hStream SEND_ONLY
 * @param kernelParams SEND_ONLY
 * @param extra SEND_ONLY
 */
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
/**
 * @disabled
 * @param config SEND_ONLY DEREFERENCE
 * @param f SEND_ONLY
 * @param kernelParams SEND_ONLY
 * @param extra SEND_ONLY
 */
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra);
/**
 * @param f SEND_ONLY
 * @param gridDimX SEND_ONLY
 * @param gridDimY SEND_ONLY
 * @param gridDimZ SEND_ONLY
 * @param blockDimX SEND_ONLY
 * @param blockDimY SEND_ONLY
 * @param blockDimZ SEND_ONLY
 * @param sharedMemBytes SEND_ONLY
 * @param hStream SEND_ONLY
 * @param kernelParams SEND_RECV
 */
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams);
/**
 * @param launchParamsList SEND_RECV
 * @param numDevices SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags);
/**
 * @param hStream SEND_ONLY
 * @param fn SEND_ONLY
 * @param userData SEND_RECV
 */
CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);
/**
 * @param hfunc SEND_ONLY
 * @param x SEND_ONLY
 * @param y SEND_ONLY
 * @param z SEND_ONLY
 */
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
/**
 * @param hfunc SEND_ONLY
 * @param bytes SEND_ONLY
 */
CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
/**
 * @param hfunc SEND_ONLY
 * @param numbytes SEND_ONLY
 */
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
/**
 * @param hfunc SEND_ONLY
 * @param offset SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value);
/**
 * @param hfunc SEND_ONLY
 * @param offset SEND_ONLY
 * @param value SEND_ONLY
 */
CUresult cuParamSetf(CUfunction hfunc, int offset, float value);
/**
 * @param hfunc SEND_ONLY
 * @param offset SEND_ONLY
 * @param ptr SEND_RECV
 * @param numbytes SEND_ONLY
 */
CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
/**
 * @param f SEND_ONLY
 */
CUresult cuLaunch(CUfunction f);
/**
 * @param f SEND_ONLY
 * @param grid_width SEND_ONLY
 * @param grid_height SEND_ONLY
 */
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
/**
 * @param f SEND_ONLY
 * @param grid_width SEND_ONLY
 * @param grid_height SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
/**
 * @param hfunc SEND_ONLY
 * @param texunit SEND_ONLY
 * @param hTexRef SEND_ONLY
 */
CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
/**
 * @param phGraph SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param copyParams SEND_RECV
 * @param ctx SEND_ONLY
 */
CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param memsetParams SEND_RECV
 * @param ctx SEND_ONLY
 */
CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param childGraph SEND_ONLY
 */
CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph);
/**
 * @param hNode SEND_ONLY
 * @param phGraph SEND_RECV
 */
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 */
CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
/**
 * @param hNode SEND_ONLY
 * @param event_out SEND_RECV
 */
CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
/**
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
/**
 * @param hNode SEND_ONLY
 * @param event_out SEND_RECV
 */
CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
/**
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out SEND_RECV
 */
CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out SEND_RECV
 */
CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddBatchMemOpNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams_out SEND_RECV
 */
CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out SEND_RECV
 */
CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out);
/**
 * @param phGraphNode SEND_RECV
 * @param hGraph SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param dptr SEND_ONLY
 */
CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr);
/**
 * @param hNode SEND_ONLY
 * @param dptr_out SEND_RECV
 */
CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out);
/**
 * @param device SEND_ONLY
 */
CUresult cuDeviceGraphMemTrim(CUdevice device);
/**
 * @param device SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value);
/**
 * @param device SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void *value);
/**
 * @param phGraphClone SEND_RECV
 * @param originalGraph SEND_ONLY
 */
CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph);
/**
 * @param phNode SEND_RECV
 * @param hOriginalNode SEND_ONLY
 * @param hClonedGraph SEND_ONLY
 */
CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
/**
 * @param hNode SEND_ONLY
 * @param type SEND_RECV
 */
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type);
/**
 * @param hGraph SEND_ONLY
 * @param nodes SEND_RECV
 * @param numNodes SEND_RECV
 */
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
/**
 * @param hGraph SEND_ONLY
 * @param rootNodes SEND_RECV
 * @param numRootNodes SEND_RECV
 */
CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes);
/**
 * @param hGraph SEND_ONLY
 * @param from SEND_RECV
 * @param to SEND_RECV
 * @param numEdges SEND_RECV
 */
CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges);
/**
 * @param hNode SEND_ONLY
 * @param dependencies SEND_RECV
 * @param numDependencies SEND_RECV
 */
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies);
/**
 * @param hNode SEND_ONLY
 * @param dependentNodes SEND_RECV
 * @param numDependentNodes SEND_RECV
 */
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes);
/**
 * @param hGraph SEND_ONLY
 * @param from SEND_RECV
 * @param to SEND_RECV
 * @param numDependencies SEND_ONLY
 */
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
/**
 * @param hGraph SEND_ONLY
 * @param from SEND_RECV
 * @param to SEND_RECV
 * @param numDependencies SEND_ONLY
 */
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
/**
 * @param hNode SEND_ONLY
 */
CUresult cuGraphDestroyNode(CUgraphNode hNode);
/**
 * @param phGraphExec SEND_RECV
 * @param hGraph SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
/**
 * @param phGraphExec SEND_RECV
 * @param hGraph SEND_ONLY
 * @param instantiateParams SEND_RECV
 */
CUresult cuGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param flags SEND_RECV
 */
CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t *flags);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param copyParams SEND_RECV
 * @param ctx SEND_ONLY
 */
CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param memsetParams SEND_RECV
 * @param ctx SEND_ONLY
 */
CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param childGraph SEND_ONLY
 */
CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_ONLY
 */
CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_RECV
 */
CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int *isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
/**
 * @param hGraphExec SEND_ONLY
 * @param hStream SEND_ONLY
 */
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
/**
 * @param hGraphExec SEND_ONLY
 */
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec);
/**
 * @param hGraph SEND_ONLY
 */
CUresult cuGraphDestroy(CUgraph hGraph);
/**
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param resultInfo SEND_RECV
 */
CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo *resultInfo);
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out SEND_RECV
 */
CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value);
/**
 * @param hGraph SEND_ONLY
 * @param path SEND_RECV
 * @param flags SEND_ONLY
 */
CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags);
/**
 * @param object_out SEND_RECV
 * @param ptr SEND_RECV
 * @param destroy SEND_ONLY
 * @param initialRefcount SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuUserObjectRetain(CUuserObject object, unsigned int count);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuUserObjectRelease(CUuserObject object, unsigned int count);
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count);
/**
 * @param numBlocks SEND_RECV
 * @param func SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
/**
 * @param numBlocks SEND_RECV
 * @param func SEND_ONLY
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
/**
 * @disabled
 * @param minGridSize SEND_RECV
 * @param blockSize SEND_RECV
 * @param func SEND_ONLY
 * @param blockSizeToDynamicSMemSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param blockSizeLimit SEND_ONLY
 */
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
/**
 * @disabled
 * @param minGridSize SEND_RECV
 * @param blockSize SEND_RECV
 * @param func SEND_ONLY
 * @param blockSizeToDynamicSMemSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param blockSizeLimit SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
/**
 * @param dynamicSmemSize SEND_RECV
 * @param func SEND_ONLY
 * @param numBlocks SEND_ONLY
 * @param blockSize SEND_ONLY
 */
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
/**
 * @param clusterSize SEND_RECV
 * @param func SEND_ONLY
 * @param config SEND_RECV
 */
CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func, const CUlaunchConfig *config);
/**
 * @param numClusters SEND_RECV
 * @param func SEND_ONLY
 * @param config SEND_RECV
 */
CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func, const CUlaunchConfig *config);
/**
 * @param hTexRef SEND_ONLY
 * @param hArray SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
/**
 * @param hTexRef SEND_ONLY
 * @param hMipmappedArray SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
/**
 * @param ByteOffset SEND_RECV
 * @param hTexRef SEND_ONLY
 * @param dptr SEND_ONLY
 * @param bytes SEND_ONLY
 */
CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
/**
 * @param hTexRef SEND_ONLY
 * @param desc SEND_RECV
 * @param dptr SEND_ONLY
 * @param Pitch SEND_ONLY
 */
CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
/**
 * @param hTexRef SEND_ONLY
 * @param fmt SEND_ONLY
 * @param NumPackedComponents SEND_ONLY
 */
CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
/**
 * @param hTexRef SEND_ONLY
 * @param dim SEND_ONLY
 * @param am SEND_ONLY
 */
CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
/**
 * @param hTexRef SEND_ONLY
 * @param fm SEND_ONLY
 */
CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
/**
 * @param hTexRef SEND_ONLY
 * @param fm SEND_ONLY
 */
CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
/**
 * @param hTexRef SEND_ONLY
 * @param bias SEND_ONLY
 */
CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
/**
 * @param hTexRef SEND_ONLY
 * @param minMipmapLevelClamp SEND_ONLY
 * @param maxMipmapLevelClamp SEND_ONLY
 */
CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
/**
 * @param hTexRef SEND_ONLY
 * @param maxAniso SEND_ONLY
 */
CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
/**
 * @param hTexRef SEND_ONLY
 * @param pBorderColor SEND_RECV
 */
CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor);
/**
 * @param hTexRef SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
/**
 * @param pdptr SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef);
/**
 * @param phArray SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef);
/**
 * @param phMipmappedArray SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef);
/**
 * @param pam SEND_RECV
 * @param hTexRef SEND_ONLY
 * @param dim SEND_ONLY
 */
CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim);
/**
 * @param pfm SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
/**
 * @param pFormat SEND_RECV
 * @param pNumChannels SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef);
/**
 * @param pfm SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
/**
 * @param pbias SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef);
/**
 * @param pminMipmapLevelClamp SEND_RECV
 * @param pmaxMipmapLevelClamp SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef);
/**
 * @param pmaxAniso SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef);
/**
 * @param pBorderColor SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef);
/**
 * @param pFlags SEND_RECV
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef);
/**
 * @param pTexRef SEND_RECV
 */
CUresult cuTexRefCreate(CUtexref *pTexRef);
/**
 * @param hTexRef SEND_ONLY
 */
CUresult cuTexRefDestroy(CUtexref hTexRef);
/**
 * @param hSurfRef SEND_ONLY
 * @param hArray SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
/**
 * @param phArray SEND_RECV
 * @param hSurfRef SEND_ONLY
 */
CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef);
/**
 * @param pTexObject SEND_RECV
 * @param pResDesc SEND_RECV
 * @param pTexDesc SEND_RECV
 * @param pResViewDesc SEND_RECV
 */
CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);
/**
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectDestroy(CUtexObject texObject);
/**
 * @param pResDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject);
/**
 * @param pTexDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject);
/**
 * @param pResViewDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject);
/**
 * @param pSurfObject SEND_RECV
 * @param pResDesc SEND_RECV
 */
CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc);
/**
 * @param surfObject SEND_ONLY
 */
CUresult cuSurfObjectDestroy(CUsurfObject surfObject);
/**
 * @param pResDesc SEND_RECV
 * @param surfObject SEND_ONLY
 */
CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject);
/**
 * @param tensorMap SEND_RECV
 * @param tensorDataType SEND_ONLY
 * @param tensorRank SEND_ONLY
 * @param globalAddress SEND_RECV
 * @param globalDim SEND_RECV
 * @param globalStrides SEND_RECV
 * @param boxDim SEND_RECV
 * @param elementStrides SEND_RECV
 * @param interleave SEND_ONLY
 * @param swizzle SEND_ONLY
 * @param l2Promotion SEND_ONLY
 * @param oobFill SEND_ONLY
 */
CUresult cuTensorMapEncodeTiled(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const cuuint32_t *boxDim, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
/**
 * @param tensorMap SEND_RECV
 * @param tensorDataType SEND_ONLY
 * @param tensorRank SEND_ONLY
 * @param globalAddress SEND_RECV
 * @param globalDim SEND_RECV
 * @param globalStrides SEND_RECV
 * @param pixelBoxLowerCorner SEND_RECV
 * @param pixelBoxUpperCorner SEND_RECV
 * @param channelsPerPixel SEND_ONLY
 * @param pixelsPerColumn SEND_ONLY
 * @param elementStrides SEND_RECV
 * @param interleave SEND_ONLY
 * @param swizzle SEND_ONLY
 * @param l2Promotion SEND_ONLY
 * @param oobFill SEND_ONLY
 */
CUresult cuTensorMapEncodeIm2col(CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim, const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner, const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
/**
 * @param tensorMap SEND_RECV
 * @param globalAddress SEND_RECV
 */
CUresult cuTensorMapReplaceAddress(CUtensorMap *tensorMap, void *globalAddress);
/**
 * @param canAccessPeer SEND_RECV
 * @param dev SEND_ONLY
 * @param peerDev SEND_ONLY
 */
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
/**
 * @param peerContext SEND_ONLY
 * @param Flags SEND_ONLY
 */
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
/**
 * @param peerContext SEND_ONLY
 */
CUresult cuCtxDisablePeerAccess(CUcontext peerContext);
/**
 * @param value SEND_RECV
 * @param attrib SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param dstDevice SEND_ONLY
 */
CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
/**
 * @param resource SEND_ONLY
 */
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource);
/**
 * @param pArray SEND_RECV
 * @param resource SEND_ONLY
 * @param arrayIndex SEND_ONLY
 * @param mipLevel SEND_ONLY
 */
CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
/**
 * @param pMipmappedArray SEND_RECV
 * @param resource SEND_ONLY
 */
CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource);
/**
 * @param pDevPtr SEND_RECV
 * @param pSize SEND_RECV
 * @param resource SEND_ONLY
 */
CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource);
/**
 * @param resource SEND_ONLY
 * @param flags SEND_ONLY
 */
CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags);
/**
 * @param count SEND_ONLY
 * @param resources SEND_RECV
 * @param hStream SEND_ONLY
 */
CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
/**
 * @param count SEND_ONLY
 * @param resources SEND_RECV
 * @param hStream SEND_ONLY
 */
CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
/**
 * @disabled
 * @param symbol SEND_RECV
 * @param pfn SEND_RECV
 * @param cudaVersion SEND_ONLY
 * @param flags SEND_ONLY
 * @param symbolStatus SEND_RECV
 */
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
/**
 * @disabled
 * @param ppExportTable SEND_RECV
 * @param pExportTableId SEND_RECV
 */
CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);
/**
 */
cudaError_t cudaDeviceReset();
/**
 */
cudaError_t cudaDeviceSynchronize();
/**
 * @param limit SEND_ONLY
 * @param value SEND_ONLY
 */
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
/**
 * @param pValue SEND_RECV
 * @param limit SEND_ONLY
 */
cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
/**
 * @param maxWidthInElements SEND_RECV
 * @param fmtDesc SEND_RECV
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device);
/**
 * @param pCacheConfig SEND_RECV
 */
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
/**
 * @param leastPriority SEND_RECV
 * @param greatestPriority SEND_RECV
 */
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
/**
 * @param cacheConfig SEND_ONLY
 */
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
/**
 * @param pConfig SEND_RECV
 */
cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
/**
 * @param config SEND_ONLY
 */
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
/**
 * @param device SEND_RECV
 * @param pciBusId SEND_RECV
 */
cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
/**
 * @param pciBusId SEND_RECV
 * @param len SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
/**
 * @param handle SEND_RECV
 * @param event SEND_ONLY
 */
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);
/**
 * @param event SEND_RECV
 * @param handle SEND_ONLY
 */
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);
/**
 * @param handle SEND_RECV
 * @param devPtr SEND_RECV
 */
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
/**
 * @param devPtr SEND_RECV
 * @param handle SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
/**
 * @param devPtr SEND_RECV
 */
cudaError_t cudaIpcCloseMemHandle(void *devPtr);
/**
 * @param target SEND_ONLY
 * @param scope SEND_ONLY
 */
cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope);
/**
 */
cudaError_t cudaThreadExit();
/**
 */
cudaError_t cudaThreadSynchronize();
/**
 * @param limit SEND_ONLY
 * @param value SEND_ONLY
 */
cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);
/**
 * @param pValue SEND_RECV
 * @param limit SEND_ONLY
 */
cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit);
/**
 * @param pCacheConfig SEND_RECV
 */
cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig);
/**
 * @param cacheConfig SEND_ONLY
 */
cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);
/**
 */
cudaError_t cudaGetLastError();
/**
 */
cudaError_t cudaPeekAtLastError();
/**
 * @disabled
 * @param error SEND_ONLY
 */
const char *cudaGetErrorName(cudaError_t error);
/**
 * @disabled
 * @param error SEND_ONLY
 */
const char *cudaGetErrorString(cudaError_t error);
/**
 * @param count RECV_ONLY
 */
cudaError_t cudaGetDeviceCount(int *count);
/**
 * @param prop RECV_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device);
/**
 * @param value SEND_RECV
 * @param attr SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
/**
 * @param memPool SEND_RECV
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device);
/**
 * @param device SEND_ONLY
 * @param memPool SEND_ONLY
 */
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
/**
 * @param memPool SEND_RECV
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device);
/**
 * @param nvSciSyncAttrList SEND_RECV
 * @param device SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device, int flags);
/**
 * @param value SEND_RECV
 * @param attr SEND_ONLY
 * @param srcDevice SEND_ONLY
 * @param dstDevice SEND_ONLY
 */
cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
/**
 * @param device SEND_RECV
 * @param prop SEND_RECV
 */
cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
/**
 * @param device SEND_ONLY
 * @param deviceFlags SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags);
/**
 * @param device SEND_ONLY
 */
cudaError_t cudaSetDevice(int device);
/**
 * @param device SEND_RECV
 */
cudaError_t cudaGetDevice(int *device);
/**
 * @param device_arr SEND_RECV
 * @param len SEND_ONLY
 */
cudaError_t cudaSetValidDevices(int *device_arr, int len);
/**
 * @param flags SEND_ONLY
 */
cudaError_t cudaSetDeviceFlags(unsigned int flags);
/**
 * @param flags SEND_RECV
 */
cudaError_t cudaGetDeviceFlags(unsigned int *flags);
/**
 * @param pStream SEND_RECV
 */
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
/**
 * @param pStream SEND_RECV
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
/**
 * @param pStream SEND_RECV
 * @param flags SEND_ONLY
 * @param priority SEND_ONLY
 */
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
/**
 * @param hStream SEND_ONLY
 * @param priority SEND_RECV
 */
cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority);
/**
 * @param hStream SEND_ONLY
 * @param flags SEND_RECV
 */
cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
/**
 * @param hStream SEND_ONLY
 * @param streamId SEND_RECV
 */
cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long *streamId);
/**
 */
cudaError_t cudaCtxResetPersistingL2Cache();
/**
 * @param dst SEND_ONLY
 * @param src SEND_ONLY
 */
cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src);
/**
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out SEND_RECV
 */
cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue *value_out);
/**
 * @param hStream SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue *value);
/**
 * @param stream SEND_ONLY
 */
cudaError_t cudaStreamDestroy(cudaStream_t stream);
/**
 * @param stream SEND_ONLY
 * @param event SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param callback SEND_ONLY
 * @param userData SEND_RECV
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags);
/**
 * @param stream SEND_ONLY
 */
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
/**
 * @param stream SEND_ONLY
 */
cudaError_t cudaStreamQuery(cudaStream_t stream);
/**
 * @param stream SEND_ONLY
 * @param devPtr SEND_RECV
 * @param length SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length, unsigned int flags);
/**
 * @param stream SEND_ONLY
 * @param mode SEND_ONLY
 */
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
/**
 * @param mode SEND_RECV
 */
cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode);
/**
 * @param stream SEND_ONLY
 * @param pGraph SEND_RECV
 */
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
/**
 * @param stream SEND_ONLY
 * @param pCaptureStatus SEND_RECV
 */
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);
/**
 * @param stream SEND_ONLY
 * @param captureStatus_out RECV_ONLY
 * @param id_out RECV_ONLY
 * @param graph_out RECV_ONLY
 * @param numDependencies_out RECV_ONLY
 * @param dependencies_out RECV_ONLY LENGTH:numDependencies_out
 */
cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out, cudaGraph_t *graph_out, const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out);
/**
 * @param stream SEND_ONLY
 * @param numDependencies SEND_ONLY
 * @param dependencies SEND_ONLY LENGTH:numDependencies
 * @param flags SEND_ONLY
 */
cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags);
/**
 * @param event RECV_ONLY
 */
cudaError_t cudaEventCreate(cudaEvent_t *event);
/**
 * @param event RECV_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
/**
 * @param event SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
/**
 * @param event SEND_ONLY
 * @param stream SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
/**
 * @param event SEND_ONLY
 */
cudaError_t cudaEventQuery(cudaEvent_t event);
/**
 * @param event SEND_ONLY
 */
cudaError_t cudaEventSynchronize(cudaEvent_t event);
/**
 * @param event SEND_ONLY
 */
cudaError_t cudaEventDestroy(cudaEvent_t event);
/**
 * @param ms RECV_ONLY
 * @param start SEND_ONLY
 * @param end SEND_ONLY
 */
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
/**
 * @disabled
 * @param extMem_out RECV_ONLY
 * @param memHandleDesc SEND_ONLY
 */
cudaError_t cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc);
/**
 * @param devPtr SEND_RECV
 * @param extMem SEND_ONLY
 * @param bufferDesc SEND_RECV
 */
cudaError_t cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc);
/**
 * @param mipmap SEND_RECV
 * @param extMem SEND_ONLY
 * @param mipmapDesc SEND_RECV
 */
cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc);
/**
 * @param extMem SEND_ONLY
 */
cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem);
/**
 * @param extSem_out SEND_RECV
 * @param semHandleDesc SEND_RECV
 */
cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc);
/**
 * @param extSemArray SEND_RECV
 * @param paramsArray SEND_RECV
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaSignalExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream);
/**
 * @param extSemArray SEND_RECV
 * @param paramsArray SEND_RECV
 * @param numExtSems SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaWaitExternalSemaphoresAsync_v2(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream);
/**
 * @param extSem SEND_ONLY
 */
cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);
/**
 * @disabled
 * @param func SEND_ONLY
 * @param gridDim SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param args SEND_ONLY
 * @param sharedMem SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
/**
 * @param config SEND_RECV
 * @param func SEND_RECV
 * @param args SEND_RECV
 */
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args);
/**
 * @param func SEND_RECV
 * @param gridDim SEND_ONLY
 * @param blockDim SEND_ONLY
 * @param args SEND_RECV
 * @param sharedMem SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
/**
 * @param launchParamsList SEND_RECV
 * @param numDevices SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags);
/**
 * @param func SEND_RECV
 * @param cacheConfig SEND_ONLY
 */
cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);
/**
 * @param func SEND_RECV
 * @param config SEND_ONLY
 */
cudaError_t cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config);
/**
 * @param attr SEND_RECV
 * @param func SEND_RECV
 */
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
/**
 * @param func SEND_RECV
 * @param attr SEND_ONLY
 * @param value SEND_ONLY
 */
cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
/**
 * @param d SEND_RECV
 */
cudaError_t cudaSetDoubleForDevice(double *d);
/**
 * @param d SEND_RECV
 */
cudaError_t cudaSetDoubleForHost(double *d);
/**
 * @param stream SEND_ONLY
 * @param fn SEND_ONLY
 * @param userData SEND_RECV
 */
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
/**
 * @param numBlocks SEND_RECV
 * @param func SEND_RECV
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
/**
 * @param dynamicSmemSize SEND_RECV
 * @param func SEND_RECV
 * @param numBlocks SEND_ONLY
 * @param blockSize SEND_ONLY
 */
cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize);
/**
 * @param numBlocks SEND_RECV
 * @param func SEND_RECV
 * @param blockSize SEND_ONLY
 * @param dynamicSMemSize SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
/**
 * @param clusterSize SEND_RECV
 * @param func SEND_RECV
 * @param launchConfig SEND_RECV
 */
cudaError_t cudaOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func, const cudaLaunchConfig_t *launchConfig);
/**
 * @param numClusters SEND_RECV
 * @param func SEND_RECV
 * @param launchConfig SEND_RECV
 */
cudaError_t cudaOccupancyMaxActiveClusters(int *numClusters, const void *func, const cudaLaunchConfig_t *launchConfig);
/**
 * @param devPtr SEND_RECV
 * @param size SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
/**
 * @param devPtr RECV_ONLY
 * @param size SEND_ONLY
 */
cudaError_t cudaMalloc(void **devPtr, size_t size);
/**
 * @disabled
 * @param ptr SEND_RECV
 * @param size SEND_ONLY
 */
cudaError_t cudaMallocHost(void **ptr, size_t size);
/**
 * @param devPtr SEND_RECV
 * @param pitch SEND_RECV
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
/**
 * @param array SEND_RECV
 * @param desc SEND_RECV
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags);
/**
 * @param devPtr SEND_ONLY
 */
cudaError_t cudaFree(void *devPtr);
/**
 * @disabled
 * @param ptr SEND_ONLY
 */
cudaError_t cudaFreeHost(void *ptr);
/**
 * @param array SEND_ONLY
 */
cudaError_t cudaFreeArray(cudaArray_t array);
/**
 * @param mipmappedArray SEND_ONLY
 */
cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
/**
 * @param pHost SEND_RECV
 * @param size SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
/**
 * @disabled
 * @param ptr SEND_RECV
 * @param size SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
/**
 * @disabled
 * @param ptr SEND_RECV
 */
cudaError_t cudaHostUnregister(void *ptr);
/**
 * @param pDevice SEND_RECV
 * @param pHost SEND_RECV
 * @param flags SEND_ONLY
 */
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
/**
 * @param pFlags SEND_RECV
 * @param pHost SEND_RECV
 */
cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
/**
 * @param pitchedDevPtr SEND_RECV
 * @param extent SEND_ONLY
 */
cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent);
/**
 * @param array SEND_RECV
 * @param desc SEND_RECV
 * @param extent SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags);
/**
 * @param mipmappedArray SEND_RECV
 * @param desc SEND_RECV
 * @param extent SEND_ONLY
 * @param numLevels SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags);
/**
 * @param levelArray SEND_RECV
 * @param mipmappedArray SEND_ONLY
 * @param level SEND_ONLY
 */
cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
/**
 * @param p SEND_RECV
 */
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
/**
 * @param p SEND_RECV
 */
cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p);
/**
 * @param p SEND_RECV
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
/**
 * @param p SEND_RECV
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p, cudaStream_t stream);
/**
 * @param free SEND_RECV
 * @param total SEND_RECV
 */
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
/**
 * @param desc SEND_RECV
 * @param extent SEND_RECV
 * @param flags SEND_RECV
 * @param array SEND_ONLY
 */
cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array);
/**
 * @param pPlaneArray SEND_RECV
 * @param hArray SEND_ONLY
 * @param planeIdx SEND_ONLY
 */
cudaError_t cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray, unsigned int planeIdx);
/**
 * @param memoryRequirements SEND_RECV
 * @param array SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements *memoryRequirements, cudaArray_t array, int device);
/**
 * @param memoryRequirements SEND_RECV
 * @param mipmap SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaMipmappedArrayGetMemoryRequirements(struct cudaArrayMemoryRequirements *memoryRequirements, cudaMipmappedArray_t mipmap, int device);
/**
 * @param sparseProperties SEND_RECV
 * @param array SEND_ONLY
 */
cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array);
/**
 * @param sparseProperties SEND_RECV
 * @param mipmap SEND_ONLY
 */
cudaError_t cudaMipmappedArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap);
/**
 * @disabled
 * @param dst SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_RECV
 * @param dstDevice SEND_ONLY
 * @param src SEND_RECV
 * @param srcDevice SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
/**
 * @param dst SEND_RECV
 * @param dpitch SEND_ONLY
 * @param src SEND_RECV
 * @param spitch SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param src SEND_RECV
 * @param spitch SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_RECV
 * @param dpitch SEND_ONLY
 * @param src SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_ONLY
 * @param wOffsetDst SEND_ONLY
 * @param hOffsetDst SEND_ONLY
 * @param src SEND_ONLY
 * @param wOffsetSrc SEND_ONLY
 * @param hOffsetSrc SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind);
/**
 * @param symbol SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_RECV
 * @param symbol SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @disabled - manually implemented
 * @param dst SEND_ONLY
 * @param count SEND_ONLY
 * @param src SEND_ONLY SIZE:count
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param dst SEND_RECV
 * @param dstDevice SEND_ONLY
 * @param src SEND_RECV
 * @param srcDevice SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream);
/**
 * @param dst SEND_RECV
 * @param dpitch SEND_ONLY
 * @param src SEND_RECV
 * @param spitch SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param dst SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param src SEND_RECV
 * @param spitch SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param dst SEND_RECV
 * @param dpitch SEND_ONLY
 * @param src SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param symbol SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param dst SEND_RECV
 * @param symbol SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param devPtr SEND_RECV
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
/**
 * @param devPtr SEND_RECV
 * @param pitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 */
cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
/**
 * @param pitchedDevPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param extent SEND_ONLY
 */
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
/**
 * @param devPtr SEND_RECV
 * @param value SEND_ONLY
 * @param count SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
/**
 * @param devPtr SEND_RECV
 * @param pitch SEND_ONLY
 * @param value SEND_ONLY
 * @param width SEND_ONLY
 * @param height SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
/**
 * @param pitchedDevPtr SEND_ONLY
 * @param value SEND_ONLY
 * @param extent SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
/**
 * @param devPtr SEND_RECV
 * @param symbol SEND_RECV
 */
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
/**
 * @param size SEND_RECV
 * @param symbol SEND_RECV
 */
cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol);
/**
 * @param devPtr SEND_RECV
 * @param count SEND_ONLY
 * @param dstDevice SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
/**
 * @param devPtr SEND_RECV
 * @param count SEND_ONLY
 * @param advice SEND_ONLY
 * @param device SEND_ONLY
 */
cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
/**
 * @param data SEND_RECV
 * @param dataSize SEND_ONLY
 * @param attribute SEND_ONLY
 * @param devPtr SEND_RECV
 * @param count SEND_ONLY
 */
cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count);
/**
 * @param data SEND_RECV
 * @param dataSizes SEND_RECV
 * @param attributes SEND_RECV
 * @param numAttributes SEND_ONLY
 * @param devPtr SEND_RECV
 * @param count SEND_ONLY
 */
cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);
/**
 * @param dst SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_RECV
 * @param src SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_ONLY
 * @param wOffsetDst SEND_ONLY
 * @param hOffsetDst SEND_ONLY
 * @param src SEND_ONLY
 * @param wOffsetSrc SEND_ONLY
 * @param hOffsetSrc SEND_ONLY
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind);
/**
 * @param dst SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param dst SEND_RECV
 * @param src SEND_ONLY
 * @param wOffset SEND_ONLY
 * @param hOffset SEND_ONLY
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
/**
 * @param devPtr SEND_RECV
 * @param size SEND_ONLY
 * @param hStream SEND_ONLY
 */
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);
/**
 * @param devPtr SEND_RECV
 * @param hStream SEND_ONLY
 */
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream);
/**
 * @param memPool SEND_ONLY
 * @param minBytesToKeep SEND_ONLY
 */
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
/**
 * @param memPool SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value);
/**
 * @param memPool SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value);
/**
 * @param memPool SEND_ONLY
 * @param descList SEND_RECV
 * @param count SEND_ONLY
 */
cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count);
/**
 * @param flags SEND_RECV
 * @param memPool SEND_ONLY
 * @param location SEND_RECV
 */
cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location);
/**
 * @param memPool SEND_RECV
 * @param poolProps SEND_RECV
 */
cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps);
/**
 * @param memPool SEND_ONLY
 */
cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool);
/**
 * @param ptr SEND_RECV
 * @param size SEND_ONLY
 * @param memPool SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
/**
 * @param shareableHandle SEND_RECV
 * @param memPool SEND_ONLY
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMemPoolExportToShareableHandle(void *shareableHandle, cudaMemPool_t memPool, enum cudaMemAllocationHandleType handleType, unsigned int flags);
/**
 * @param memPool SEND_RECV
 * @param shareableHandle SEND_RECV
 * @param handleType SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t *memPool, void *shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags);
/**
 * @param exportData SEND_RECV
 * @param ptr SEND_RECV
 */
cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr);
/**
 * @param ptr SEND_RECV
 * @param memPool SEND_ONLY
 * @param exportData SEND_RECV
 */
cudaError_t cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData);
/**
 * @param attributes SEND_RECV
 * @param ptr SEND_RECV
 */
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);
/**
 * @param canAccessPeer SEND_RECV
 * @param device SEND_ONLY
 * @param peerDevice SEND_ONLY
 */
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);
/**
 * @param peerDevice SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
/**
 * @param peerDevice SEND_ONLY
 */
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
/**
 * @param resource SEND_ONLY
 */
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
/**
 * @param resource SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags);
/**
 * @param count SEND_ONLY
 * @param resources SEND_RECV
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream);
/**
 * @param count SEND_ONLY
 * @param resources SEND_RECV
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream);
/**
 * @param devPtr SEND_RECV
 * @param size SEND_RECV
 * @param resource SEND_ONLY
 */
cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);
/**
 * @param array SEND_RECV
 * @param resource SEND_ONLY
 * @param arrayIndex SEND_ONLY
 * @param mipLevel SEND_ONLY
 */
cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
/**
 * @param mipmappedArray SEND_RECV
 * @param resource SEND_ONLY
 */
cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource);
/**
 * @param desc SEND_RECV
 * @param array SEND_ONLY
 */
cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array);
/**
 * @disabled
 * @param x SEND_ONLY
 * @param y SEND_ONLY
 * @param z SEND_ONLY
 * @param w SEND_ONLY
 * @param f SEND_ONLY
 */
struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
/**
 * @param pTexObject SEND_RECV
 * @param pResDesc SEND_RECV
 * @param pTexDesc SEND_RECV
 * @param pResViewDesc SEND_RECV
 */
cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc);
/**
 * @param texObject SEND_ONLY
 */
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);
/**
 * @param pResDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject);
/**
 * @param pTexDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject);
/**
 * @param pResViewDesc SEND_RECV
 * @param texObject SEND_ONLY
 */
cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject);
/**
 * @param pSurfObject SEND_RECV
 * @param pResDesc SEND_RECV
 */
cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc);
/**
 * @param surfObject SEND_ONLY
 */
cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
/**
 * @param pResDesc SEND_RECV
 * @param surfObject SEND_ONLY
 */
cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject);
/**
 * @param driverVersion SEND_RECV
 */
cudaError_t cudaDriverGetVersion(int *driverVersion);
/**
 * @param runtimeVersion SEND_RECV
 */
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
/**
 * @param pGraph SEND_RECV
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);
/**
 * @param hSrc SEND_ONLY
 * @param hDst SEND_ONLY
 */
cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value_out SEND_RECV
 */
cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue *value_out);
/**
 * @param hNode SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue *value);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param pCopyParams SEND_RECV
 */
cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param symbol SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param dst SEND_RECV
 * @param symbol SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param dst SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param symbol SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param node SEND_ONLY
 * @param dst SEND_RECV
 * @param symbol SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param node SEND_ONLY
 * @param dst SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param pMemsetParams SEND_RECV
 */
cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams);
/**
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param childGraph SEND_ONLY
 */
cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph);
/**
 * @param node SEND_ONLY
 * @param pGraph SEND_RECV
 */
cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
/**
 * @param node SEND_ONLY
 * @param event_out SEND_RECV
 */
cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
/**
 * @param node SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
/**
 * @param node SEND_ONLY
 * @param event_out SEND_RECV
 */
cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
/**
 * @param node SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out SEND_RECV
 */
cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
/**
 * @param hNode SEND_ONLY
 * @param params_out SEND_RECV
 */
cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out);
/**
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams *nodeParams);
/**
 * @param node SEND_ONLY
 * @param params_out SEND_RECV
 */
cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams *params_out);
/**
 * @param pGraphNode SEND_RECV
 * @param graph SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param numDependencies SEND_ONLY
 * @param dptr SEND_RECV
 */
cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr);
/**
 * @param node SEND_ONLY
 * @param dptr_out SEND_RECV
 */
cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out);
/**
 * @param device SEND_ONLY
 */
cudaError_t cudaDeviceGraphMemTrim(int device);
/**
 * @param device SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void *value);
/**
 * @param device SEND_ONLY
 * @param attr SEND_ONLY
 * @param value SEND_RECV
 */
cudaError_t cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void *value);
/**
 * @param pGraphClone SEND_RECV
 * @param originalGraph SEND_ONLY
 */
cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);
/**
 * @param pNode SEND_RECV
 * @param originalNode SEND_ONLY
 * @param clonedGraph SEND_ONLY
 */
cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);
/**
 * @param node SEND_ONLY
 * @param pType SEND_RECV
 */
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType);
/**
 * @param graph SEND_ONLY
 * @param nodes SEND_RECV
 * @param numNodes SEND_RECV
 */
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);
/**
 * @param graph SEND_ONLY
 * @param pRootNodes SEND_RECV
 * @param pNumRootNodes SEND_RECV
 */
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes);
/**
 * @param graph SEND_ONLY
 * @param from SEND_RECV
 * @param to SEND_RECV
 * @param numEdges SEND_RECV
 */
cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges);
/**
 * @param node SEND_ONLY
 * @param pDependencies SEND_RECV
 * @param pNumDependencies SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies);
/**
 * @param node SEND_ONLY
 * @param pDependentNodes SEND_RECV
 * @param pNumDependentNodes SEND_RECV
 */
cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes);
/**
 * @param graph SEND_ONLY
 * @param from SEND_RECV
 * @param to SEND_RECV
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
/**
 * @param graph SEND_ONLY
 * @param from SEND_RECV
 * @param to SEND_RECV
 * @param numDependencies SEND_ONLY
 */
cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
/**
 * @param node SEND_ONLY
 */
cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node);
/**
 * @param pGraphExec SEND_RECV
 * @param graph SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags);
/**
 * @param pGraphExec SEND_RECV
 * @param graph SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags);
/**
 * @param pGraphExec SEND_RECV
 * @param graph SEND_ONLY
 * @param instantiateParams SEND_RECV
 */
cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams *instantiateParams);
/**
 * @param graphExec SEND_ONLY
 * @param flags SEND_RECV
 */
cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long *flags);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param symbol SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param dst SEND_RECV
 * @param symbol SEND_RECV
 * @param count SEND_ONLY
 * @param offset SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param dst SEND_RECV
 * @param src SEND_RECV
 * @param count SEND_ONLY
 * @param kind SEND_ONLY
 */
cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param pNodeParams SEND_RECV
 */
cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param node SEND_ONLY
 * @param childGraph SEND_ONLY
 */
cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param event SEND_ONLY
 */
cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param nodeParams SEND_RECV
 */
cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_ONLY
 */
cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hNode SEND_ONLY
 * @param isEnabled SEND_RECV
 */
cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int *isEnabled);
/**
 * @param hGraphExec SEND_ONLY
 * @param hGraph SEND_ONLY
 * @param resultInfo SEND_RECV
 */
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo *resultInfo);
/**
 * @param graphExec SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
/**
 * @param graphExec SEND_ONLY
 * @param stream SEND_ONLY
 */
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
/**
 * @param graphExec SEND_ONLY
 */
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
/**
 * @param graph SEND_ONLY
 */
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
/**
 * @param graph SEND_ONLY
 * @param path SEND_RECV
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags);
/**
 * @param object_out SEND_RECV
 * @param ptr SEND_RECV
 * @param destroy SEND_ONLY
 * @param initialRefcount SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count);
/**
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count);
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 * @param flags SEND_ONLY
 */
cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags);
/**
 * @param graph SEND_ONLY
 * @param object SEND_ONLY
 * @param count SEND_ONLY
 */
cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count);
/**
 * @param symbol SEND_RECV
 * @param funcPtr SEND_RECV
 * @param flags SEND_ONLY
 * @param driverStatus SEND_RECV
 */
cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus);
/**
 * @param ppExportTable SEND_RECV
 * @param pExportTableId SEND_RECV
 */
cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId);
/**
 * @param functionPtr SEND_RECV
 * @param symbolPtr SEND_RECV
 */
cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr, const void *symbolPtr);
/**
 * @param handle RECV_ONLY
 */
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
/**
 * @param handle SEND_ONLY
 */
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
/**
 * @param handle SEND_ONLY
 * @param transa SEND_ONLY
 * @param transb SEND_ONLY
 * @param m SEND_ONLY
 * @param n SEND_ONLY
 * @param k SEND_ONLY
 * @param alpha SEND_ONLY NULLABLE
 * @param A SEND_ONLY
 * @param lda SEND_ONLY
 * @param B SEND_ONLY
 * @param ldb SEND_ONLY
 * @param beta SEND_ONLY NULLABLE
 * @param C SEND_ONLY
 * @param ldc SEND_ONLY
 */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n, int k,
                              const float *alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              const float *beta,
                              float *C, int ldc);
