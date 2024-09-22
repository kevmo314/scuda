#include <nvml.h>

#include <unistd.h>

#include "gen_api.h"

int handle_nvmlInit_v2(int connfd)
{
    nvmlReturn_t result = nvmlInit_v2();

    return result;
}

int handle_nvmlInitWithFlags(int connfd)
{
    unsigned int flags;

    if (read(connfd, &flags, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlInitWithFlags(flags);

    return result;
}

int handle_nvmlShutdown(int connfd)
{
    nvmlReturn_t result = nvmlShutdown();

    return result;
}

int handle_nvmlSystemGetDriverVersion(int connfd)
{
    unsigned int length;

    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *version = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlSystemGetDriverVersion(version, length);

    if (write(connfd, version, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetNVMLVersion(int connfd)
{
    unsigned int length;

    if (read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *version = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlSystemGetNVMLVersion(version, length);

    if (write(connfd, version, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetCudaDriverVersion(int connfd)
{
    int cudaDriverVersion;

    if (read(connfd, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetCudaDriverVersion(&cudaDriverVersion);

    if (write(connfd, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetCudaDriverVersion_v2(int connfd)
{
    int cudaDriverVersion;

    if (read(connfd, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetCudaDriverVersion_v2(&cudaDriverVersion);

    if (write(connfd, &cudaDriverVersion, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetProcessName(int connfd)
{
    unsigned int pid;
    unsigned int length;

    if (read(connfd, &pid, sizeof(unsigned int)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *name = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlSystemGetProcessName(pid, name, length);

    if (write(connfd, name, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetCount(int connfd)
{
    unsigned int unitCount;

    if (read(connfd, &unitCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetCount(&unitCount);

    if (write(connfd, &unitCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetHandleByIndex(int connfd)
{
    unsigned int index;
    nvmlUnit_t unit;

    if (read(connfd, &index, sizeof(unsigned int)) < 0 ||
        read(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetHandleByIndex(index, &unit);

    if (write(connfd, &unit, sizeof(nvmlUnit_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetUnitInfo(int connfd)
{
    nvmlUnit_t unit;
    nvmlUnitInfo_t info;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &info, sizeof(nvmlUnitInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetUnitInfo(unit, &info);

    if (write(connfd, &info, sizeof(nvmlUnitInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetLedState(int connfd)
{
    nvmlUnit_t unit;
    nvmlLedState_t state;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &state, sizeof(nvmlLedState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetLedState(unit, &state);

    if (write(connfd, &state, sizeof(nvmlLedState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetPsuInfo(int connfd)
{
    nvmlUnit_t unit;
    nvmlPSUInfo_t psu;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &psu, sizeof(nvmlPSUInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetPsuInfo(unit, &psu);

    if (write(connfd, &psu, sizeof(nvmlPSUInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetTemperature(int connfd)
{
    nvmlUnit_t unit;
    unsigned int type;
    unsigned int temp;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &type, sizeof(unsigned int)) < 0 ||
        read(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetTemperature(unit, type, &temp);

    if (write(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetFanSpeedInfo(int connfd)
{
    nvmlUnit_t unit;
    nvmlUnitFanSpeeds_t fanSpeeds;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds);

    if (write(connfd, &fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitGetDevices(int connfd)
{
    nvmlUnit_t unit;
    unsigned int deviceCount;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlDevice_t *devices = (nvmlDevice_t *) malloc(deviceCount * sizeof(nvmlDevice_t));

    nvmlReturn_t result = nvmlUnitGetDevices(unit, &deviceCount, devices);

    if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0 ||
        write(connfd, devices, deviceCount * sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetHicVersion(int connfd)
{
    unsigned int hwbcCount;

    if (read(connfd, &hwbcCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlHwbcEntry_t *hwbcEntries = (nvmlHwbcEntry_t *) malloc(hwbcCount * sizeof(nvmlHwbcEntry_t));

    nvmlReturn_t result = nvmlSystemGetHicVersion(&hwbcCount, hwbcEntries);

    if (write(connfd, &hwbcCount, sizeof(unsigned int)) < 0 ||
        write(connfd, hwbcEntries, hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCount_v2(int connfd)
{
    unsigned int deviceCount;

    if (read(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCount_v2(&deviceCount);

    if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAttributes_v2(int connfd)
{
    nvmlDevice_t device;
    nvmlDeviceAttributes_t attributes;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAttributes_v2(device, &attributes);

    if (write(connfd, &attributes, sizeof(nvmlDeviceAttributes_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByIndex_v2(int connfd)
{
    unsigned int index;
    nvmlDevice_t device;

    if (read(connfd, &index, sizeof(unsigned int)) < 0 ||
        read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(index, &device);

    if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleBySerial(int connfd)
{
    char serial;
    nvmlDevice_t device;

    if (read(connfd, &serial, sizeof(char)) < 0 ||
        read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleBySerial(&serial, &device);

    if (write(connfd, &serial, sizeof(char)) < 0 ||
        write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByUUID(int connfd)
{
    char uuid;
    nvmlDevice_t device;

    if (read(connfd, &uuid, sizeof(char)) < 0 ||
        read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByUUID(&uuid, &device);

    if (write(connfd, &uuid, sizeof(char)) < 0 ||
        write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(int connfd)
{
    char pciBusId;
    nvmlDevice_t device;

    if (read(connfd, &pciBusId, sizeof(char)) < 0 ||
        read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHandleByPciBusId_v2(&pciBusId, &device);

    if (write(connfd, &pciBusId, sizeof(char)) < 0 ||
        write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetName(int connfd)
{
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *name = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetName(device, name, length);

    if (write(connfd, name, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBrand(int connfd)
{
    nvmlDevice_t device;
    nvmlBrandType_t type;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlBrandType_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBrand(device, &type);

    if (write(connfd, &type, sizeof(nvmlBrandType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetIndex(int connfd)
{
    nvmlDevice_t device;
    unsigned int index;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &index, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetIndex(device, &index);

    if (write(connfd, &index, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSerial(int connfd)
{
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *serial = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetSerial(device, serial, length);

    if (write(connfd, serial, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryAffinity(int connfd)
{
    nvmlDevice_t device;
    unsigned int nodeSetSize;
    nvmlAffinityScope_t scope;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &nodeSetSize, sizeof(unsigned int)) < 0 ||
        read(connfd, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        return -1;

    unsigned long *nodeSet = (unsigned long *) malloc(nodeSetSize * sizeof(unsigned long));

    nvmlReturn_t result = nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope);

    if (write(connfd, nodeSet, nodeSetSize * sizeof(unsigned long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCpuAffinityWithinScope(int connfd)
{
    nvmlDevice_t device;
    unsigned int cpuSetSize;
    nvmlAffinityScope_t scope;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &cpuSetSize, sizeof(unsigned int)) < 0 ||
        read(connfd, &scope, sizeof(nvmlAffinityScope_t)) < 0)
        return -1;

    unsigned long *cpuSet = (unsigned long *) malloc(cpuSetSize * sizeof(unsigned long));

    nvmlReturn_t result = nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope);

    if (write(connfd, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCpuAffinity(int connfd)
{
    nvmlDevice_t device;
    unsigned int cpuSetSize;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &cpuSetSize, sizeof(unsigned int)) < 0)
        return -1;

    unsigned long *cpuSet = (unsigned long *) malloc(cpuSetSize * sizeof(unsigned long));

    nvmlReturn_t result = nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);

    if (write(connfd, cpuSet, cpuSetSize * sizeof(unsigned long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetCpuAffinity(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetCpuAffinity(device);

    return result;
}

int handle_nvmlDeviceClearCpuAffinity(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearCpuAffinity(device);

    return result;
}

int handle_nvmlDeviceGetTopologyCommonAncestor(int connfd)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuTopologyLevel_t pathInfo;

    if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &device2, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &pathInfo);

    if (write(connfd, &pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTopologyNearestGpus(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuTopologyLevel_t level;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlDevice_t *deviceArray = (nvmlDevice_t *) malloc(count * sizeof(nvmlDevice_t));

    nvmlReturn_t result = nvmlDeviceGetTopologyNearestGpus(device, level, &count, deviceArray);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
        write(connfd, deviceArray, count * sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSystemGetTopologyGpuSet(int connfd)
{
    unsigned int cpuNumber;
    unsigned int count;
    nvmlDevice_t deviceArray;

    if (read(connfd, &cpuNumber, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0 ||
        read(connfd, &deviceArray, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlSystemGetTopologyGpuSet(cpuNumber, &count, &deviceArray);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
        write(connfd, &deviceArray, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetP2PStatus(int connfd)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    nvmlGpuP2PCapsIndex_t p2pIndex;
    nvmlGpuP2PStatus_t p2pStatus;

    if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &device2, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        read(connfd, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, &p2pStatus);

    if (write(connfd, &p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetUUID(int connfd)
{
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *uuid = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetUUID(device, uuid, length);

    if (write(connfd, uuid, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetMdevUUID(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    char *mdevUuid = (char *) malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size);

    if (write(connfd, mdevUuid, size * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMinorNumber(int connfd)
{
    nvmlDevice_t device;
    unsigned int minorNumber;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minorNumber, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinorNumber(device, &minorNumber);

    if (write(connfd, &minorNumber, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBoardPartNumber(int connfd)
{
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *partNumber = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetBoardPartNumber(device, partNumber, length);

    if (write(connfd, partNumber, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetInforomVersion(int connfd)
{
    nvmlDevice_t device;
    nvmlInforomObject_t object;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &object, sizeof(nvmlInforomObject_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *version = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetInforomVersion(device, object, version, length);

    if (write(connfd, version, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetInforomImageVersion(int connfd)
{
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *version = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetInforomImageVersion(device, version, length);

    if (write(connfd, version, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetInforomConfigurationChecksum(int connfd)
{
    nvmlDevice_t device;
    unsigned int checksum;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &checksum, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetInforomConfigurationChecksum(device, &checksum);

    if (write(connfd, &checksum, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceValidateInforom(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceValidateInforom(device);

    return result;
}

int handle_nvmlDeviceGetDisplayMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t display;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &display, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDisplayMode(device, &display);

    if (write(connfd, &display, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDisplayActive(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t isActive;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDisplayActive(device, &isActive);

    if (write(connfd, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPersistenceMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPersistenceMode(device, &mode);

    if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPciInfo_v3(int connfd)
{
    nvmlDevice_t device;
    nvmlPciInfo_t pci;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPciInfo_v3(device, &pci);

    if (write(connfd, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(int connfd)
{
    nvmlDevice_t device;
    unsigned int maxLinkGen;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &maxLinkGen, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkGeneration(device, &maxLinkGen);

    if (write(connfd, &maxLinkGen, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuMaxPcieLinkGeneration(int connfd)
{
    nvmlDevice_t device;
    unsigned int maxLinkGenDevice;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &maxLinkGenDevice, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &maxLinkGenDevice);

    if (write(connfd, &maxLinkGenDevice, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(int connfd)
{
    nvmlDevice_t device;
    unsigned int maxLinkWidth;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &maxLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxPcieLinkWidth(device, &maxLinkWidth);

    if (write(connfd, &maxLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(int connfd)
{
    nvmlDevice_t device;
    unsigned int currLinkGen;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &currLinkGen, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &currLinkGen);

    if (write(connfd, &currLinkGen, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(int connfd)
{
    nvmlDevice_t device;
    unsigned int currLinkWidth;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &currLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkWidth(device, &currLinkWidth);

    if (write(connfd, &currLinkWidth, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieThroughput(int connfd)
{
    nvmlDevice_t device;
    nvmlPcieUtilCounter_t counter;
    unsigned int value;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        read(connfd, &value, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieThroughput(device, counter, &value);

    if (write(connfd, &value, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieReplayCounter(int connfd)
{
    nvmlDevice_t device;
    unsigned int value;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &value, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieReplayCounter(device, &value);

    if (write(connfd, &value, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetClockInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int clock;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clock, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetClockInfo(device, type, &clock);

    if (write(connfd, &clock, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxClockInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t type;
    unsigned int clock;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clock, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxClockInfo(device, type, &clock);

    if (write(connfd, &clock, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetApplicationsClock(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetApplicationsClock(device, clockType, &clockMHz);

    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDefaultApplicationsClock(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDefaultApplicationsClock(device, clockType, &clockMHz);

    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetApplicationsClocks(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetApplicationsClocks(device);

    return result;
}

int handle_nvmlDeviceGetClock(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    nvmlClockId_t clockId;
    unsigned int clockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clockId, sizeof(nvmlClockId_t)) < 0 ||
        read(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetClock(device, clockType, clockId, &clockMHz);

    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxCustomerBoostClock(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t clockType;
    unsigned int clockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clockType, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxCustomerBoostClock(device, clockType, &clockMHz);

    if (write(connfd, &clockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedMemoryClocks(int connfd)
{
    nvmlDevice_t device;
    unsigned int count;
    unsigned int clocksMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0 ||
        read(connfd, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedMemoryClocks(device, &count, &clocksMHz);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
        write(connfd, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedGraphicsClocks(int connfd)
{
    nvmlDevice_t device;
    unsigned int memoryClockMHz;
    unsigned int count;
    unsigned int clocksMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &memoryClockMHz, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0 ||
        read(connfd, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, &count, &clocksMHz);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
        write(connfd, &clocksMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAutoBoostedClocksEnabled(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t isEnabled;
    nvmlEnableState_t defaultIsEnabled;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        read(connfd, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAutoBoostedClocksEnabled(device, &isEnabled, &defaultIsEnabled);

    if (write(connfd, &isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        write(connfd, &defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetAutoBoostedClocksEnabled(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t enabled;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &enabled, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled);

    return result;
}

int handle_nvmlDeviceSetDefaultAutoBoostedClocksEnabled(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t enabled;
    unsigned int flags;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &enabled, sizeof(nvmlEnableState_t)) < 0 ||
        read(connfd, &flags, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags);

    return result;
}

int handle_nvmlDeviceGetFanSpeed(int connfd)
{
    nvmlDevice_t device;
    unsigned int speed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFanSpeed(device, &speed);

    if (write(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFanSpeed_v2(int connfd)
{
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int speed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fan, sizeof(unsigned int)) < 0 ||
        read(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFanSpeed_v2(device, fan, &speed);

    if (write(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTargetFanSpeed(int connfd)
{
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int targetSpeed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fan, sizeof(unsigned int)) < 0 ||
        read(connfd, &targetSpeed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTargetFanSpeed(device, fan, &targetSpeed);

    if (write(connfd, &targetSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetDefaultFanSpeed_v2(int connfd)
{
    nvmlDevice_t device;
    unsigned int fan;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fan, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetDefaultFanSpeed_v2(device, fan);

    return result;
}

int handle_nvmlDeviceGetMinMaxFanSpeed(int connfd)
{
    nvmlDevice_t device;
    unsigned int minSpeed;
    unsigned int maxSpeed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minSpeed, sizeof(unsigned int)) < 0 ||
        read(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinMaxFanSpeed(device, &minSpeed, &maxSpeed);

    if (write(connfd, &minSpeed, sizeof(unsigned int)) < 0 ||
        write(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFanControlPolicy_v2(int connfd)
{
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fan, sizeof(unsigned int)) < 0 ||
        read(connfd, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFanControlPolicy_v2(device, fan, &policy);

    if (write(connfd, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetFanControlPolicy(int connfd)
{
    nvmlDevice_t device;
    unsigned int fan;
    nvmlFanControlPolicy_t policy;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fan, sizeof(unsigned int)) < 0 ||
        read(connfd, &policy, sizeof(nvmlFanControlPolicy_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetFanControlPolicy(device, fan, policy);

    return result;
}

int handle_nvmlDeviceGetNumFans(int connfd)
{
    nvmlDevice_t device;
    unsigned int numFans;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &numFans, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNumFans(device, &numFans);

    if (write(connfd, &numFans, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTemperature(int connfd)
{
    nvmlDevice_t device;
    nvmlTemperatureSensors_t sensorType;
    unsigned int temp;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        read(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTemperature(device, sensorType, &temp);

    if (write(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTemperatureThreshold(int connfd)
{
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    unsigned int temp;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        read(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTemperatureThreshold(device, thresholdType, &temp);

    if (write(connfd, &temp, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetTemperatureThreshold(int connfd)
{
    nvmlDevice_t device;
    nvmlTemperatureThresholds_t thresholdType;
    int temp;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        read(connfd, &temp, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetTemperatureThreshold(device, thresholdType, &temp);

    if (write(connfd, &temp, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetThermalSettings(int connfd)
{
    nvmlDevice_t device;
    unsigned int sensorIndex;
    nvmlGpuThermalSettings_t pThermalSettings;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &sensorIndex, sizeof(unsigned int)) < 0 ||
        read(connfd, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetThermalSettings(device, sensorIndex, &pThermalSettings);

    if (write(connfd, &pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPerformanceState(int connfd)
{
    nvmlDevice_t device;
    nvmlPstates_t pState;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPerformanceState(device, &pState);

    if (write(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCurrentClocksThrottleReasons(int connfd)
{
    nvmlDevice_t device;
    unsigned long long clocksThrottleReasons;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &clocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clocksThrottleReasons);

    if (write(connfd, &clocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedClocksThrottleReasons(int connfd)
{
    nvmlDevice_t device;
    unsigned long long supportedClocksThrottleReasons;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedClocksThrottleReasons(device, &supportedClocksThrottleReasons);

    if (write(connfd, &supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerState(int connfd)
{
    nvmlDevice_t device;
    nvmlPstates_t pState;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerState(device, &pState);

    if (write(connfd, &pState, sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementMode(device, &mode);

    if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementLimit(int connfd)
{
    nvmlDevice_t device;
    unsigned int limit;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimit(device, &limit);

    if (write(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementLimitConstraints(int connfd)
{
    nvmlDevice_t device;
    unsigned int minLimit;
    unsigned int maxLimit;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minLimit, sizeof(unsigned int)) < 0 ||
        read(connfd, &maxLimit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementLimitConstraints(device, &minLimit, &maxLimit);

    if (write(connfd, &minLimit, sizeof(unsigned int)) < 0 ||
        write(connfd, &maxLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerManagementDefaultLimit(int connfd)
{
    nvmlDevice_t device;
    unsigned int defaultLimit;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &defaultLimit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);

    if (write(connfd, &defaultLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerUsage(int connfd)
{
    nvmlDevice_t device;
    unsigned int power;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &power, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);

    if (write(connfd, &power, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTotalEnergyConsumption(int connfd)
{
    nvmlDevice_t device;
    unsigned long long energy;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &energy, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);

    if (write(connfd, &energy, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEnforcedPowerLimit(int connfd)
{
    nvmlDevice_t device;
    unsigned int limit;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEnforcedPowerLimit(device, &limit);

    if (write(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuOperationMode(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuOperationMode_t current;
    nvmlGpuOperationMode_t pending;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        read(connfd, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuOperationMode(device, &current, &pending);

    if (write(connfd, &current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        write(connfd, &pending, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlMemory_t memory;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &memory, sizeof(nvmlMemory_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);

    if (write(connfd, &memory, sizeof(nvmlMemory_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryInfo_v2(int connfd)
{
    nvmlDevice_t device;
    nvmlMemory_v2_t memory;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &memory, sizeof(nvmlMemory_v2_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryInfo_v2(device, &memory);

    if (write(connfd, &memory, sizeof(nvmlMemory_v2_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetComputeMode(int connfd)
{
    nvmlDevice_t device;
    nvmlComputeMode_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetComputeMode(device, &mode);

    if (write(connfd, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCudaComputeCapability(int connfd)
{
    nvmlDevice_t device;
    int major;
    int minor;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &major, sizeof(int)) < 0 ||
        read(connfd, &minor, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);

    if (write(connfd, &major, sizeof(int)) < 0 ||
        write(connfd, &minor, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEccMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t current;
    nvmlEnableState_t pending;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &current, sizeof(nvmlEnableState_t)) < 0 ||
        read(connfd, &pending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEccMode(device, &current, &pending);

    if (write(connfd, &current, sizeof(nvmlEnableState_t)) < 0 ||
        write(connfd, &pending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDefaultEccMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t defaultMode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &defaultMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDefaultEccMode(device, &defaultMode);

    if (write(connfd, &defaultMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBoardId(int connfd)
{
    nvmlDevice_t device;
    unsigned int boardId;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &boardId, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBoardId(device, &boardId);

    if (write(connfd, &boardId, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMultiGpuBoard(int connfd)
{
    nvmlDevice_t device;
    unsigned int multiGpuBool;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &multiGpuBool, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMultiGpuBoard(device, &multiGpuBool);

    if (write(connfd, &multiGpuBool, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetTotalEccErrors(int connfd)
{
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    unsigned long long eccCounts;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        read(connfd, &eccCounts, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetTotalEccErrors(device, errorType, counterType, &eccCounts);

    if (write(connfd, &eccCounts, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDetailedEccErrors(int connfd)
{
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlEccErrorCounts_t eccCounts;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        read(connfd, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, &eccCounts);

    if (write(connfd, &eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryErrorCounter(int connfd)
{
    nvmlDevice_t device;
    nvmlMemoryErrorType_t errorType;
    nvmlEccCounterType_t counterType;
    nvmlMemoryLocation_t locationType;
    unsigned long long count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        read(connfd, &locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType, locationType, &count);

    if (write(connfd, &count, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetUtilizationRates(int connfd)
{
    nvmlDevice_t device;
    nvmlUtilization_t utilization;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &utilization, sizeof(nvmlUtilization_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);

    if (write(connfd, &utilization, sizeof(nvmlUtilization_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderUtilization(int connfd)
{
    nvmlDevice_t device;
    unsigned int utilization;
    unsigned int samplingPeriodUs;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &utilization, sizeof(unsigned int)) < 0 ||
        read(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderUtilization(device, &utilization, &samplingPeriodUs);

    if (write(connfd, &utilization, sizeof(unsigned int)) < 0 ||
        write(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderCapacity(int connfd)
{
    nvmlDevice_t device;
    nvmlEncoderType_t encoderQueryType;
    unsigned int encoderCapacity;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        read(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderCapacity(device, encoderQueryType, &encoderCapacity);

    if (write(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderStats(int connfd)
{
    nvmlDevice_t device;
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &averageFps, sizeof(unsigned int)) < 0 ||
        read(connfd, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetEncoderStats(device, &sessionCount, &averageFps, &averageLatency);

    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        write(connfd, &averageFps, sizeof(unsigned int)) < 0 ||
        write(connfd, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetEncoderSessions(int connfd)
{
    nvmlDevice_t device;
    unsigned int sessionCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &sessionCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlEncoderSessionInfo_t *sessionInfos = (nvmlEncoderSessionInfo_t *) malloc(sessionCount * sizeof(nvmlEncoderSessionInfo_t));

    nvmlReturn_t result = nvmlDeviceGetEncoderSessions(device, &sessionCount, sessionInfos);

    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        write(connfd, sessionInfos, sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDecoderUtilization(int connfd)
{
    nvmlDevice_t device;
    unsigned int utilization;
    unsigned int samplingPeriodUs;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &utilization, sizeof(unsigned int)) < 0 ||
        read(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDecoderUtilization(device, &utilization, &samplingPeriodUs);

    if (write(connfd, &utilization, sizeof(unsigned int)) < 0 ||
        write(connfd, &samplingPeriodUs, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFBCStats(int connfd)
{
    nvmlDevice_t device;
    nvmlFBCStats_t fbcStats;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFBCStats(device, &fbcStats);

    if (write(connfd, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFBCSessions(int connfd)
{
    nvmlDevice_t device;
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t sessionInfo;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetFBCSessions(device, &sessionCount, &sessionInfo);

    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        write(connfd, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDriverModel(int connfd)
{
    nvmlDevice_t device;
    nvmlDriverModel_t current;
    nvmlDriverModel_t pending;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &current, sizeof(nvmlDriverModel_t)) < 0 ||
        read(connfd, &pending, sizeof(nvmlDriverModel_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDriverModel(device, &current, &pending);

    if (write(connfd, &current, sizeof(nvmlDriverModel_t)) < 0 ||
        write(connfd, &pending, sizeof(nvmlDriverModel_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVbiosVersion(int connfd)
{
    nvmlDevice_t device;
    unsigned int length;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *version = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetVbiosVersion(device, version, length);

    if (write(connfd, version, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBridgeChipInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlBridgeChipHierarchy_t bridgeHierarchy;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy);

    if (write(connfd, &bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(int connfd)
{
    nvmlDevice_t device;
    unsigned int infoCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &infoCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *) malloc(infoCount * sizeof(nvmlProcessInfo_t));

    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses_v3(device, &infoCount, infos);

    if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
        write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(int connfd)
{
    nvmlDevice_t device;
    unsigned int infoCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &infoCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *) malloc(infoCount * sizeof(nvmlProcessInfo_t));

    nvmlReturn_t result = nvmlDeviceGetGraphicsRunningProcesses_v3(device, &infoCount, infos);

    if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
        write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(int connfd)
{
    nvmlDevice_t device;
    unsigned int infoCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &infoCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlProcessInfo_t *infos = (nvmlProcessInfo_t *) malloc(infoCount * sizeof(nvmlProcessInfo_t));

    nvmlReturn_t result = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &infoCount, infos);

    if (write(connfd, &infoCount, sizeof(unsigned int)) < 0 ||
        write(connfd, infos, infoCount * sizeof(nvmlProcessInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceOnSameBoard(int connfd)
{
    nvmlDevice_t device1;
    nvmlDevice_t device2;
    int onSameBoard;

    if (read(connfd, &device1, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &device2, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &onSameBoard, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceOnSameBoard(device1, device2, &onSameBoard);

    if (write(connfd, &onSameBoard, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAPIRestriction(int connfd)
{
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        read(connfd, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAPIRestriction(device, apiType, &isRestricted);

    if (write(connfd, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSamples(int connfd)
{
    nvmlDevice_t device;
    nvmlSamplingType_t type;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t sampleValType;
    unsigned int sampleCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlSamplingType_t)) < 0 ||
        read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        read(connfd, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        read(connfd, &sampleCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlSample_t *samples = (nvmlSample_t *) malloc(sampleCount * sizeof(nvmlSample_t));

    nvmlReturn_t result = nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, &sampleValType, &sampleCount, samples);

    if (write(connfd, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        write(connfd, &sampleCount, sizeof(unsigned int)) < 0 ||
        write(connfd, samples, sampleCount * sizeof(nvmlSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBAR1MemoryInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlBAR1Memory_t bar1Memory;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory);

    if (write(connfd, &bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetViolationStatus(int connfd)
{
    nvmlDevice_t device;
    nvmlPerfPolicyType_t perfPolicyType;
    nvmlViolationTime_t violTime;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        read(connfd, &violTime, sizeof(nvmlViolationTime_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetViolationStatus(device, perfPolicyType, &violTime);

    if (write(connfd, &violTime, sizeof(nvmlViolationTime_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetIrqNum(int connfd)
{
    nvmlDevice_t device;
    unsigned int irqNum;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &irqNum, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetIrqNum(device, &irqNum);

    if (write(connfd, &irqNum, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNumGpuCores(int connfd)
{
    nvmlDevice_t device;
    unsigned int numCores;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &numCores, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNumGpuCores(device, &numCores);

    if (write(connfd, &numCores, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPowerSource(int connfd)
{
    nvmlDevice_t device;
    nvmlPowerSource_t powerSource;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &powerSource, sizeof(nvmlPowerSource_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPowerSource(device, &powerSource);

    if (write(connfd, &powerSource, sizeof(nvmlPowerSource_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemoryBusWidth(int connfd)
{
    nvmlDevice_t device;
    unsigned int busWidth;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &busWidth, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemoryBusWidth(device, &busWidth);

    if (write(connfd, &busWidth, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieLinkMaxSpeed(int connfd)
{
    nvmlDevice_t device;
    unsigned int maxSpeed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieLinkMaxSpeed(device, &maxSpeed);

    if (write(connfd, &maxSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPcieSpeed(int connfd)
{
    nvmlDevice_t device;
    unsigned int pcieSpeed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pcieSpeed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed);

    if (write(connfd, &pcieSpeed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAdaptiveClockInfoStatus(int connfd)
{
    nvmlDevice_t device;
    unsigned int adaptiveClockStatus;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &adaptiveClockStatus, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptiveClockStatus);

    if (write(connfd, &adaptiveClockStatus, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingMode(device, &mode);

    if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingStats(int connfd)
{
    nvmlDevice_t device;
    unsigned int pid;
    nvmlAccountingStats_t stats;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pid, sizeof(unsigned int)) < 0 ||
        read(connfd, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingStats(device, pid, &stats);

    if (write(connfd, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingPids(int connfd)
{
    nvmlDevice_t device;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    unsigned int *pids = (unsigned int *) malloc(count * sizeof(unsigned int));

    nvmlReturn_t result = nvmlDeviceGetAccountingPids(device, &count, pids);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
        write(connfd, pids, count * sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetAccountingBufferSize(int connfd)
{
    nvmlDevice_t device;
    unsigned int bufferSize;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetAccountingBufferSize(device, &bufferSize);

    if (write(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPages(int connfd)
{
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        read(connfd, &pageCount, sizeof(unsigned int)) < 0)
        return -1;

    unsigned long long *addresses = (unsigned long long *) malloc(pageCount * sizeof(unsigned long long));

    nvmlReturn_t result = nvmlDeviceGetRetiredPages(device, cause, &pageCount, addresses);

    if (write(connfd, &pageCount, sizeof(unsigned int)) < 0 ||
        write(connfd, addresses, pageCount * sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPages_v2(int connfd)
{
    nvmlDevice_t device;
    nvmlPageRetirementCause_t cause;
    unsigned int pageCount;
    unsigned long long timestamps;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        read(connfd, &pageCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &timestamps, sizeof(unsigned long long)) < 0)
        return -1;

    unsigned long long *addresses = (unsigned long long *) malloc(pageCount * sizeof(unsigned long long));

    nvmlReturn_t result = nvmlDeviceGetRetiredPages_v2(device, cause, &pageCount, addresses, &timestamps);

    if (write(connfd, &pageCount, sizeof(unsigned int)) < 0 ||
        write(connfd, addresses, pageCount * sizeof(unsigned long long)) < 0 ||
        write(connfd, &timestamps, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRetiredPagesPendingStatus(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t isPending;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &isPending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRetiredPagesPendingStatus(device, &isPending);

    if (write(connfd, &isPending, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRemappedRows(int connfd)
{
    nvmlDevice_t device;
    unsigned int corrRows;
    unsigned int uncRows;
    unsigned int isPending;
    unsigned int failureOccurred;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &corrRows, sizeof(unsigned int)) < 0 ||
        read(connfd, &uncRows, sizeof(unsigned int)) < 0 ||
        read(connfd, &isPending, sizeof(unsigned int)) < 0 ||
        read(connfd, &failureOccurred, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRemappedRows(device, &corrRows, &uncRows, &isPending, &failureOccurred);

    if (write(connfd, &corrRows, sizeof(unsigned int)) < 0 ||
        write(connfd, &uncRows, sizeof(unsigned int)) < 0 ||
        write(connfd, &isPending, sizeof(unsigned int)) < 0 ||
        write(connfd, &failureOccurred, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetRowRemapperHistogram(int connfd)
{
    nvmlDevice_t device;
    nvmlRowRemapperHistogramValues_t values;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetRowRemapperHistogram(device, &values);

    if (write(connfd, &values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetArchitecture(int connfd)
{
    nvmlDevice_t device;
    nvmlDeviceArchitecture_t arch;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetArchitecture(device, &arch);

    if (write(connfd, &arch, sizeof(nvmlDeviceArchitecture_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlUnitSetLedState(int connfd)
{
    nvmlUnit_t unit;
    nvmlLedColor_t color;

    if (read(connfd, &unit, sizeof(nvmlUnit_t)) < 0 ||
        read(connfd, &color, sizeof(nvmlLedColor_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlUnitSetLedState(unit, color);

    return result;
}

int handle_nvmlDeviceSetPersistenceMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetPersistenceMode(device, mode);

    return result;
}

int handle_nvmlDeviceSetComputeMode(int connfd)
{
    nvmlDevice_t device;
    nvmlComputeMode_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlComputeMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetComputeMode(device, mode);

    return result;
}

int handle_nvmlDeviceSetEccMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t ecc;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &ecc, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetEccMode(device, ecc);

    return result;
}

int handle_nvmlDeviceClearEccErrorCounts(int connfd)
{
    nvmlDevice_t device;
    nvmlEccCounterType_t counterType;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &counterType, sizeof(nvmlEccCounterType_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearEccErrorCounts(device, counterType);

    return result;
}

int handle_nvmlDeviceSetDriverModel(int connfd)
{
    nvmlDevice_t device;
    nvmlDriverModel_t driverModel;
    unsigned int flags;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &driverModel, sizeof(nvmlDriverModel_t)) < 0 ||
        read(connfd, &flags, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetDriverModel(device, driverModel, flags);

    return result;
}

int handle_nvmlDeviceSetGpuLockedClocks(int connfd)
{
    nvmlDevice_t device;
    unsigned int minGpuClockMHz;
    unsigned int maxGpuClockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minGpuClockMHz, sizeof(unsigned int)) < 0 ||
        read(connfd, &maxGpuClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz);

    return result;
}

int handle_nvmlDeviceResetGpuLockedClocks(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetGpuLockedClocks(device);

    return result;
}

int handle_nvmlDeviceSetMemoryLockedClocks(int connfd)
{
    nvmlDevice_t device;
    unsigned int minMemClockMHz;
    unsigned int maxMemClockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minMemClockMHz, sizeof(unsigned int)) < 0 ||
        read(connfd, &maxMemClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetMemoryLockedClocks(device, minMemClockMHz, maxMemClockMHz);

    return result;
}

int handle_nvmlDeviceResetMemoryLockedClocks(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetMemoryLockedClocks(device);

    return result;
}

int handle_nvmlDeviceSetApplicationsClocks(int connfd)
{
    nvmlDevice_t device;
    unsigned int memClockMHz;
    unsigned int graphicsClockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &memClockMHz, sizeof(unsigned int)) < 0 ||
        read(connfd, &graphicsClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);

    return result;
}

int handle_nvmlDeviceGetClkMonStatus(int connfd)
{
    nvmlDevice_t device;
    nvmlClkMonStatus_t status;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &status, sizeof(nvmlClkMonStatus_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetClkMonStatus(device, &status);

    if (write(connfd, &status, sizeof(nvmlClkMonStatus_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetPowerManagementLimit(int connfd)
{
    nvmlDevice_t device;
    unsigned int limit;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &limit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetPowerManagementLimit(device, limit);

    return result;
}

int handle_nvmlDeviceSetGpuOperationMode(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuOperationMode_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlGpuOperationMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetGpuOperationMode(device, mode);

    return result;
}

int handle_nvmlDeviceSetAPIRestriction(int connfd)
{
    nvmlDevice_t device;
    nvmlRestrictedAPI_t apiType;
    nvmlEnableState_t isRestricted;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        read(connfd, &isRestricted, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetAPIRestriction(device, apiType, isRestricted);

    return result;
}

int handle_nvmlDeviceSetAccountingMode(int connfd)
{
    nvmlDevice_t device;
    nvmlEnableState_t mode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetAccountingMode(device, mode);

    return result;
}

int handle_nvmlDeviceClearAccountingPids(int connfd)
{
    nvmlDevice_t device;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceClearAccountingPids(device);

    return result;
}

int handle_nvmlDeviceGetNvLinkState(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlEnableState_t isActive;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkState(device, link, &isActive);

    if (write(connfd, &isActive, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkVersion(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int version;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &version, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkVersion(device, link, &version);

    if (write(connfd, &version, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkCapability(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlNvLinkCapability_t capability;
    unsigned int capResult;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        read(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkCapability(device, link, capability, &capResult);

    if (write(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkRemotePciInfo_v2(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlPciInfo_t pci;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, &pci);

    if (write(connfd, &pci, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkErrorCounter(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlNvLinkErrorCounter_t counter;
    unsigned long long counterValue;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        read(connfd, &counterValue, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkErrorCounter(device, link, counter, &counterValue);

    if (write(connfd, &counterValue, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceResetNvLinkErrorCounters(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetNvLinkErrorCounters(device, link);

    return result;
}

int handle_nvmlDeviceSetNvLinkUtilizationControl(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t control;
    unsigned int reset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &counter, sizeof(unsigned int)) < 0 ||
        read(connfd, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        read(connfd, &reset, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, &control, reset);

    if (write(connfd, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkUtilizationControl(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlNvLinkUtilizationControl_t control;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &counter, sizeof(unsigned int)) < 0 ||
        read(connfd, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, &control);

    if (write(connfd, &control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetNvLinkUtilizationCounter(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    unsigned long long rxcounter;
    unsigned long long txcounter;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &counter, sizeof(unsigned int)) < 0 ||
        read(connfd, &rxcounter, sizeof(unsigned long long)) < 0 ||
        read(connfd, &txcounter, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, &rxcounter, &txcounter);

    if (write(connfd, &rxcounter, sizeof(unsigned long long)) < 0 ||
        write(connfd, &txcounter, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceFreezeNvLinkUtilizationCounter(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;
    nvmlEnableState_t freeze;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &counter, sizeof(unsigned int)) < 0 ||
        read(connfd, &freeze, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze);

    return result;
}

int handle_nvmlDeviceResetNvLinkUtilizationCounter(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    unsigned int counter;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &counter, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter);

    return result;
}

int handle_nvmlDeviceGetNvLinkRemoteDeviceType(int connfd)
{
    nvmlDevice_t device;
    unsigned int link;
    nvmlIntNvLinkDeviceType_t pNvLinkDeviceType;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &link, sizeof(unsigned int)) < 0 ||
        read(connfd, &pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetNvLinkRemoteDeviceType(device, link, &pNvLinkDeviceType);

    if (write(connfd, &pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlEventSetCreate(int connfd)
{
    nvmlEventSet_t set;

    if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetCreate(&set);

    if (write(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceRegisterEvents(int connfd)
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

int handle_nvmlDeviceGetSupportedEventTypes(int connfd)
{
    nvmlDevice_t device;
    unsigned long long eventTypes;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &eventTypes, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedEventTypes(device, &eventTypes);

    if (write(connfd, &eventTypes, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlEventSetWait_v2(int connfd)
{
    nvmlEventSet_t set;
    nvmlEventData_t data;
    unsigned int timeoutms;

    if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0 ||
        read(connfd, &data, sizeof(nvmlEventData_t)) < 0 ||
        read(connfd, &timeoutms, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetWait_v2(set, &data, timeoutms);

    if (write(connfd, &data, sizeof(nvmlEventData_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlEventSetFree(int connfd)
{
    nvmlEventSet_t set;

    if (read(connfd, &set, sizeof(nvmlEventSet_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlEventSetFree(set);

    return result;
}

int handle_nvmlDeviceModifyDrainState(int connfd)
{
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t newState;

    if (read(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        read(connfd, &newState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceModifyDrainState(&pciInfo, newState);

    if (write(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceQueryDrainState(int connfd)
{
    nvmlPciInfo_t pciInfo;
    nvmlEnableState_t currentState;

    if (read(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        read(connfd, &currentState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceQueryDrainState(&pciInfo, &currentState);

    if (write(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        write(connfd, &currentState, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceRemoveGpu_v2(int connfd)
{
    nvmlPciInfo_t pciInfo;
    nvmlDetachGpuState_t gpuState;
    nvmlPcieLinkState_t linkState;

    if (read(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        read(connfd, &gpuState, sizeof(nvmlDetachGpuState_t)) < 0 ||
        read(connfd, &linkState, sizeof(nvmlPcieLinkState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceRemoveGpu_v2(&pciInfo, gpuState, linkState);

    if (write(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceDiscoverGpus(int connfd)
{
    nvmlPciInfo_t pciInfo;

    if (read(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceDiscoverGpus(&pciInfo);

    if (write(connfd, &pciInfo, sizeof(nvmlPciInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetFieldValues(int connfd)
{
    nvmlDevice_t device;
    int valuesCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &valuesCount, sizeof(int)) < 0)
        return -1;

    nvmlFieldValue_t *values = (nvmlFieldValue_t *) malloc(valuesCount * sizeof(nvmlFieldValue_t));

    nvmlReturn_t result = nvmlDeviceGetFieldValues(device, valuesCount, values);

    if (write(connfd, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceClearFieldValues(int connfd)
{
    nvmlDevice_t device;
    int valuesCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &valuesCount, sizeof(int)) < 0)
        return -1;

    nvmlFieldValue_t *values = (nvmlFieldValue_t *) malloc(valuesCount * sizeof(nvmlFieldValue_t));

    nvmlReturn_t result = nvmlDeviceClearFieldValues(device, valuesCount, values);

    if (write(connfd, values, valuesCount * sizeof(nvmlFieldValue_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVirtualizationMode(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuVirtualizationMode_t pVirtualMode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVirtualizationMode(device, &pVirtualMode);

    if (write(connfd, &pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetHostVgpuMode(int connfd)
{
    nvmlDevice_t device;
    nvmlHostVgpuMode_t pHostVgpuMode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetHostVgpuMode(device, &pHostVgpuMode);

    if (write(connfd, &pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetVirtualizationMode(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuVirtualizationMode_t virtualMode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetVirtualizationMode(device, virtualMode);

    return result;
}

int handle_nvmlDeviceGetGridLicensableFeatures_v4(int connfd)
{
    nvmlDevice_t device;
    nvmlGridLicensableFeatures_t pGridLicensableFeatures;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGridLicensableFeatures_v4(device, &pGridLicensableFeatures);

    if (write(connfd, &pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetProcessUtilization(int connfd)
{
    nvmlDevice_t device;
    nvmlProcessUtilizationSample_t utilization;
    unsigned int processSamplesCount;
    unsigned long long lastSeenTimeStamp;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        read(connfd, &processSamplesCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetProcessUtilization(device, &utilization, &processSamplesCount, lastSeenTimeStamp);

    if (write(connfd, &utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        write(connfd, &processSamplesCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGspFirmwareVersion(int connfd)
{
    nvmlDevice_t device;
    char version;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &version, sizeof(char)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGspFirmwareVersion(device, &version);

    if (write(connfd, &version, sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGspFirmwareMode(int connfd)
{
    nvmlDevice_t device;
    unsigned int isEnabled;
    unsigned int defaultMode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &isEnabled, sizeof(unsigned int)) < 0 ||
        read(connfd, &defaultMode, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGspFirmwareMode(device, &isEnabled, &defaultMode);

    if (write(connfd, &isEnabled, sizeof(unsigned int)) < 0 ||
        write(connfd, &defaultMode, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuDriverCapabilities(int connfd)
{
    nvmlVgpuDriverCapability_t capability;
    unsigned int capResult;

    if (read(connfd, &capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        read(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuDriverCapabilities(capability, &capResult);

    if (write(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuCapabilities(int connfd)
{
    nvmlDevice_t device;
    nvmlDeviceVgpuCapability_t capability;
    unsigned int capResult;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        read(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuCapabilities(device, capability, &capResult);

    if (write(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedVgpus(int connfd)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlVgpuTypeId_t *vgpuTypeIds = (nvmlVgpuTypeId_t *) malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    nvmlReturn_t result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, vgpuTypeIds);

    if (write(connfd, &vgpuCount, sizeof(unsigned int)) < 0 ||
        write(connfd, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetCreatableVgpus(int connfd)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlVgpuTypeId_t *vgpuTypeIds = (nvmlVgpuTypeId_t *) malloc(vgpuCount * sizeof(nvmlVgpuTypeId_t));

    nvmlReturn_t result = nvmlDeviceGetCreatableVgpus(device, &vgpuCount, vgpuTypeIds);

    if (write(connfd, &vgpuCount, sizeof(unsigned int)) < 0 ||
        write(connfd, vgpuTypeIds, vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetClass(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    char vgpuTypeClass;
    unsigned int size;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &vgpuTypeClass, sizeof(char)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetClass(vgpuTypeId, &vgpuTypeClass, &size);

    if (write(connfd, &vgpuTypeClass, sizeof(char)) < 0 ||
        write(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetName(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    char *vgpuTypeName = (char *) malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, &size);

    if (write(connfd, vgpuTypeName, size * sizeof(char)) < 0 ||
        write(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetGpuInstanceProfileId(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int gpuInstanceProfileId;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &gpuInstanceProfileId, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, &gpuInstanceProfileId);

    if (write(connfd, &gpuInstanceProfileId, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetDeviceID(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long deviceID;
    unsigned long long subsystemID;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &deviceID, sizeof(unsigned long long)) < 0 ||
        read(connfd, &subsystemID, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetDeviceID(vgpuTypeId, &deviceID, &subsystemID);

    if (write(connfd, &deviceID, sizeof(unsigned long long)) < 0 ||
        write(connfd, &subsystemID, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetFramebufferSize(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned long long fbSize;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &fbSize, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, &fbSize);

    if (write(connfd, &fbSize, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetNumDisplayHeads(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int numDisplayHeads;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &numDisplayHeads, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, &numDisplayHeads);

    if (write(connfd, &numDisplayHeads, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetResolution(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int displayIndex;
    unsigned int xdim;
    unsigned int ydim;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &displayIndex, sizeof(unsigned int)) < 0 ||
        read(connfd, &xdim, sizeof(unsigned int)) < 0 ||
        read(connfd, &ydim, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, &xdim, &ydim);

    if (write(connfd, &xdim, sizeof(unsigned int)) < 0 ||
        write(connfd, &ydim, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetLicense(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int size;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    char *vgpuTypeLicenseString = (char *) malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size);

    if (write(connfd, vgpuTypeLicenseString, size * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetFrameRateLimit(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int frameRateLimit;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, &frameRateLimit);

    if (write(connfd, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetMaxInstances(int connfd)
{
    nvmlDevice_t device;
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int vgpuInstanceCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &vgpuInstanceCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, &vgpuInstanceCount);

    if (write(connfd, &vgpuInstanceCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetMaxInstancesPerVm(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    unsigned int vgpuInstanceCountPerVm;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, &vgpuInstanceCountPerVm);

    if (write(connfd, &vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetActiveVgpus(int connfd)
{
    nvmlDevice_t device;
    unsigned int vgpuCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &vgpuCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlVgpuInstance_t *vgpuInstances = (nvmlVgpuInstance_t *) malloc(vgpuCount * sizeof(nvmlVgpuInstance_t));

    nvmlReturn_t result = nvmlDeviceGetActiveVgpus(device, &vgpuCount, vgpuInstances);

    if (write(connfd, &vgpuCount, sizeof(unsigned int)) < 0 ||
        write(connfd, vgpuInstances, vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetVmID(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;
    nvmlVgpuVmIdType_t vmIdType;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0 ||
        read(connfd, &vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0)
        return -1;

    char *vmId = (char *) malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, &vmIdType);

    if (write(connfd, vmId, size * sizeof(char)) < 0 ||
        write(connfd, &vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetUUID(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int size;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    char *uuid = (char *) malloc(size * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size);

    if (write(connfd, uuid, size * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetVmDriverVersion(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int length;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *version = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length);

    if (write(connfd, version, length * sizeof(char)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFbUsage(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned long long fbUsage;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &fbUsage, sizeof(unsigned long long)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFbUsage(vgpuInstance, &fbUsage);

    if (write(connfd, &fbUsage, sizeof(unsigned long long)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetLicenseStatus(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int licensed;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &licensed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, &licensed);

    if (write(connfd, &licensed, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetType(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuTypeId_t vgpuTypeId;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetType(vgpuInstance, &vgpuTypeId);

    if (write(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFrameRateLimit(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int frameRateLimit;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, &frameRateLimit);

    if (write(connfd, &frameRateLimit, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEccMode(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t eccMode;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &eccMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEccMode(vgpuInstance, &eccMode);

    if (write(connfd, &eccMode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderCapacity(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int encoderCapacity;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, &encoderCapacity);

    if (write(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceSetEncoderCapacity(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int encoderCapacity;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &encoderCapacity, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity);

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderStats(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    unsigned int averageFps;
    unsigned int averageLatency;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &averageFps, sizeof(unsigned int)) < 0 ||
        read(connfd, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderStats(vgpuInstance, &sessionCount, &averageFps, &averageLatency);

    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        write(connfd, &averageFps, sizeof(unsigned int)) < 0 ||
        write(connfd, &averageLatency, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetEncoderSessions(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    nvmlEncoderSessionInfo_t sessionInfo;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &sessionCount, &sessionInfo);

    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        write(connfd, &sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFBCStats(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlFBCStats_t fbcStats;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFBCStats(vgpuInstance, &fbcStats);

    if (write(connfd, &fbcStats, sizeof(nvmlFBCStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetFBCSessions(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int sessionCount;
    nvmlFBCSessionInfo_t sessionInfo;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        read(connfd, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetFBCSessions(vgpuInstance, &sessionCount, &sessionInfo);

    if (write(connfd, &sessionCount, sizeof(unsigned int)) < 0 ||
        write(connfd, &sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetGpuInstanceId(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int gpuInstanceId;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &gpuInstanceId, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, &gpuInstanceId);

    if (write(connfd, &gpuInstanceId, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetGpuPciId(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int length;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    char *vgpuPciId = (char *) malloc(length * sizeof(char));

    nvmlReturn_t result = nvmlVgpuInstanceGetGpuPciId(vgpuInstance, vgpuPciId, &length);

    if (write(connfd, vgpuPciId, length * sizeof(char)) < 0 ||
        write(connfd, &length, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuTypeGetCapabilities(int connfd)
{
    nvmlVgpuTypeId_t vgpuTypeId;
    nvmlVgpuCapability_t capability;
    unsigned int capResult;

    if (read(connfd, &vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        read(connfd, &capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        read(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability, &capResult);

    if (write(connfd, &capResult, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetMetadata(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuMetadata_t vgpuMetadata;
    unsigned int bufferSize;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        read(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetMetadata(vgpuInstance, &vgpuMetadata, &bufferSize);

    if (write(connfd, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        write(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuMetadata(int connfd)
{
    nvmlDevice_t device;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    unsigned int bufferSize;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        read(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuMetadata(device, &pgpuMetadata, &bufferSize);

    if (write(connfd, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        write(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuCompatibility(int connfd)
{
    nvmlVgpuMetadata_t vgpuMetadata;
    nvmlVgpuPgpuMetadata_t pgpuMetadata;
    nvmlVgpuPgpuCompatibility_t compatibilityInfo;

    if (read(connfd, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        read(connfd, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        read(connfd, &compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &compatibilityInfo);

    if (write(connfd, &vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        write(connfd, &pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        write(connfd, &compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetPgpuMetadataString(int connfd)
{
    nvmlDevice_t device;
    unsigned int bufferSize;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    char *pgpuMetadata = (char *) malloc(bufferSize * sizeof(char));

    nvmlReturn_t result = nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, &bufferSize);

    if (write(connfd, pgpuMetadata, bufferSize * sizeof(char)) < 0 ||
        write(connfd, &bufferSize, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerLog(int connfd)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerLog_t pSchedulerLog;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuSchedulerLog(device, &pSchedulerLog);

    if (write(connfd, &pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerState(int connfd)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerGetState_t pSchedulerState;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuSchedulerState(device, &pSchedulerState);

    if (write(connfd, &pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuSchedulerCapabilities(int connfd)
{
    nvmlDevice_t device;
    nvmlVgpuSchedulerCapabilities_t pCapabilities;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetVgpuSchedulerCapabilities(device, &pCapabilities);

    if (write(connfd, &pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetVgpuVersion(int connfd)
{
    nvmlVgpuVersion_t supported;
    nvmlVgpuVersion_t current;

    if (read(connfd, &supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        read(connfd, &current, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGetVgpuVersion(&supported, &current);

    if (write(connfd, &supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        write(connfd, &current, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlSetVgpuVersion(int connfd)
{
    nvmlVgpuVersion_t vgpuVersion;

    if (read(connfd, &vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlSetVgpuVersion(&vgpuVersion);

    if (write(connfd, &vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuUtilization(int connfd)
{
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    nvmlValueType_t sampleValType;
    unsigned int vgpuInstanceSamplesCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        read(connfd, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        read(connfd, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlVgpuInstanceUtilizationSample_t *utilizationSamples = (nvmlVgpuInstanceUtilizationSample_t *) malloc(vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t));

    nvmlReturn_t result = nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, &sampleValType, &vgpuInstanceSamplesCount, utilizationSamples);

    if (write(connfd, &sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        write(connfd, &vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        write(connfd, utilizationSamples, vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetVgpuProcessUtilization(int connfd)
{
    nvmlDevice_t device;
    unsigned long long lastSeenTimeStamp;
    unsigned int vgpuProcessSamplesCount;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        read(connfd, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlVgpuProcessUtilizationSample_t *utilizationSamples = (nvmlVgpuProcessUtilizationSample_t *) malloc(vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t));

    nvmlReturn_t result = nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp, &vgpuProcessSamplesCount, utilizationSamples);

    if (write(connfd, &vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        write(connfd, utilizationSamples, vgpuProcessSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingMode(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlEnableState_t mode;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingMode(vgpuInstance, &mode);

    if (write(connfd, &mode, sizeof(nvmlEnableState_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingPids(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int count;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    unsigned int *pids = (unsigned int *) malloc(count * sizeof(unsigned int));

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingPids(vgpuInstance, &count, pids);

    if (write(connfd, &count, sizeof(unsigned int)) < 0 ||
        write(connfd, pids, count * sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceGetAccountingStats(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    unsigned int pid;
    nvmlAccountingStats_t stats;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &pid, sizeof(unsigned int)) < 0 ||
        read(connfd, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, &stats);

    if (write(connfd, &stats, sizeof(nvmlAccountingStats_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlVgpuInstanceClearAccountingPids(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceClearAccountingPids(vgpuInstance);

    return result;
}

int handle_nvmlVgpuInstanceGetLicenseInfo_v2(int connfd)
{
    nvmlVgpuInstance_t vgpuInstance;
    nvmlVgpuLicenseInfo_t licenseInfo;

    if (read(connfd, &vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        read(connfd, &licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, &licenseInfo);

    if (write(connfd, &licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetExcludedDeviceCount(int connfd)
{
    unsigned int deviceCount;

    if (read(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGetExcludedDeviceCount(&deviceCount);

    if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGetExcludedDeviceInfoByIndex(int connfd)
{
    unsigned int index;
    nvmlExcludedDeviceInfo_t info;

    if (read(connfd, &index, sizeof(unsigned int)) < 0 ||
        read(connfd, &info, sizeof(nvmlExcludedDeviceInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGetExcludedDeviceInfoByIndex(index, &info);

    if (write(connfd, &info, sizeof(nvmlExcludedDeviceInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetMigMode(int connfd)
{
    nvmlDevice_t device;
    unsigned int mode;
    nvmlReturn_t activationStatus;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &mode, sizeof(unsigned int)) < 0 ||
        read(connfd, &activationStatus, sizeof(nvmlReturn_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetMigMode(device, mode, &activationStatus);

    if (write(connfd, &activationStatus, sizeof(nvmlReturn_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMigMode(int connfd)
{
    nvmlDevice_t device;
    unsigned int currentMode;
    unsigned int pendingMode;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &currentMode, sizeof(unsigned int)) < 0 ||
        read(connfd, &pendingMode, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);

    if (write(connfd, &currentMode, sizeof(unsigned int)) < 0 ||
        write(connfd, &pendingMode, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfo(int connfd)
{
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_t info;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profile, sizeof(unsigned int)) < 0 ||
        read(connfd, &info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, &info);

    if (write(connfd, &info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceProfileInfoV(int connfd)
{
    nvmlDevice_t device;
    unsigned int profile;
    nvmlGpuInstanceProfileInfo_v2_t info;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profile, sizeof(unsigned int)) < 0 ||
        read(connfd, &info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfoV(device, profile, &info);

    if (write(connfd, &info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstancePossiblePlacements_v2(int connfd)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlGpuInstancePlacement_t *placements = (nvmlGpuInstancePlacement_t *) malloc(count * sizeof(nvmlGpuInstancePlacement_t));

    nvmlReturn_t result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, profileId, placements, &count);

    if (write(connfd, placements, count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceRemainingCapacity(int connfd)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, &count);

    if (write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceCreateGpuInstance(int connfd)
{
    nvmlDevice_t device;
    unsigned int profileId;
    nvmlGpuInstance_t gpuInstance;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCreateGpuInstance(device, profileId, &gpuInstance);

    if (write(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceCreateGpuInstanceWithPlacement(int connfd)
{
    nvmlDevice_t device;
    unsigned int profileId;
    nvmlGpuInstancePlacement_t placement;
    nvmlGpuInstance_t gpuInstance;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &placement, sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, &placement, &gpuInstance);

    if (write(connfd, &placement, sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        write(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceDestroy(int connfd)
{
    nvmlGpuInstance_t gpuInstance;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceDestroy(gpuInstance);

    return result;
}

int handle_nvmlDeviceGetGpuInstances(int connfd)
{
    nvmlDevice_t device;
    unsigned int profileId;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlGpuInstance_t *gpuInstances = (nvmlGpuInstance_t *) malloc(count * sizeof(nvmlGpuInstance_t));

    nvmlReturn_t result = nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, &count);

    if (write(connfd, gpuInstances, count * sizeof(nvmlGpuInstance_t)) < 0 ||
        write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceById(int connfd)
{
    nvmlDevice_t device;
    unsigned int id;
    nvmlGpuInstance_t gpuInstance;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &id, sizeof(unsigned int)) < 0 ||
        read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceById(device, id, &gpuInstance);

    if (write(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetInfo(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    nvmlGpuInstanceInfo_t info;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &info, sizeof(nvmlGpuInstanceInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetInfo(gpuInstance, &info);

    if (write(connfd, &info, sizeof(nvmlGpuInstanceInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfo(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_t info;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profile, sizeof(unsigned int)) < 0 ||
        read(connfd, &engProfile, sizeof(unsigned int)) < 0 ||
        read(connfd, &info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile, engProfile, &info);

    if (write(connfd, &info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceProfileInfoV(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profile;
    unsigned int engProfile;
    nvmlComputeInstanceProfileInfo_v2_t info;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profile, sizeof(unsigned int)) < 0 ||
        read(connfd, &engProfile, sizeof(unsigned int)) < 0 ||
        read(connfd, &info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, profile, engProfile, &info);

    if (write(connfd, &info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceRemainingCapacity(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int count;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, profileId, &count);

    if (write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstancePossiblePlacements(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    nvmlComputeInstancePlacement_t placements;
    unsigned int count;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, &placements, &count);

    if (write(connfd, &placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceCreateComputeInstance(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    nvmlComputeInstance_t computeInstance;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId, &computeInstance);

    if (write(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceCreateComputeInstanceWithPlacement(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    nvmlComputeInstancePlacement_t placement;
    nvmlComputeInstance_t computeInstance;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &placement, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        read(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceCreateComputeInstanceWithPlacement(gpuInstance, profileId, &placement, &computeInstance);

    if (write(connfd, &placement, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        write(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlComputeInstanceDestroy(int connfd)
{
    nvmlComputeInstance_t computeInstance;

    if (read(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlComputeInstanceDestroy(computeInstance);

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstances(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int profileId;
    unsigned int count;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &profileId, sizeof(unsigned int)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlComputeInstance_t *computeInstances = (nvmlComputeInstance_t *) malloc(count * sizeof(nvmlComputeInstance_t));

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId, computeInstances, &count);

    if (write(connfd, computeInstances, count * sizeof(nvmlComputeInstance_t)) < 0 ||
        write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpuInstanceGetComputeInstanceById(int connfd)
{
    nvmlGpuInstance_t gpuInstance;
    unsigned int id;
    nvmlComputeInstance_t computeInstance;

    if (read(connfd, &gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        read(connfd, &id, sizeof(unsigned int)) < 0 ||
        read(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, &computeInstance);

    if (write(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlComputeInstanceGetInfo_v2(int connfd)
{
    nvmlComputeInstance_t computeInstance;
    nvmlComputeInstanceInfo_t info;

    if (read(connfd, &computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        read(connfd, &info, sizeof(nvmlComputeInstanceInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlComputeInstanceGetInfo_v2(computeInstance, &info);

    if (write(connfd, &info, sizeof(nvmlComputeInstanceInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceIsMigDeviceHandle(int connfd)
{
    nvmlDevice_t device;
    unsigned int isMigDevice;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &isMigDevice, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceIsMigDeviceHandle(device, &isMigDevice);

    if (write(connfd, &isMigDevice, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuInstanceId(int connfd)
{
    nvmlDevice_t device;
    unsigned int id;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &id, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuInstanceId(device, &id);

    if (write(connfd, &id, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetComputeInstanceId(int connfd)
{
    nvmlDevice_t device;
    unsigned int id;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &id, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetComputeInstanceId(device, &id);

    if (write(connfd, &id, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMaxMigDeviceCount(int connfd)
{
    nvmlDevice_t device;
    unsigned int count;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMaxMigDeviceCount(device, &count);

    if (write(connfd, &count, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMigDeviceHandleByIndex(int connfd)
{
    nvmlDevice_t device;
    unsigned int index;
    nvmlDevice_t migDevice;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &index, sizeof(unsigned int)) < 0 ||
        read(connfd, &migDevice, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMigDeviceHandleByIndex(device, index, &migDevice);

    if (write(connfd, &migDevice, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDeviceHandleFromMigDeviceHandle(int connfd)
{
    nvmlDevice_t migDevice;
    nvmlDevice_t device;

    if (read(connfd, &migDevice, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, &device);

    if (write(connfd, &device, sizeof(nvmlDevice_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetBusType(int connfd)
{
    nvmlDevice_t device;
    nvmlBusType_t type;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlBusType_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetBusType(device, &type);

    if (write(connfd, &type, sizeof(nvmlBusType_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetDynamicPstatesInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuDynamicPstatesInfo_t pDynamicPstatesInfo;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetDynamicPstatesInfo(device, &pDynamicPstatesInfo);

    if (write(connfd, &pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetFanSpeed_v2(int connfd)
{
    nvmlDevice_t device;
    unsigned int fan;
    unsigned int speed;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &fan, sizeof(unsigned int)) < 0 ||
        read(connfd, &speed, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetFanSpeed_v2(device, fan, speed);

    return result;
}

int handle_nvmlDeviceGetGpcClkVfOffset(int connfd)
{
    nvmlDevice_t device;
    int offset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &offset, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpcClkVfOffset(device, &offset);

    if (write(connfd, &offset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetGpcClkVfOffset(int connfd)
{
    nvmlDevice_t device;
    int offset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &offset, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetGpcClkVfOffset(device, offset);

    return result;
}

int handle_nvmlDeviceGetMemClkVfOffset(int connfd)
{
    nvmlDevice_t device;
    int offset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &offset, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemClkVfOffset(device, &offset);

    if (write(connfd, &offset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceSetMemClkVfOffset(int connfd)
{
    nvmlDevice_t device;
    int offset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &offset, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetMemClkVfOffset(device, offset);

    return result;
}

int handle_nvmlDeviceGetMinMaxClockOfPState(int connfd)
{
    nvmlDevice_t device;
    nvmlClockType_t type;
    nvmlPstates_t pstate;
    unsigned int minClockMHz;
    unsigned int maxClockMHz;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &type, sizeof(nvmlClockType_t)) < 0 ||
        read(connfd, &pstate, sizeof(nvmlPstates_t)) < 0 ||
        read(connfd, &minClockMHz, sizeof(unsigned int)) < 0 ||
        read(connfd, &maxClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMinMaxClockOfPState(device, type, pstate, &minClockMHz, &maxClockMHz);

    if (write(connfd, &minClockMHz, sizeof(unsigned int)) < 0 ||
        write(connfd, &maxClockMHz, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetSupportedPerformanceStates(int connfd)
{
    nvmlDevice_t device;
    nvmlPstates_t pstates;
    unsigned int size;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &pstates, sizeof(nvmlPstates_t)) < 0 ||
        read(connfd, &size, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetSupportedPerformanceStates(device, &pstates, size);

    if (write(connfd, &pstates, sizeof(nvmlPstates_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpcClkMinMaxVfOffset(int connfd)
{
    nvmlDevice_t device;
    int minOffset;
    int maxOffset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minOffset, sizeof(int)) < 0 ||
        read(connfd, &maxOffset, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpcClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (write(connfd, &minOffset, sizeof(int)) < 0 ||
        write(connfd, &maxOffset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetMemClkMinMaxVfOffset(int connfd)
{
    nvmlDevice_t device;
    int minOffset;
    int maxOffset;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &minOffset, sizeof(int)) < 0 ||
        read(connfd, &maxOffset, sizeof(int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetMemClkMinMaxVfOffset(device, &minOffset, &maxOffset);

    if (write(connfd, &minOffset, sizeof(int)) < 0 ||
        write(connfd, &maxOffset, sizeof(int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceGetGpuFabricInfo(int connfd)
{
    nvmlDevice_t device;
    nvmlGpuFabricInfo_t gpuFabricInfo;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo);

    if (write(connfd, &gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmMetricsGet(int connfd)
{
    nvmlGpmMetricsGet_t metricsGet;

    if (read(connfd, &metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmMetricsGet(&metricsGet);

    if (write(connfd, &metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmSampleFree(int connfd)
{
    nvmlGpmSample_t gpmSample;

    if (read(connfd, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleFree(gpmSample);

    return result;
}

int handle_nvmlGpmSampleAlloc(int connfd)
{
    nvmlGpmSample_t gpmSample;

    if (read(connfd, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleAlloc(&gpmSample);

    if (write(connfd, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlGpmSampleGet(int connfd)
{
    nvmlDevice_t device;
    nvmlGpmSample_t gpmSample;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmSampleGet(device, gpmSample);

    return result;
}

int handle_nvmlGpmMigSampleGet(int connfd)
{
    nvmlDevice_t device;
    unsigned int gpuInstanceId;
    nvmlGpmSample_t gpmSample;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &gpuInstanceId, sizeof(unsigned int)) < 0 ||
        read(connfd, &gpmSample, sizeof(nvmlGpmSample_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample);

    return result;
}

int handle_nvmlGpmQueryDeviceSupport(int connfd)
{
    nvmlDevice_t device;
    nvmlGpmSupport_t gpmSupport;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &gpmSupport, sizeof(nvmlGpmSupport_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlGpmQueryDeviceSupport(device, &gpmSupport);

    if (write(connfd, &gpmSupport, sizeof(nvmlGpmSupport_t)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceCcuGetStreamState(int connfd)
{
    nvmlDevice_t device;
    unsigned int state;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &state, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCcuGetStreamState(device, &state);

    if (write(connfd, &state, sizeof(unsigned int)) < 0)
        return -1;

    return result;
}

int handle_nvmlDeviceCcuSetStreamState(int connfd)
{
    nvmlDevice_t device;
    unsigned int state;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &state, sizeof(unsigned int)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceCcuSetStreamState(device, state);

    return result;
}

int handle_nvmlDeviceSetNvLinkDeviceLowPowerThreshold(int connfd)
{
    nvmlDevice_t device;
    nvmlNvLinkPowerThres_t info;

    if (read(connfd, &device, sizeof(nvmlDevice_t)) < 0 ||
        read(connfd, &info, sizeof(nvmlNvLinkPowerThres_t)) < 0)
        return -1;

    nvmlReturn_t result = nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, &info);

    if (write(connfd, &info, sizeof(nvmlNvLinkPowerThres_t)) < 0)
        return -1;

    return result;
}

