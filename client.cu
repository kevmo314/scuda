#include <arpa/inet.h>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <netdb.h>
#include <nvml.h>
#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <unordered_map>

#include "api.h"

int sockfd = -1;
char *port;

int open_rpc_client()
{
    // if socket is already opened, return our socket.
    if (sockfd != -1)
    {
        // << "socket already opened" << std::endl;

        return sockfd;
    }

    // << "opening tcp socket..." << std::endl;

    char *server_ip = getenv("SCUDA_SERVER");
    if (server_ip == NULL)
    {
        printf("SCUDA_SERVER environment variable not set\n");
        std::exit(1);
    }

    char *p = getenv("SCUDA_PORT");

    if (p == NULL)
    {
        port = (char *)"14833";
    }
    else
    {
        port = p;
        // << "using SCUDA_PORT: " << port << std::endl;
    }

    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(server_ip, port, &hints, &res) != 0)
    {
        printf("getaddrinfo failed\n");
        return -1;
    }

    sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1)
    {
        printf("socket creation failed...\n");
        return -1;
    }

    if (connect(sockfd, res->ai_addr, res->ai_addrlen) != 0)
    {
        printf("connection with the server failed...\n");
        return -1;
    }
    return sockfd;
}

pthread_mutex_t mutex;
pthread_cond_t cond;

int rpc_start_request(const unsigned int op)
{
    static int next_request_id = 1;

    // Ensure socket is open
    if (sockfd < 0) {
        std::cerr << "Socket not open" << std::endl;
        return -1;
    }

    // Lock the mutex for atomic operation
    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++;

    // Write the request ID and operation code
    if (write(sockfd, &request_id, sizeof(int)) < 0)
    {
        std::cerr << "Failed to write request_id. Error: " << strerror(errno) << std::endl;
        pthread_mutex_unlock(&mutex);
        return -1;
    }

    if (write(sockfd, &op, sizeof(unsigned int)) < 0)
    {
        std::cerr << "Failed to write operation code. Error: " << strerror(errno) << std::endl;
        pthread_mutex_unlock(&mutex);
        return -1;
    }

    // Return the request ID
    return request_id;
}


int rpc_write(const void *data, size_t size)
{
    if (write(sockfd, data, size) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    return 0;
}

int rpc_read(void *data, size_t size)
{
    if (read(sockfd, data, size) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    return 0;
}

int rpc_wait_for_response(int request_id)
{
    static int active_response_id = -1;

    // wait for the response
    while (true)
    {
        while (active_response_id != request_id && active_response_id != -1)
            pthread_cond_wait(&cond, &mutex);

        // we currently own mutex. if active response id is -1, read the response id
        if (active_response_id == -1)
        {
            if (read(sockfd, &active_response_id, sizeof(int)) < 0)
            {
                pthread_mutex_unlock(&mutex);
                return -1;
            }

            if (active_response_id != request_id)
            {
                pthread_cond_broadcast(&cond);
                continue;
            }
        }

        active_response_id = -1;
        return 0;
    }
}

template <typename T>
T rpc_get_return(int request_id, T error_value)
{
    T result;
    if (read(sockfd, &result, sizeof(T)) < 0)
        result = error_value;

    pthread_mutex_unlock(&mutex);

    return result;
}

void close_rpc_client()
{
    close(sockfd);
    sockfd = 0;
}

// 4.11 Initialization and Cleanup
nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    if (open_rpc_client() < 0)
        return NVML_ERROR_GPU_IS_LOST;
    int request_id = rpc_start_request(RPC_nvmlInitWithFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;

    nvmlReturn_t result = rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
    return result;
}

nvmlReturn_t nvmlInit_v2()
{
    if (open_rpc_client() < 0)
        return NVML_ERROR_GPU_IS_LOST;
    int request_id = rpc_start_request(RPC_nvmlInit_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlShutdown()
{
    int request_id = rpc_start_request(RPC_nvmlShutdown);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    nvmlReturn_t result = rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
    close_rpc_client();
    return result;
}

// 4.14 System Queries
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(version, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount,
                                     nvmlHwbcEntry_t *hwbcEntries)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    if (request_id < 0 ||
        rpc_write(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_read(hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    nvmlReturn_t result = rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
    close_rpc_client();
    return result;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetNVMLVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(version, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name,
                                      unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetProcessName);
    if (request_id < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(name, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber,
                                         unsigned int *count,
                                         nvmlDevice_t *deviceArray)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetTopologyGpuSet);
    if (request_id < 0 ||
        rpc_write(&cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

// 4.15 Unit Queries
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetCount);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unitCount, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount,
                                nvmlDevice_t *devices)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetDevices);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_read(devices, *deviceCount * sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit,
                                     nvmlUnitFanSpeeds_t *fanSpeeds)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetFanSpeedInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unit, sizeof(nvmlUnit_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetLedState);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(state, sizeof(nvmlLedState_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetPsuInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(psu, sizeof(nvmlPSUInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type,
                                    unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetTemperature);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&type, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetUnitInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlUnitInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

// 4.16 Device Queries
nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClkMonStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(status, sizeof(nvmlClkMonStatus_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int *clockMHz)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int *clock)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlComputeMode_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCount_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayActive);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(display, sizeof(nvmlEnableState_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetDriverModel_v2(nvmlDevice_t device, nvmlDriverModel_t *current, nvmlDriverModel_t *pending)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDriverModel_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlDriverModel_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDynamicPstatesInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current, nvmlEnableState_t *pending)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlEnableState_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount, nvmlEncoderSessionInfo_t *sessionInfos)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfos, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount, unsigned int *averageFps, unsigned int *averageLatency)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEnforcedPowerLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                           nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(index, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device,
                                        nvmlMemory_v2_t *memoryInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memoryInfo, sizeof(nvmlMemory_v2_t)) < 0)
    {
        
        return NVML_ERROR_GPU_IS_LOST;
    }

    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                               unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetName);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(name, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPciInfo_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int *pcieSpeed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pcieSpeed, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int *value)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieThroughput);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_write(value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPerformanceState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device,
                                          nvmlEnableState_t *mode)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPersistenceMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t *powerSource)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerSource);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(powerSource, sizeof(nvmlPowerSource_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerUsage);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(power, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization, unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetProcessUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t *pstates, unsigned int size)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedPerformanceStates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pstates, sizeof(nvmlPstates_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pstates, size * sizeof(nvmlPstates_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int *targetSpeed)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTargetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(targetSpeed, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperature);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t *pThermalSettings)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetThermalSettings);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t *pathInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int *count, nvmlDevice_t *deviceArray)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyNearestGpus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long *energy)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEnergyConsumption);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(energy, sizeof(unsigned long long)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetUUID);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetUtilizationRates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(nvmlUtilization_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceValidateInforom);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

// 4.17 Unit Commands
nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color)
{
    int request_id = rpc_start_request(RPC_nvmlUnitSetLedState);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&color, sizeof(nvmlLedColor_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

// 4.20 Event Handling Methods
nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long *eventTypes)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedEventTypes);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eventTypes, sizeof(unsigned long long)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceRegisterEvents);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetCreate);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(set, sizeof(nvmlEventSet_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetFree);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetWait_v2);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_write(&timeoutms, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(nvmlEventData_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return<nvmlReturn_t>(request_id, NVML_ERROR_GPU_IS_LOST);
}

CUresult cuDriverGetVersion(int *driverVersion)
{
    
    int request_id = rpc_start_request(RPC_cuDriverGetVersion);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(driverVersion, sizeof(int)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut)
{
    
    int request_id = rpc_start_request(RPC_cuLinkCreate_v2);
    if (request_id < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(optionValues, numOptions * sizeof(void *)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stateOut, sizeof(CUlinkState)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    
    int request_id = rpc_start_request(RPC_cuLinkAddData_v2);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_write(&type, sizeof(CUjitInputType)) < 0 ||
        rpc_write(&size, sizeof(size_t)) < 0 ||
        rpc_write(name, strlen(name) + 1) < 0 ||
        rpc_write(data, size) < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(optionValues, numOptions * sizeof(void *)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut)
{
    
    int request_id = rpc_start_request(RPC_cuLinkComplete);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cubinOut, sizeof(void *)) < 0 ||
        rpc_read(sizeOut, sizeof(size_t)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
    
    int request_id = rpc_start_request(RPC_cuModuleLoadData);
    if (request_id < 0 ||
        rpc_write(image, sizeof(image)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(module, sizeof(CUmodule)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuModuleUnload(CUmodule hmod)
{
    
    int request_id = rpc_start_request(RPC_cuModuleUnload);
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuGetErrorString(CUresult error, const char **pStr)
{
    
    int request_id = rpc_start_request(RPC_cuGetErrorString);
    if (request_id < 0 ||
        rpc_write(&error, sizeof(CUresult)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pStr, sizeof(const char *)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLinkDestroy(CUlinkState state)
{
    
    int request_id = rpc_start_request(RPC_cuLinkDestroy);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    
    int request_id = rpc_start_request(RPC_cuModuleGetFunction);
    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(name, strlen(name) + 1) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hfunc, sizeof(CUfunction)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
    
    int request_id = rpc_start_request(RPC_cuFuncSetAttribute);
    if (request_id < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_write(&attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(&value, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
    
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
        rpc_write(kernelParams, sizeof(void *)) < 0 ||
        rpc_write(extra, sizeof(void *)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuGetErrorName(CUresult error, const char **pStr)
{
    
    int request_id = rpc_start_request(RPC_cuGetErrorName);
    if (request_id < 0 ||
        rpc_write(&error, sizeof(CUresult)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pStr, sizeof(const char *)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
    
    int request_id = rpc_start_request(RPC_cuModuleLoadFatBinary);
    if (request_id < 0 ||
        rpc_write(fatCubin, sizeof(fatCubin)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(module, sizeof(CUmodule)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    
    int request_id = rpc_start_request(RPC_cuModuleLoadDataEx);
    if (request_id < 0 ||
        rpc_write(image, sizeof(image)) < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(optionValues, numOptions * sizeof(void *)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(module, sizeof(CUmodule)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    
    int request_id = rpc_start_request(RPC_cuLinkAddFile_v2);
    if (request_id < 0 ||
        rpc_write(&state, sizeof(CUlinkState)) < 0 ||
        rpc_write(&type, sizeof(CUjitInputType)) < 0 ||
        rpc_write(path, strlen(path) + 1) < 0 ||
        rpc_write(&numOptions, sizeof(unsigned int)) < 0 ||
        rpc_write(options, numOptions * sizeof(CUjit_option)) < 0 ||
        rpc_write(optionValues, numOptions * sizeof(void *)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuInit(unsigned int Flags)
{
    
    int request_id = rpc_start_request(RPC_cuInit);
    if (request_id < 0 ||
        rpc_write(&Flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
    
    int request_id = rpc_start_request(RPC_cuFuncGetAttribute);
    if (request_id < 0 ||
        rpc_write(&attrib, sizeof(CUfunction_attribute)) < 0 ||
        rpc_write(&hfunc, sizeof(CUfunction)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pi, sizeof(int)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuCtxPushCurrent(CUcontext ctx)
{
    
    int request_id = rpc_start_request(RPC_cuCtxPushCurrent);
    if (request_id < 0 ||
        rpc_write(&ctx, sizeof(CUcontext)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuCtxPopCurrent(CUcontext *pctx)
{
    
    int request_id = rpc_start_request(RPC_cuCtxPopCurrent);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuCtxGetDevice(CUdevice *device)
{
    
    int request_id = rpc_start_request(RPC_cuCtxGetDevice);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(CUdevice)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    
    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxRetain);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pctx, sizeof(CUcontext)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
{
    
    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxRelease);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuDevicePrimaryCtxReset(CUdevice dev)
{
    
    int request_id = rpc_start_request(RPC_cuDevicePrimaryCtxReset);
    if (request_id < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
    
    int request_id = rpc_start_request(RPC_cuDeviceGet);
    if (request_id < 0 ||
        rpc_write(&ordinal, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(CUdevice)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    
    int request_id = rpc_start_request(RPC_cuDeviceGetAttribute);
    if (request_id < 0 ||
        rpc_write(&attrib, sizeof(CUdevice_attribute)) < 0 ||
        rpc_write(&dev, sizeof(CUdevice)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pi, sizeof(int)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuStreamSynchronize(CUstream hStream)
{
    
    int request_id = rpc_start_request(RPC_cuStreamSynchronize);
    if (request_id < 0 ||
        rpc_write(&hStream, sizeof(CUstream)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize)
{
    
    int request_id = rpc_start_request(RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor);
    if (request_id < 0 ||
        rpc_write(&func, sizeof(CUfunction)) < 0 ||
        rpc_write(&blockSize, sizeof(int)) < 0 ||
        rpc_write(&dynamicSMemSize, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numBlocks, sizeof(int)) < 0)
        return CUDA_ERROR_UNKNOWN;
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra)
{
    
    // Start the RPC request for cuLaunchKernelEx
    int request_id = rpc_start_request(RPC_cuLaunchKernelEx);

    // Error handling for request initiation
    if (request_id < 0)
        return CUDA_ERROR_UNKNOWN;

    // Write config to the request
    if (rpc_write(config, sizeof(CUlaunchConfig)) < 0)
        return CUDA_ERROR_UNKNOWN;

    // Write CUfunction f to the request
    if (rpc_write(&f, sizeof(CUfunction)) < 0)
        return CUDA_ERROR_UNKNOWN;

    // Write kernel parameters to the request
    size_t kernelParamsSize = sizeof(void *) * config->gridDimX; // Adjust to the appropriate size as per your logic
    if (rpc_write(kernelParams, kernelParamsSize) < 0)
        return CUDA_ERROR_UNKNOWN;

    // Write extra parameters to the request
    size_t extraSize = sizeof(void *) * config->gridDimX; // Adjust to the appropriate size as per your logic
    if (rpc_write(extra, extraSize) < 0)
        return CUDA_ERROR_UNKNOWN;

    // Wait for the response
    if (rpc_wait_for_response(request_id) < 0)
        return CUDA_ERROR_UNKNOWN;

    // Return the result from the response
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    int request_id = rpc_start_request(RPC_cuMemcpyDtoH_v2);

    if (request_id < 0 ||
        rpc_write(dstHost, sizeof(void*)) < 0 ||
        rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
        rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dstHost, ByteCount) < 0)
    {
        return CUDA_ERROR_UNKNOWN;
    }

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    int request_id = rpc_start_request(RPC_cuModuleGetGlobal_v2);

    if (request_id < 0 ||
        rpc_write(&hmod, sizeof(CUmodule)) < 0 ||
        rpc_write(name, strlen(name) + 1) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_read(bytes, sizeof(size_t)) < 0)
    {
        return CUDA_ERROR_UNKNOWN;
    }

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuGetProcAddress_v2_handler(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
{
    // Open RPC client if not already opened
    if (open_rpc_client() < 0)
        return CUDA_ERROR_UNKNOWN;

    // Start the RPC request for cuGetProcAddress
    int request_id = rpc_start_request(RPC_cuGetProcAddress_v2);
    if (request_id < 0)
        return CUDA_ERROR_UNKNOWN;

    int symbol_length = strlen(symbol) + 1;

    void *placeholder = nullptr;
    if (rpc_write(&symbol_length, sizeof(int)) < 0 ||
        rpc_write(symbol, symbol_length) < 0 ||
        rpc_write(&cudaVersion, sizeof(int)) < 0 ||
        rpc_write(&placeholder, sizeof(void *)) < 0 ||  // Write a placeholder instead of pfn
        rpc_write(&flags, sizeof(cuuint64_t)) < 0) {
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server response
    if (rpc_wait_for_response(request_id) < 0)
    {
        return CUDA_ERROR_UNKNOWN;
    }

    //  // Read the function pointer from the response
    if (rpc_read(pfn, sizeof(void *)) < 0 || rpc_read(symbolStatus, sizeof(CUdriverProcAddressQueryResult)) < 0)
    {
        return CUDA_ERROR_UNKNOWN;
    }

    std::cout << "call complete: " << symbolStatus << std::endl;

    // Retrieve and return the result from the response
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

std::unordered_map<std::string, void *> functionMap;

void initializeFunctionMap()
{
    
    // simple cache check to make sure we only init handlers on the first run
    // attach all handlers to our function map
    functionMap["nvmlInitWithFlags"] = (void *)nvmlInitWithFlags;
    functionMap["nvmlInit_v2"] = (void *)nvmlInit_v2;
    functionMap["nvmlShutdown"] = (void *)nvmlShutdown;
    functionMap["nvmlSystemGetDriverVersion"] =
        (void *)nvmlSystemGetDriverVersion;
    functionMap["nvmlSystemGetHicVersion"] = (void *)nvmlSystemGetHicVersion;
    functionMap["nvmlSystemGetNVMLVersion"] = (void *)nvmlSystemGetNVMLVersion;
    functionMap["nvmlSystemGetProcessName"] = (void *)nvmlSystemGetProcessName;
    functionMap["nvmlSystemGetTopologyGpuSet"] =
        (void *)nvmlSystemGetTopologyGpuSet;
    functionMap["nvmlUnitGetCount"] = (void *)nvmlUnitGetCount;
    functionMap["nvmlUnitGetDevices"] = (void *)nvmlUnitGetDevices;
    functionMap["nvmlUnitGetFanSpeedInfo"] = (void *)nvmlUnitGetFanSpeedInfo;
    functionMap["nvmlUnitGetHandleByIndex"] = (void *)nvmlUnitGetHandleByIndex;
    functionMap["nvmlUnitGetLedState"] = (void *)nvmlUnitGetLedState;
    functionMap["nvmlUnitGetPsuInfo"] = (void *)nvmlUnitGetPsuInfo;
    functionMap["nvmlUnitGetTemperature"] = (void *)nvmlUnitGetTemperature;
    functionMap["nvmlUnitGetUnitInfo"] = (void *)nvmlUnitGetUnitInfo;

    // 4.16 Device Queries
    functionMap["nvmlDeviceGetClkMonStatus"] = (void *)nvmlDeviceGetClkMonStatus;
    functionMap["nvmlDeviceGetClock"] = (void *)nvmlDeviceGetClock;
    functionMap["nvmlDeviceGetClockInfo"] = (void *)nvmlDeviceGetClockInfo;
    functionMap["nvmlDeviceGetComputeMode"] = (void *)nvmlDeviceGetComputeMode;
    functionMap["nvmlDeviceGetCount_v2"] = (void *)nvmlDeviceGetCount_v2;
    functionMap["nvmlDeviceGetDisplayActive"] = (void *)nvmlDeviceGetDisplayActive;
    functionMap["nvmlDeviceGetDisplayMode"] = (void *)nvmlDeviceGetDisplayMode;
    functionMap["nvmlDeviceGetDriverModel_v2"] = (void *)nvmlDeviceGetDriverModel_v2;
    functionMap["nvmlDeviceGetDynamicPstatesInfo"] = (void *)nvmlDeviceGetDynamicPstatesInfo;
    functionMap["nvmlDeviceGetEccMode"] = (void *)nvmlDeviceGetEccMode;
    functionMap["nvmlDeviceGetEncoderCapacity"] = (void *)nvmlDeviceGetEncoderCapacity;
    functionMap["nvmlDeviceGetEncoderSessions"] = (void *)nvmlDeviceGetEncoderSessions;
    functionMap["nvmlDeviceGetEncoderStats"] = (void *)nvmlDeviceGetEncoderStats;
    functionMap["nvmlDeviceGetEncoderUtilization"] = (void *)nvmlDeviceGetEncoderUtilization;
    functionMap["nvmlDeviceGetEnforcedPowerLimit"] = (void *)nvmlDeviceGetEnforcedPowerLimit;
    functionMap["nvmlDeviceGetFanSpeed"] = (void *)nvmlDeviceGetFanSpeed;
    functionMap["nvmlDeviceGetHandleByIndex_v2"] =
        (void *)nvmlDeviceGetHandleByIndex_v2;
    functionMap["nvmlDeviceGetIndex"] = (void *)nvmlDeviceGetIndex;
    functionMap["nvmlDeviceGetMemoryInfo_v2"] =
        (void *)nvmlDeviceGetMemoryInfo_v2;
    functionMap["nvmlDeviceGetName"] = (void *)nvmlDeviceGetName;
    functionMap["nvmlDeviceGetPciInfo_v3"] = (void *)nvmlDeviceGetPciInfo_v3;
    functionMap["nvmlDeviceGetPcieSpeed"] = (void *)nvmlDeviceGetPcieSpeed;
    functionMap["nvmlDeviceGetPcieThroughput"] = (void *)nvmlDeviceGetPcieThroughput;
    functionMap["nvmlDeviceGetPerformanceState"] = (void *)nvmlDeviceGetPerformanceState;
    functionMap["nvmlDeviceGetPersistenceMode"] =
        (void *)nvmlDeviceGetPersistenceMode;
    functionMap["nvmlDeviceGetPowerSource"] = (void *)nvmlDeviceGetPowerSource;
    functionMap["nvmlDeviceGetPowerState"] = (void *)nvmlDeviceGetPowerState;
    functionMap["nvmlDeviceGetPowerUsage"] = (void *)nvmlDeviceGetPowerUsage;
    functionMap["nvmlDeviceGetSupportedPerformanceStates"] =
        (void *)nvmlDeviceGetSupportedPerformanceStates;
    functionMap["nvmlDeviceGetTargetFanSpeed"] = (void *)nvmlDeviceGetTargetFanSpeed;
    functionMap["nvmlDeviceGetTemperature"] = (void *)nvmlDeviceGetTemperature;
    functionMap["nvmlDeviceGetTemperatureThreshold"] = (void *)nvmlDeviceGetTemperatureThreshold;
    functionMap["nvmlDeviceGetThermalSettings"] = (void *)nvmlDeviceGetThermalSettings;
    functionMap["nvmlDeviceGetTopologyCommonAncestor"] = (void *)nvmlDeviceGetTopologyCommonAncestor;
    functionMap["nvmlDeviceGetTopologyNearestGpus"] = (void *)nvmlDeviceGetTopologyNearestGpus;
    functionMap["nvmlDeviceGetTotalEnergyConsumption"] = (void *)nvmlDeviceGetTotalEnergyConsumption;
    functionMap["nvmlDeviceGetUUID"] = (void *)nvmlDeviceGetUUID;
    functionMap["nvmlDeviceGetUtilizationRates"] = (void *)nvmlDeviceGetUtilizationRates;
    functionMap["nvmlDeviceValidateInforom"] = (void *)nvmlDeviceValidateInforom;

    // 4.17 Unit Commands
    functionMap["nvmlUnitSetLedState"] = (void *)nvmlUnitSetLedState;

    // 4.20 Event Handling Methods
    functionMap["nvmlDeviceGetSupportedEventTypes"] = (void *)nvmlDeviceGetSupportedEventTypes;
    functionMap["nvmlDeviceRegisterEvents"] = (void *)nvmlDeviceRegisterEvents;
    functionMap["nvmlEventSetCreate"] = (void *)nvmlEventSetCreate;
    functionMap["nvmlEventSetFree"] = (void *)nvmlEventSetFree;
    functionMap["nvmlEventSetWait_v2"] = (void *)nvmlEventSetWait_v2;

    // cuda
    functionMap["cuDriverGetVersion"] = (void *)cuDriverGetVersion;
    functionMap["cuLinkCreate_v2"] = (void *)cuLinkCreate_v2;
    functionMap["cuLinkAddData_v2"] = (void *)cuLinkAddData_v2;
    functionMap["cuLinkComplete"] = (void *)cuLinkComplete;
    functionMap["cuModuleLoadData"] = (void *)cuModuleLoadData;
    functionMap["cuModuleUnload"] = (void *)cuModuleUnload;
    functionMap["cuGetErrorString"] = (void *)cuGetErrorString;
    functionMap["cuLinkDestroy"] = (void *)cuLinkDestroy;
    functionMap["cuModuleGetFunction"] = (void *)cuModuleGetFunction;
    functionMap["cuFuncSetAttribute"] = (void *)cuFuncSetAttribute;
    functionMap["cuLaunchKernel"] = (void *)cuLaunchKernel;
    functionMap["cuGetErrorName"] = (void *)cuGetErrorName;
    functionMap["cuModuleLoadFatBinary"] = (void *)cuModuleLoadFatBinary;
    functionMap["cuModuleLoadDataEx"] = (void *)cuModuleLoadDataEx;
    functionMap["cuLinkAddFile_v2"] = (void *)cuLinkAddFile_v2;
    functionMap["cuInit"] = (void *)cuInit;
    functionMap["cuFuncGetAttribute"] = (void *)cuFuncGetAttribute;
    functionMap["cuCtxPushCurrent"] = (void *)cuCtxPushCurrent;
    functionMap["cuCtxPopCurrent"] = (void *)cuCtxPopCurrent;
    functionMap["cuCtxGetDevice"] = (void *)cuCtxGetDevice;
    functionMap["cuDevicePrimaryCtxRetain"] = (void *)cuDevicePrimaryCtxRetain;
    functionMap["cuDevicePrimaryCtxRelease"] = (void *)cuDevicePrimaryCtxRelease;
    functionMap["cuDevicePrimaryCtxReset"] = (void *)cuDevicePrimaryCtxReset;
    functionMap["cuDeviceGet"] = (void *)cuDeviceGet;
    functionMap["cuDeviceGetAttribute"] = (void *)cuDeviceGetAttribute;
    functionMap["cuStreamSynchronize"] = (void *)cuStreamSynchronize;
    functionMap["cuOccupancyMaxActiveBlocksPerMultiprocessor"] = (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor;
    functionMap["cuLaunchKernelEx"] = (void *)cuLaunchKernelEx;
    functionMap["cuMemcpyDtoH_v2"] = (void *)cuMemcpyDtoH_v2;
    functionMap["cuModuleGetGlobal_v2"] = (void *)cuModuleGetGlobal_v2;
    functionMap["cuGetProcAddress_v2"] = (void *)cuGetProcAddress_v2_handler;
}

// Lookup function similar to dlsym
void *getFunctionByName(const char *name)
{
    auto it = functionMap.find(name);
    if (it != functionMap.end()) {
        std::cout << "Function found: " << it->second << std::endl;
        return it->second;
    }
        
    return nullptr;
}

void *dlsym(void *handle, const char *name) __THROW
{
    initializeFunctionMap();  // Initialize the function map

    void *func = getFunctionByName(name);  // Lookup function by name

    if (func != nullptr) {
        std::cout << "[dlsym] Function address from functionMap: " << func << std::endl;
        return func;
    }

    // Real dlsym lookup
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL) {
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    std::cout << "[dlsym] Falling back to real_dlsym for name: " << name << std::endl;
    return real_dlsym(handle, name);
}
