
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

int rpc_start_request(const unsigned int op) {
    static int next_request_id = 1; // Initialized once and retains value across function calls

    // Ensure socket is open
    if (sockfd < 0) {
        std::cerr << "Socket not open" << std::endl;
        return -1;
    }

    // Lock the mutex for atomic operation
    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++; // Assign and then increment

    // Write the request ID and operation code
    if (write(sockfd, &request_id, sizeof(int)) < 0) {
        std::cerr << "Failed to write request_id. Error: " << strerror(errno) << std::endl;
        pthread_mutex_unlock(&mutex);
        return -1;
    }

    if (write(sockfd, &op, sizeof(unsigned int)) < 0) {
        std::cerr << "Failed to write operation code. Error: " << strerror(errno) << std::endl;
        pthread_mutex_unlock(&mutex);
        return -1;
    }

    pthread_mutex_unlock(&mutex);

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
    if (data == nullptr) {
        // temp buffer to discard data
        char tempBuffer[256];
        while (size > 0) {
            ssize_t bytesRead = read(sockfd, tempBuffer, std::min(size, sizeof(tempBuffer)));
            if (bytesRead < 0) {
                pthread_mutex_unlock(&mutex);
                return -1; // error if reading fails
            }
            size -= bytesRead;
        }
    } else if (read(sockfd, data, size) < 0)
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

CUresult cuInit(unsigned int Flags) {
    // Open RPC client if not already opened
    if (open_rpc_client() < 0)
        return CUDA_ERROR_UNKNOWN;

    // Start the RPC request for cuInit
    int request_id = rpc_start_request(RPC_cuInit);
    if (request_id < 0) {
        std::cerr << "Failed to start cuInit request" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Write the flags to the server
    if (rpc_write(&Flags, sizeof(unsigned int)) < 0) {
        std::cerr << "Failed to write flags to server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server response
    if (rpc_wait_for_response(request_id) < 0) {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Read the result code from the server
    CUresult result;
    if (rpc_read(&result, sizeof(CUresult)) < 0) {
        std::cerr << "Failed to read result code from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Log the successful initialization
    if (result == CUDA_SUCCESS) {
        std::cout << "cuInit successful, Flags: " << Flags << std::endl;
    } else {
        const char *errorStr = nullptr;
        cuGetErrorString(result, &errorStr);
        std::cerr << "cuInit failed with error code: " << result 
                  << " (" << (errorStr ? errorStr : "Unknown error") << ")" << std::endl;
    }

    // Return the result received from the server
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

CUresult cuDeviceGetShim(CUdevice *device, int ordinal) {
    std::cout << "Client: calling cuDeviceGetShim" << std::endl;

    // Check if the device pointer is valid
    if (device == nullptr) {
        std::cerr << "Invalid device pointer provided." << std::endl;
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Ensure the socket is connected
    if (sockfd < 0) {
        std::cerr << "Socket not connected." << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Start the request with the specific RPC operation code for cuDeviceGet
    int request_id = rpc_start_request(RPC_cuDeviceGet);
    if (request_id < 0) {
        std::cerr << "Failed to start request for cuDeviceGet" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Write the ordinal value to the server
    if (rpc_write(&ordinal, sizeof(int)) < 0) {
        std::cerr << "Failed to write ordinal to server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server's response
    if (rpc_wait_for_response(request_id) < 0) {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Read the result code from the server
    CUresult result;
    if (rpc_read(&result, sizeof(CUresult)) < 0) {
        std::cerr << "Failed to read result code from server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Check if the cuDeviceGet call was successful
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet call failed on the server. Error code: " << result << std::endl;
        return result;
    }

    // Read the device handle from the server
    if (rpc_read(device, sizeof(CUdevice)) < 0) {
        std::cerr << "Failed to read device handle from server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    std::cout << "Client: Received device handle from server: " << *device << std::endl;

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuDeviceGetCountShim(int *deviceCount) {
    std::cout << "Client: calling cuDeviceGetCountShim" << std::endl;

    // Check if the socket is connected
    if (sockfd < 0) {
        std::cerr << "Socket not connected." << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Start the RPC request with the appropriate opcode for cuDeviceGetCount
    int request_id = rpc_start_request(RPC_cuDeviceGetCount);
    if (request_id < 0) {
        std::cerr << "Failed to start request for cuDeviceGetCount" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server's response
    if (rpc_wait_for_response(request_id) < 0) {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Read the result code from the server
    CUresult result;
    ssize_t bytes_read = rpc_read(&result, sizeof(CUresult));
    if (bytes_read < 0) {
        std::cerr << "Failed to read result from server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Check if the cuDeviceGetCount call was successful
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGetCount call failed on the server. Error code: " << result << std::endl;
        return result;
    }

    // Read the device count from the server
    bytes_read = rpc_read(deviceCount, sizeof(int));
    if (bytes_read < 0) {
        std::cerr << "Failed to read device count from server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    std::cout << "Client: Received device count from server: " << *deviceCount << std::endl;

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}


void cuDeviceGetNameShim() {
    std::cout << "calling cuDeviceGetNameShim" << std::endl;
}

void cuDeviceTotalMemShim() {
    std::cout << "calling cuDeviceTotalMemShim" << std::endl;
}

void cuDeviceGetAttributeShim() {
    std::cout << "calling cuDeviceGetAttributeShim" << std::endl;
}

void cuDeviceGetP2PAttributeShim() {
    std::cout << "calling cuDeviceGetP2PAttributeShim" << std::endl;
}

void cuDeviceGetByPCIBusIdShim() {
    std::cout << "calling cuDeviceGetByPCIBusIdShim" << std::endl;
}

void cuDeviceGetPCIBusIdShim() {
    std::cout << "calling cuDeviceGetPCIBusIdShim" << std::endl;
}

void cuDeviceGetUuidShim() {
    std::cout << "calling cuDeviceGetUuidShim" << std::endl;
}

void cuDeviceGetTexture1DLinearMaxWidthShim() {
    std::cout << "calling cuDeviceGetTexture1DLinearMaxWidthShim" << std::endl;
}

void cuDeviceGetDefaultMemPoolShim() {
    std::cout << "calling cuDeviceGetDefaultMemPoolShim" << std::endl;
}

void cuDeviceSetMemPoolShim() {
    std::cout << "calling cuDeviceSetMemPoolShim" << std::endl;
}

void cuDeviceGetMemPoolShim() {
    std::cout << "calling cuDeviceGetMemPoolShim" << std::endl;
}

void cuFlushGPUDirectRDMAWritesShim() {
    std::cout << "calling cuFlushGPUDirectRDMAWritesShim" << std::endl;
}

void cuDevicePrimaryCtxRetainShim() {
    std::cout << "calling cuDevicePrimaryCtxRetainShim" << std::endl;
}

void cuDevicePrimaryCtxReleaseShim() {
    std::cout << "calling cuDevicePrimaryCtxReleaseShim" << std::endl;
}

void cuDevicePrimaryCtxSetFlagsShim() {
    std::cout << "calling cuDevicePrimaryCtxSetFlagsShim" << std::endl;
}

void cuDevicePrimaryCtxGetStateShim() {
    std::cout << "calling cuDevicePrimaryCtxGetStateShim" << std::endl;
}

void cuDevicePrimaryCtxResetShim() {
    std::cout << "calling cuDevicePrimaryCtxResetShim" << std::endl;
}

void cuCtxCreateShim() {
    std::cout << "calling cuCtxCreateShim" << std::endl;
}

void cuCtxGetFlagsShim() {
    std::cout << "calling cuCtxGetFlagsShim" << std::endl;
}

void cuCtxSetCurrentShim() {
    std::cout << "calling cuCtxSetCurrentShim" << std::endl;
}

void cuCtxGetCurrentShim() {
    std::cout << "calling cuCtxGetCurrentShim" << std::endl;
}

void cuCtxDetachShim() {
    std::cout << "calling cuCtxDetachShim" << std::endl;
}

void cuCtxGetApiVersionShim() {
    std::cout << "calling cuCtxGetApiVersionShim" << std::endl;
}

void cuCtxGetDeviceShim() {
    std::cout << "calling cuCtxGetDeviceShim" << std::endl;
}

void cuCtxGetLimitShim() {
    std::cout << "calling cuCtxGetLimitShim" << std::endl;
}

void cuCtxSetLimitShim() {
    std::cout << "calling cuCtxSetLimitShim" << std::endl;
}

void cuCtxGetCacheConfigShim() {
    std::cout << "calling cuCtxGetCacheConfigShim" << std::endl;
}

void cuCtxSetCacheConfigShim() {
    std::cout << "calling cuCtxSetCacheConfigShim" << std::endl;
}

void cuCtxGetSharedMemConfigShim() {
    std::cout << "calling cuCtxGetSharedMemConfigShim" << std::endl;
}

void cuCtxGetStreamPriorityRangeShim() {
    std::cout << "calling cuCtxGetStreamPriorityRangeShim" << std::endl;
}

void cuCtxSetSharedMemConfigShim() {
    std::cout << "calling cuCtxSetSharedMemConfigShim" << std::endl;
}

void cuCtxSynchronizeShim() {
    std::cout << "calling cuCtxSynchronizeShim" << std::endl;
}

void cuCtxResetPersistingL2CacheShim() {
    std::cout << "calling cuCtxResetPersistingL2CacheShim" << std::endl;
}

void cuCtxPopCurrentShim() {
    std::cout << "calling cuCtxPopCurrentShim" << std::endl;
}

void cuCtxPushCurrentShim() {
    std::cout << "calling cuCtxPushCurrentShim" << std::endl;
}

void cuModuleLoadShim() {
    std::cout << "calling cuModuleLoadShim" << std::endl;
}

void cuModuleLoadDataShim() {
    std::cout << "calling cuModuleLoadDataShim" << std::endl;
}

void cuModuleLoadFatBinaryShim() {
    std::cout << "calling cuModuleLoadFatBinaryShim" << std::endl;
}

void cuModuleUnloadShim() {
    std::cout << "calling cuModuleUnloadShim" << std::endl;
}

void cuModuleGetFunctionShim() {
    std::cout << "calling cuModuleGetFunctionShim" << std::endl;
}

void cuModuleGetGlobalShim() {
    std::cout << "calling cuModuleGetGlobalShim" << std::endl;
}

void cuModuleGetTexRefShim() {
    std::cout << "calling cuModuleGetTexRefShim" << std::endl;
}

void cuModuleGetSurfRefShim() {
    std::cout << "calling cuModuleGetSurfRefShim" << std::endl;
}

CUresult cuModuleGetLoadingModeShim(CUmoduleLoadingMode *mode) {
    std::cout << "Client: calling cuModuleGetLoadingModeShim" << std::endl;

    if (sockfd < 0) {
        std::cerr << "Socket not connected." << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Prepare the request ID for the server
    int request_id = rpc_start_request(RPC_cuModuleGetLoadingMode);
    if (request_id < 0) {
        std::cerr << "Failed to start request for cuModuleGetLoadingMode" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server's response
    if (rpc_wait_for_response(request_id) < 0) {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Read the result code from the server
    CUresult result;
    ssize_t bytes_read = rpc_read(&result, sizeof(CUresult));
    if (bytes_read < 0) {
        std::cerr << "Failed to read result from server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Check if the cuModuleGetLoadingMode call was successful
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuModuleGetLoadingMode call failed on the server. Error code: " << result << std::endl;
        return result;
    }

    // Read the loading mode from the server
    bytes_read = rpc_read(mode, sizeof(CUmoduleLoadingMode));
    if (bytes_read < 0) {
        std::cerr << "Failed to read loading mode from server. Error: " << strerror(errno) << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    std::cout << "Client: Received loading mode from server: " << *mode << std::endl;

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

void cuLibraryLoadDataShim() {
    std::cout << "calling cuLibraryLoadDataShim" << std::endl;
}

void cuLibraryLoadFromFileShim() {
    std::cout << "calling cuLibraryLoadFromFileShim" << std::endl;
}

void cuLibraryUnloadShim() {
    std::cout << "calling cuLibraryUnloadShim" << std::endl;
}

void cuLibraryGetKernelShim() {
    std::cout << "calling cuLibraryGetKernelShim" << std::endl;
}

void cuLibraryGetModuleShim() {
    std::cout << "calling cuLibraryGetModuleShim" << std::endl;
}

void cuKernelGetFunctionShim() {
    std::cout << "calling cuKernelGetFunctionShim" << std::endl;
}

void cuLibraryGetGlobalShim() {
    std::cout << "calling cuLibraryGetGlobalShim" << std::endl;
}

void cuLibraryGetManagedShim() {
    std::cout << "calling cuLibraryGetManagedShim" << std::endl;
}

void cuKernelGetAttributeShim() {
    std::cout << "calling cuKernelGetAttributeShim" << std::endl;
}

void cuKernelSetAttributeShim() {
    std::cout << "calling cuKernelSetAttributeShim" << std::endl;
}

void cuKernelSetCacheConfigShim() {
    std::cout << "calling cuKernelSetCacheConfigShim" << std::endl;
}

void cuLinkCreateShim() {
    std::cout << "calling cuLinkCreateShim" << std::endl;
}

void cuLinkAddDataShim() {
    std::cout << "calling cuLinkAddDataShim" << std::endl;
}

void cuLinkAddFileShim() {
    std::cout << "calling cuLinkAddFileShim" << std::endl;
}

void cuLinkCompleteShim() {
    std::cout << "calling cuLinkCompleteShim" << std::endl;
}

void cuLinkDestroyShim() {
    std::cout << "calling cuLinkDestroyShim" << std::endl;
}

void cuMemGetInfoShim() {
    std::cout << "calling cuMemGetInfoShim" << std::endl;
}

void cuMemAllocManagedShim() {
    std::cout << "calling cuMemAllocManagedShim" << std::endl;
}

void cuMemAllocShim() {
    std::cout << "calling cuMemAllocShim" << std::endl;
}

void cuMemAllocPitchShim() {
    std::cout << "calling cuMemAllocPitchShim" << std::endl;
}

void cuMemFreeShim() {
    std::cout << "calling cuMemFreeShim" << std::endl;
}

void cuMemGetAddressRangeShim() {
    std::cout << "calling cuMemGetAddressRangeShim" << std::endl;
}

void cuMemFreeHostShim() {
    std::cout << "calling cuMemFreeHostShim" << std::endl;
}

void cuMemHostAllocShim() {
    std::cout << "calling cuMemHostAllocShim" << std::endl;
}

void cuMemHostGetDevicePointerShim() {
    std::cout << "calling cuMemHostGetDevicePointerShim" << std::endl;
}

void cuMemHostGetFlagsShim() {
    std::cout << "calling cuMemHostGetFlagsShim" << std::endl;
}

void cuMemHostRegisterShim() {
    std::cout << "calling cuMemHostRegisterShim" << std::endl;
}

void cuMemHostUnregisterShim() {
    std::cout << "calling cuMemHostUnregisterShim" << std::endl;
}

void cuPointerGetAttributeShim() {
    std::cout << "calling cuPointerGetAttributeShim" << std::endl;
}

void cuPointerGetAttributesShim() {
    std::cout << "calling cuPointerGetAttributesShim" << std::endl;
}

void cuMemAllocAsyncShim() {
    std::cout << "calling cuMemAllocAsyncShim" << std::endl;
}

void cuMemAllocFromPoolAsyncShim() {
    std::cout << "calling cuMemAllocFromPoolAsyncShim" << std::endl;
}

void cuMemFreeAsyncShim() {
    std::cout << "calling cuMemFreeAsyncShim" << std::endl;
}

void cuMemPoolTrimToShim() {
    std::cout << "calling cuMemPoolTrimToShim" << std::endl;
}

void cuMemPoolSetAttributeShim() {
    std::cout << "calling cuMemPoolSetAttributeShim" << std::endl;
}

void cuMemPoolGetAttributeShim() {
    std::cout << "calling cuMemPoolGetAttributeShim" << std::endl;
}

void cuMemPoolSetAccessShim() {
    std::cout << "calling cuMemPoolSetAccessShim" << std::endl;
}

void cuMemPoolGetAccessShim() {
    std::cout << "calling cuMemPoolGetAccessShim" << std::endl;
}

void cuMemPoolCreateShim() {
    std::cout << "calling cuMemPoolCreateShim" << std::endl;
}

void cuMemPoolDestroyShim() {
    std::cout << "calling cuMemPoolDestroyShim" << std::endl;
}

void cuMemPoolExportToShareableHandleShim() {
    std::cout << "calling cuMemPoolExportToShareableHandleShim" << std::endl;
}

void cuMemPoolImportFromShareableHandleShim() {
    std::cout << "calling cuMemPoolImportFromShareableHandleShim" << std::endl;
}

void cuMemPoolExportPointerShim() {
    std::cout << "calling cuMemPoolExportPointerShim" << std::endl;
}

void cuMemPoolImportPointerShim() {
    std::cout << "calling cuMemPoolImportPointerShim" << std::endl;
}

void cuMemcpyShim() {
    std::cout << "calling cuMemcpyShim" << std::endl;
}

void cuMemcpyAsyncShim() {
    std::cout << "calling cuMemcpyAsyncShim" << std::endl;
}

void cuMemcpyPeerShim() {
    std::cout << "calling cuMemcpyPeerShim" << std::endl;
}

void cuMemcpyPeerAsyncShim() {
    std::cout << "calling cuMemcpyPeerAsyncShim" << std::endl;
}

void cuMemcpyHtoDShim() {
    std::cout << "calling cuMemcpyHtoDShim" << std::endl;
}

void cuMemcpyHtoDAsyncShim() {
    std::cout << "calling cuMemcpyHtoDAsyncShim" << std::endl;
}

void cuMemcpyDtoHShim() {
    std::cout << "calling cuMemcpyDtoHShim" << std::endl;
}

void cuMemcpyDtoHAsyncShim() {
    std::cout << "calling cuMemcpyDtoHAsyncShim" << std::endl;
}

void cuMemcpyDtoDShim() {
    std::cout << "calling cuMemcpyDtoDShim" << std::endl;
}

void cuMemcpyDtoDAsyncShim() {
    std::cout << "calling cuMemcpyDtoDAsyncShim" << std::endl;
}

void cuMemcpy2DUnalignedShim() {
    std::cout << "calling cuMemcpy2DUnalignedShim" << std::endl;
}

void cuMemcpy2DAsyncShim() {
    std::cout << "calling cuMemcpy2DAsyncShim" << std::endl;
}

void cuMemcpy3DShim() {
    std::cout << "calling cuMemcpy3DShim" << std::endl;
}

void cuMemcpy3DAsyncShim() {
    std::cout << "calling cuMemcpy3DAsyncShim" << std::endl;
}

void cuMemcpy3DPeerShim() {
    std::cout << "calling cuMemcpy3DPeerShim" << std::endl;
}

void cuMemcpy3DPeerAsyncShim() {
    std::cout << "calling cuMemcpy3DPeerAsyncShim" << std::endl;
}

void cuMemsetD8Shim() {
    std::cout << "calling cuMemsetD8Shim" << std::endl;
}

void cuMemsetD8AsyncShim() {
    std::cout << "calling cuMemsetD8AsyncShim" << std::endl;
}

void cuMemsetD2D8Shim() {
    std::cout << "calling cuMemsetD2D8Shim" << std::endl;
}

void cuMemsetD2D8AsyncShim() {
    std::cout << "calling cuMemsetD2D8AsyncShim" << std::endl;
}

void cuFuncSetCacheConfigShim() {
    std::cout << "calling cuFuncSetCacheConfigShim" << std::endl;
}

void cuFuncSetSharedMemConfigShim() {
    std::cout << "calling cuFuncSetSharedMemConfigShim" << std::endl;
}

void cuFuncGetAttributeShim() {
    std::cout << "calling cuFuncGetAttributeShim" << std::endl;
}

void cuFuncSetAttributeShim() {
    std::cout << "calling cuFuncSetAttributeShim" << std::endl;
}

void cuArrayCreateShim() {
    std::cout << "calling cuArrayCreateShim" << std::endl;
}

void cuArrayGetDescriptorShim() {
    std::cout << "calling cuArrayGetDescriptorShim" << std::endl;
}

void cuArrayGetSparsePropertiesShim() {
    std::cout << "calling cuArrayGetSparsePropertiesShim" << std::endl;
}

void cuArrayGetPlaneShim() {
    std::cout << "calling cuArrayGetPlaneShim" << std::endl;
}

void cuArray3DCreateShim() {
    std::cout << "calling cuArray3DCreateShim" << std::endl;
}

void cuArray3DGetDescriptorShim() {
    std::cout << "calling cuArray3DGetDescriptorShim" << std::endl;
}

void cuArrayDestroyShim() {
    std::cout << "calling cuArrayDestroyShim" << std::endl;
}

void cuMipmappedArrayCreateShim() {
    std::cout << "calling cuMipmappedArrayCreateShim" << std::endl;
}

void cuMipmappedArrayGetLevelShim() {
    std::cout << "calling cuMipmappedArrayGetLevelShim" << std::endl;
}

void cuMipmappedArrayGetSparsePropertiesShim() {
    std::cout << "calling cuMipmappedArrayGetSparsePropertiesShim" << std::endl;
}

void cuMipmappedArrayDestroyShim() {
    std::cout << "calling cuMipmappedArrayDestroyShim" << std::endl;
}

void cuArrayGetMemoryRequirementsShim() {
    std::cout << "calling cuArrayGetMemoryRequirementsShim" << std::endl;
}

void cuMipmappedArrayGetMemoryRequirementsShim() {
    std::cout << "calling cuMipmappedArrayGetMemoryRequirementsShim" << std::endl;
}

void cuTexObjectCreateShim() {
    std::cout << "calling cuTexObjectCreateShim" << std::endl;
}

void cuTexObjectDestroyShim() {
    std::cout << "calling cuTexObjectDestroyShim" << std::endl;
}

void cuTexObjectGetResourceDescShim() {
    std::cout << "calling cuTexObjectGetResourceDescShim" << std::endl;
}

void cuTexObjectGetTextureDescShim() {
    std::cout << "calling cuTexObjectGetTextureDescShim" << std::endl;
}

void cuTexObjectGetResourceViewDescShim() {
    std::cout << "calling cuTexObjectGetResourceViewDescShim" << std::endl;
}

void cuSurfObjectCreateShim() {
    std::cout << "calling cuSurfObjectCreateShim" << std::endl;
}

void cuSurfObjectDestroyShim() {
    std::cout << "calling cuSurfObjectDestroyShim" << std::endl;
}

void cuSurfObjectGetResourceDescShim() {
    std::cout << "calling cuSurfObjectGetResourceDescShim" << std::endl;
}

void cuImportExternalMemoryShim() {
    std::cout << "calling cuImportExternalMemoryShim" << std::endl;
}

void cuExternalMemoryGetMappedBufferShim() {
    std::cout << "calling cuExternalMemoryGetMappedBufferShim" << std::endl;
}

void cuExternalMemoryGetMappedMipmappedArrayShim() {
    std::cout << "calling cuExternalMemoryGetMappedMipmappedArrayShim" << std::endl;
}

void cuDestroyExternalMemoryShim() {
    std::cout << "calling cuDestroyExternalMemoryShim" << std::endl;
}

void cuImportExternalSemaphoreShim() {
    std::cout << "calling cuImportExternalSemaphoreShim" << std::endl;
}

void cuSignalExternalSemaphoresAsyncShim() {
    std::cout << "calling cuSignalExternalSemaphoresAsyncShim" << std::endl;
}

void cuWaitExternalSemaphoresAsyncShim() {
    std::cout << "calling cuWaitExternalSemaphoresAsyncShim" << std::endl;
}

void cuDestroyExternalSemaphoreShim() {
    std::cout << "calling cuDestroyExternalSemaphoreShim" << std::endl;
}

void cuDeviceGetNvSciSyncAttributesShim() {
    std::cout << "calling cuDeviceGetNvSciSyncAttributesShim" << std::endl;
}

void cuLaunchKernelShim() {
    std::cout << "calling cuLaunchKernelShim" << std::endl;
}

void cuLaunchCooperativeKernelShim() {
    std::cout << "calling cuLaunchCooperativeKernelShim" << std::endl;
}

void cuLaunchCooperativeKernelMultiDeviceShim() {
    std::cout << "calling cuLaunchCooperativeKernelMultiDeviceShim" << std::endl;
}

void cuLaunchHostFuncShim() {
    std::cout << "calling cuLaunchHostFuncShim" << std::endl;
}

void cuLaunchKernelExShim() {
    std::cout << "calling cuLaunchKernelExShim" << std::endl;
}

void cuEventCreateShim() {
    std::cout << "calling cuEventCreateShim" << std::endl;
}

void cuEventRecordShim() {
    std::cout << "calling cuEventRecordShim" << std::endl;
}

void cuEventRecordWithFlagsShim() {
    std::cout << "calling cuEventRecordWithFlagsShim" << std::endl;
}

void cuEventQueryShim() {
    std::cout << "calling cuEventQueryShim" << std::endl;
}

void cuEventSynchronizeShim() {
    std::cout << "calling cuEventSynchronizeShim" << std::endl;
}

void cuEventDestroyShim() {
    std::cout << "calling cuEventDestroyShim" << std::endl;
}

void cuEventElapsedTimeShim() {
    std::cout << "calling cuEventElapsedTimeShim" << std::endl;
}

void cuStreamWaitValue32Shim() {
    std::cout << "calling cuStreamWaitValue32Shim" << std::endl;
}

void cuStreamWriteValue32Shim() {
    std::cout << "calling cuStreamWriteValue32Shim" << std::endl;
}

void cuStreamWaitValue64Shim() {
    std::cout << "calling cuStreamWaitValue64Shim" << std::endl;
}

void cuStreamWriteValue64Shim() {
    std::cout << "calling cuStreamWriteValue64Shim" << std::endl;
}

void cuStreamBatchMemOpShim() {
    std::cout << "calling cuStreamBatchMemOpShim" << std::endl;
}

void cuStreamCreateShim() {
    std::cout << "calling cuStreamCreateShim" << std::endl;
}

void cuStreamCreateWithPriorityShim() {
    std::cout << "calling cuStreamCreateWithPriorityShim" << std::endl;
}

void cuStreamGetPriorityShim() {
    std::cout << "calling cuStreamGetPriorityShim" << std::endl;
}

void cuStreamGetFlagsShim() {
    std::cout << "calling cuStreamGetFlagsShim" << std::endl;
}

void cuStreamGetCtxShim() {
    std::cout << "calling cuStreamGetCtxShim" << std::endl;
}

void cuStreamGetIdShim() {
    std::cout << "calling cuStreamGetIdShim" << std::endl;
}

void cuStreamDestroyShim() {
    std::cout << "calling cuStreamDestroyShim" << std::endl;
}

void cuStreamWaitEventShim() {
    std::cout << "calling cuStreamWaitEventShim" << std::endl;
}

void cuStreamAddCallbackShim() {
    std::cout << "calling cuStreamAddCallbackShim" << std::endl;
}

void cuStreamSynchronizeShim() {
    std::cout << "calling cuStreamSynchronizeShim" << std::endl;
}

void cuStreamQueryShim() {
    std::cout << "calling cuStreamQueryShim" << std::endl;
}

void cuStreamAttachMemAsyncShim() {
    std::cout << "calling cuStreamAttachMemAsyncShim" << std::endl;
}

void cuStreamCopyAttributesShim() {
    std::cout << "calling cuStreamCopyAttributesShim" << std::endl;
}

void cuStreamGetAttributeShim() {
    std::cout << "calling cuStreamGetAttributeShim" << std::endl;
}

void cuStreamSetAttributeShim() {
    std::cout << "calling cuStreamSetAttributeShim" << std::endl;
}

void cuDeviceCanAccessPeerShim() {
    std::cout << "calling cuDeviceCanAccessPeerShim" << std::endl;
}

void cuCtxEnablePeerAccessShim() {
    std::cout << "calling cuCtxEnablePeerAccessShim" << std::endl;
}

void cuCtxDisablePeerAccessShim() {
    std::cout << "calling cuCtxDisablePeerAccessShim" << std::endl;
}

void cuIpcGetEventHandleShim() {
    std::cout << "calling cuIpcGetEventHandleShim" << std::endl;
}

void cuIpcOpenEventHandleShim() {
    std::cout << "calling cuIpcOpenEventHandleShim" << std::endl;
}

void cuIpcGetMemHandleShim() {
    std::cout << "calling cuIpcGetMemHandleShim" << std::endl;
}

void cuIpcOpenMemHandleShim() {
    std::cout << "calling cuIpcOpenMemHandleShim" << std::endl;
}

void cuIpcCloseMemHandleShim() {
    std::cout << "calling cuIpcCloseMemHandleShim" << std::endl;
}

void cuGLCtxCreateShim() {
    std::cout << "calling cuGLCtxCreateShim" << std::endl;
}

void cuGLInitShim() {
    std::cout << "calling cuGLInitShim" << std::endl;
}

void cuGLGetDevicesShim() {
    std::cout << "calling cuGLGetDevicesShim" << std::endl;
}

void cuGLRegisterBufferObjectShim() {
    std::cout << "calling cuGLRegisterBufferObjectShim" << std::endl;
}

void cuGLMapBufferObjectShim() {
    std::cout << "calling cuGLMapBufferObjectShim" << std::endl;
}

void cuGLMapBufferObjectAsyncShim() {
    std::cout << "calling cuGLMapBufferObjectAsyncShim" << std::endl;
}

void cuGLUnmapBufferObjectShim() {
    std::cout << "calling cuGLUnmapBufferObjectShim" << std::endl;
}

void cuGLUnmapBufferObjectAsyncShim() {
    std::cout << "calling cuGLUnmapBufferObjectAsyncShim" << std::endl;
}

void cuGLUnregisterBufferObjectShim() {
    std::cout << "calling cuGLUnregisterBufferObjectShim" << std::endl;
}

void cuGLSetBufferObjectMapFlagsShim() {
    std::cout << "calling cuGLSetBufferObjectMapFlagsShim" << std::endl;
}

void cuGraphicsGLRegisterImageShim() {
    std::cout << "calling cuGraphicsGLRegisterImageShim" << std::endl;
}

void cuGraphicsGLRegisterBufferShim() {
    std::cout << "calling cuGraphicsGLRegisterBufferShim" << std::endl;
}

void cuGraphicsEGLRegisterImageShim() {
    std::cout << "calling cuGraphicsEGLRegisterImageShim" << std::endl;
}

void cuEGLStreamConsumerConnectShim() {
    std::cout << "calling cuEGLStreamConsumerConnectShim" << std::endl;
}

void cuEGLStreamConsumerDisconnectShim() {
    std::cout << "calling cuEGLStreamConsumerDisconnectShim" << std::endl;
}

void cuEGLStreamConsumerAcquireFrameShim() {
    std::cout << "calling cuEGLStreamConsumerAcquireFrameShim" << std::endl;
}

void cuEGLStreamConsumerReleaseFrameShim() {
    std::cout << "calling cuEGLStreamConsumerReleaseFrameShim" << std::endl;
}

void cuEGLStreamProducerConnectShim() {
    std::cout << "calling cuEGLStreamProducerConnectShim" << std::endl;
}

void cuEGLStreamProducerDisconnectShim() {
    std::cout << "calling cuEGLStreamProducerDisconnectShim" << std::endl;
}

void cuEGLStreamProducerPresentFrameShim() {
    std::cout << "calling cuEGLStreamProducerPresentFrameShim" << std::endl;
}

void cuEGLStreamProducerReturnFrameShim() {
    std::cout << "calling cuEGLStreamProducerReturnFrameShim" << std::endl;
}

void cuGraphicsResourceGetMappedEglFrameShim() {
    std::cout << "calling cuGraphicsResourceGetMappedEglFrameShim" << std::endl;
}

void cuGraphicsUnregisterResourceShim() {
    std::cout << "calling cuGraphicsUnregisterResourceShim" << std::endl;
}

void cuGraphicsMapResourcesShim() {
    std::cout << "calling cuGraphicsMapResourcesShim" << std::endl;
}

void cuGraphicsUnmapResourcesShim() {
    std::cout << "calling cuGraphicsUnmapResourcesShim" << std::endl;
}

void cuGraphicsResourceSetMapFlagsShim() {
    std::cout << "calling cuGraphicsResourceSetMapFlagsShim" << std::endl;
}

void cuGraphicsSubResourceGetMappedArrayShim() {
    std::cout << "calling cuGraphicsSubResourceGetMappedArrayShim" << std::endl;
}

void cuGraphicsResourceGetMappedMipmappedArrayShim() {
    std::cout << "calling cuGraphicsResourceGetMappedMipmappedArrayShim" << std::endl;
}

void cuProfilerInitializeShim() {
    std::cout << "calling cuProfilerInitializeShim" << std::endl;
}

void cuProfilerStartShim() {
    std::cout << "calling cuProfilerStartShim" << std::endl;
}

void cuProfilerStopShim() {
    std::cout << "calling cuProfilerStopShim" << std::endl;
}

void cuVDPAUGetDeviceShim() {
    std::cout << "calling cuVDPAUGetDeviceShim" << std::endl;
}

void cuVDPAUCtxCreateShim() {
    std::cout << "calling cuVDPAUCtxCreateShim" << std::endl;
}

void cuGraphicsVDPAURegisterVideoSurfaceShim() {
    std::cout << "calling cuGraphicsVDPAURegisterVideoSurfaceShim" << std::endl;
}

void cuGraphicsVDPAURegisterOutputSurfaceShim() {
    std::cout << "calling cuGraphicsVDPAURegisterOutputSurfaceShim" << std::endl;
}

CUresult cuGetExportTableShim(void **ppExportTable, const CUuuid *pTableUuid) {
    if (sockfd < 0) {
        std::cerr << "Socket not connected." << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Start the request to the server
    int request_id = rpc_start_request(RPC_cuGetExportTable);
    if (request_id < 0) {
        std::cerr << "Failed to start request for cuGetExportTable" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Check if pTableUuid is valid
    if (pTableUuid == nullptr) {
        std::cerr << "Invalid UUID pointer provided to cuGetExportTableShim" << std::endl;
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Send the UUID to the server
    if (rpc_write(pTableUuid, sizeof(CUuuid)) < 0) {
        std::cerr << "Failed to write UUID to server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server response
    if (rpc_wait_for_response(request_id) < 0) {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Read the result code from the server
    CUresult result;
    if (rpc_read(&result, sizeof(CUresult)) < 0) {
        std::cerr << "Failed to read result from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    // Check if the cuGetExportTable call was successful
    if (result != CUDA_SUCCESS) {
        std::cerr << "cuGetExportTable call failed on the server. Error code: " << result << std::endl;
        return result;
    }

    // Read the export table pointer from the server
    if (rpc_read(ppExportTable, sizeof(void *)) < 0) {
        std::cerr << "Failed to read export table pointer from server" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    std::cout << "Client: Received export table pointer from server: " << *ppExportTable << std::endl;

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

void cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsShim() {
    std::cout << "calling cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsShim" << std::endl;
}

void cuOccupancyAvailableDynamicSMemPerBlockShim() {
    std::cout << "calling cuOccupancyAvailableDynamicSMemPerBlockShim" << std::endl;
}

void cuOccupancyMaxPotentialClusterSizeShim() {
    std::cout << "calling cuOccupancyMaxPotentialClusterSizeShim" << std::endl;
}

void cuOccupancyMaxActiveClustersShim() {
    std::cout << "calling cuOccupancyMaxActiveClustersShim" << std::endl;
}

void cuMemAdviseShim() {
    std::cout << "calling cuMemAdviseShim" << std::endl;
}

void cuMemPrefetchAsyncShim() {
    std::cout << "calling cuMemPrefetchAsyncShim" << std::endl;
}

void cuMemRangeGetAttributeShim() {
    std::cout << "calling cuMemRangeGetAttributeShim" << std::endl;
}

void cuMemRangeGetAttributesShim() {
    std::cout << "calling cuMemRangeGetAttributesShim" << std::endl;
}

void cuGetErrorStringShim() {
    std::cout << "calling cuGetErrorStringShim" << std::endl;
}

void cuGetErrorNameShim() {
    std::cout << "calling cuGetErrorNameShim" << std::endl;
}

void cuGraphCreateShim() {
    std::cout << "calling cuGraphCreateShim" << std::endl;
}

void cuGraphAddKernelNodeShim() {
    std::cout << "calling cuGraphAddKernelNodeShim" << std::endl;
}

void cuGraphKernelNodeGetParamsShim() {
    std::cout << "calling cuGraphKernelNodeGetParamsShim" << std::endl;
}

void cuGraphKernelNodeSetParamsShim() {
    std::cout << "calling cuGraphKernelNodeSetParamsShim" << std::endl;
}

void cuGraphAddMemcpyNodeShim() {
    std::cout << "calling cuGraphAddMemcpyNodeShim" << std::endl;
}

void cuGraphMemcpyNodeGetParamsShim() {
    std::cout << "calling cuGraphMemcpyNodeGetParamsShim" << std::endl;
}

void cuGraphMemcpyNodeSetParamsShim() {
    std::cout << "calling cuGraphMemcpyNodeSetParamsShim" << std::endl;
}

void cuGraphAddMemsetNodeShim() {
    std::cout << "calling cuGraphAddMemsetNodeShim" << std::endl;
}

void cuGraphMemsetNodeGetParamsShim() {
    std::cout << "calling cuGraphMemsetNodeGetParamsShim" << std::endl;
}

void cuGraphMemsetNodeSetParamsShim() {
    std::cout << "calling cuGraphMemsetNodeSetParamsShim" << std::endl;
}

void cuGraphAddHostNodeShim() {
    std::cout << "calling cuGraphAddHostNodeShim" << std::endl;
}

void cuGraphHostNodeGetParamsShim() {
    std::cout << "calling cuGraphHostNodeGetParamsShim" << std::endl;
}

void cuGraphHostNodeSetParamsShim() {
    std::cout << "calling cuGraphHostNodeSetParamsShim" << std::endl;
}

void cuGraphAddChildGraphNodeShim() {
    std::cout << "calling cuGraphAddChildGraphNodeShim" << std::endl;
}

void cuGraphChildGraphNodeGetGraphShim() {
    std::cout << "calling cuGraphChildGraphNodeGetGraphShim" << std::endl;
}

void cuGraphAddEmptyNodeShim() {
    std::cout << "calling cuGraphAddEmptyNodeShim" << std::endl;
}

void cuGraphAddEventRecordNodeShim() {
    std::cout << "calling cuGraphAddEventRecordNodeShim" << std::endl;
}

void cuGraphEventRecordNodeGetEventShim() {
    std::cout << "calling cuGraphEventRecordNodeGetEventShim" << std::endl;
}

void cuGraphEventRecordNodeSetEventShim() {
    std::cout << "calling cuGraphEventRecordNodeSetEventShim" << std::endl;
}

void cuGraphAddEventWaitNodeShim() {
    std::cout << "calling cuGraphAddEventWaitNodeShim" << std::endl;
}

void cuGraphEventWaitNodeGetEventShim() {
    std::cout << "calling cuGraphEventWaitNodeGetEventShim" << std::endl;
}

void cuGraphEventWaitNodeSetEventShim() {
    std::cout << "calling cuGraphEventWaitNodeSetEventShim" << std::endl;
}

void cuGraphAddExternalSemaphoresSignalNodeShim() {
    std::cout << "calling cuGraphAddExternalSemaphoresSignalNodeShim" << std::endl;
}

void cuGraphExternalSemaphoresSignalNodeGetParamsShim() {
    std::cout << "calling cuGraphExternalSemaphoresSignalNodeGetParamsShim" << std::endl;
}

void cuGraphExternalSemaphoresSignalNodeSetParamsShim() {
    std::cout << "calling cuGraphExternalSemaphoresSignalNodeSetParamsShim" << std::endl;
}

void cuGraphAddExternalSemaphoresWaitNodeShim() {
    std::cout << "calling cuGraphAddExternalSemaphoresWaitNodeShim" << std::endl;
}

void cuGraphExternalSemaphoresWaitNodeGetParamsShim() {
    std::cout << "calling cuGraphExternalSemaphoresWaitNodeGetParamsShim" << std::endl;
}

void cuGraphExternalSemaphoresWaitNodeSetParamsShim() {
    std::cout << "calling cuGraphExternalSemaphoresWaitNodeSetParamsShim" << std::endl;
}

void cuGraphExecExternalSemaphoresSignalNodeSetParamsShim() {
    std::cout << "calling cuGraphExecExternalSemaphoresSignalNodeSetParamsShim" << std::endl;
}

void cuGraphExecExternalSemaphoresWaitNodeSetParamsShim() {
    std::cout << "calling cuGraphExecExternalSemaphoresWaitNodeSetParamsShim" << std::endl;
}

void cuGraphAddMemAllocNodeShim() {
    std::cout << "calling cuGraphAddMemAllocNodeShim" << std::endl;
}

void cuGraphMemAllocNodeGetParamsShim() {
    std::cout << "calling cuGraphMemAllocNodeGetParamsShim" << std::endl;
}

void cuGraphAddMemFreeNodeShim() {
    std::cout << "calling cuGraphAddMemFreeNodeShim" << std::endl;
}

void cuGraphMemFreeNodeGetParamsShim() {
    std::cout << "calling cuGraphMemFreeNodeGetParamsShim" << std::endl;
}

void cuDeviceGraphMemTrimShim() {
    std::cout << "calling cuDeviceGraphMemTrimShim" << std::endl;
}

void cuDeviceGetGraphMemAttributeShim() {
    std::cout << "calling cuDeviceGetGraphMemAttributeShim" << std::endl;
}

void cuDeviceSetGraphMemAttributeShim() {
    std::cout << "calling cuDeviceSetGraphMemAttributeShim" << std::endl;
}

void cuGraphCloneShim() {
    std::cout << "calling cuGraphCloneShim" << std::endl;
}

void cuGraphNodeFindInCloneShim() {
    std::cout << "calling cuGraphNodeFindInCloneShim" << std::endl;
}

void cuGraphNodeGetTypeShim() {
    std::cout << "calling cuGraphNodeGetTypeShim" << std::endl;
}

void cuGraphGetNodesShim() {
    std::cout << "calling cuGraphGetNodesShim" << std::endl;
}

void cuGraphGetRootNodesShim() {
    std::cout << "calling cuGraphGetRootNodesShim" << std::endl;
}

void cuGraphGetEdgesShim() {
    std::cout << "calling cuGraphGetEdgesShim" << std::endl;
}

void cuGraphNodeGetDependenciesShim() {
    std::cout << "calling cuGraphNodeGetDependenciesShim" << std::endl;
}

void cuGraphNodeGetDependentNodesShim() {
    std::cout << "calling cuGraphNodeGetDependentNodesShim" << std::endl;
}

void cuGraphAddDependenciesShim() {
    std::cout << "calling cuGraphAddDependenciesShim" << std::endl;
}

void cuGraphRemoveDependenciesShim() {
    std::cout << "calling cuGraphRemoveDependenciesShim" << std::endl;
}

void cuGraphDestroyNodeShim() {
    std::cout << "calling cuGraphDestroyNodeShim" << std::endl;
}

void cuGraphInstantiateShim() {
    std::cout << "calling cuGraphInstantiateShim" << std::endl;
}

void cuGraphUploadShim() {
    std::cout << "calling cuGraphUploadShim" << std::endl;
}

void cuGraphLaunchShim() {
    std::cout << "calling cuGraphLaunchShim" << std::endl;
}

void cuGraphExecDestroyShim() {
    std::cout << "calling cuGraphExecDestroyShim" << std::endl;
}

void cuGraphDestroyShim() {
    std::cout << "calling cuGraphDestroyShim" << std::endl;
}

void cuStreamBeginCaptureShim() {
    std::cout << "calling cuStreamBeginCaptureShim" << std::endl;
}

void cuStreamEndCaptureShim() {
    std::cout << "calling cuStreamEndCaptureShim" << std::endl;
}

void cuStreamIsCapturingShim() {
    std::cout << "calling cuStreamIsCapturingShim" << std::endl;
}

void cuStreamGetCaptureInfoShim() {
    std::cout << "calling cuStreamGetCaptureInfoShim" << std::endl;
}

void cuStreamUpdateCaptureDependenciesShim() {
    std::cout << "calling cuStreamUpdateCaptureDependenciesShim" << std::endl;
}

void cuGraphExecKernelNodeSetParamsShim() {
    std::cout << "calling cuGraphExecKernelNodeSetParamsShim" << std::endl;
}

void cuGraphExecMemcpyNodeSetParamsShim() {
    std::cout << "calling cuGraphExecMemcpyNodeSetParamsShim" << std::endl;
}

void cuGraphExecMemsetNodeSetParamsShim() {
    std::cout << "calling cuGraphExecMemsetNodeSetParamsShim" << std::endl;
}

void cuGraphExecHostNodeSetParamsShim() {
    std::cout << "calling cuGraphExecHostNodeSetParamsShim" << std::endl;
}

void cuGraphExecChildGraphNodeSetParamsShim() {
    std::cout << "calling cuGraphExecChildGraphNodeSetParamsShim" << std::endl;
}

void cuGraphExecEventRecordNodeSetEventShim() {
    std::cout << "calling cuGraphExecEventRecordNodeSetEventShim" << std::endl;
}

void cuGraphExecEventWaitNodeSetEventShim() {
    std::cout << "calling cuGraphExecEventWaitNodeSetEventShim" << std::endl;
}

void cuThreadExchangeStreamCaptureModeShim() {
    std::cout << "calling cuThreadExchangeStreamCaptureModeShim" << std::endl;
}

void cuGraphExecUpdateShim() {
    std::cout << "calling cuGraphExecUpdateShim" << std::endl;
}

void cuGraphKernelNodeCopyAttributesShim() {
    std::cout << "calling cuGraphKernelNodeCopyAttributesShim" << std::endl;
}

void cuGraphKernelNodeGetAttributeShim() {
    std::cout << "calling cuGraphKernelNodeGetAttributeShim" << std::endl;
}

void cuGraphKernelNodeSetAttributeShim() {
    std::cout << "calling cuGraphKernelNodeSetAttributeShim" << std::endl;
}

void cuGraphDebugDotPrintShim() {
    std::cout << "calling cuGraphDebugDotPrintShim" << std::endl;
}

void cuUserObjectCreateShim() {
    std::cout << "calling cuUserObjectCreateShim" << std::endl;
}

void cuUserObjectRetainShim() {
    std::cout << "calling cuUserObjectRetainShim" << std::endl;
}

void cuUserObjectReleaseShim() {
    std::cout << "calling cuUserObjectReleaseShim" << std::endl;
}

void cuGraphRetainUserObjectShim() {
    std::cout << "calling cuGraphRetainUserObjectShim" << std::endl;
}

void cuGraphReleaseUserObjectShim() {
    std::cout << "calling cuGraphReleaseUserObjectShim" << std::endl;
}

void cuGraphNodeSetEnabledShim() {
    std::cout << "calling cuGraphNodeSetEnabledShim" << std::endl;
}

void cuGraphNodeGetEnabledShim() {
    std::cout << "calling cuGraphNodeGetEnabledShim" << std::endl;
}

void cuGraphInstantiateWithParamsShim() {
    std::cout << "calling cuGraphInstantiateWithParamsShim" << std::endl;
}

void cuGraphExecGetFlagsShim() {
    std::cout << "calling cuGraphExecGetFlagsShim" << std::endl;
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

// CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
// {
//     int request_id = rpc_start_request(RPC_cuMemcpyDtoH_v2);

//     if (request_id < 0 ||
//         rpc_write(dstHost, sizeof(void*)) < 0 ||
//         rpc_write(&srcDevice, sizeof(CUdeviceptr)) < 0 ||
//         rpc_write(&ByteCount, sizeof(size_t)) < 0 ||
//         rpc_wait_for_response(request_id) < 0 ||
//         rpc_read(dstHost, ByteCount) < 0)
//     {
//         return CUDA_ERROR_UNKNOWN;
//     }

//     return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
// }

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

// Map of symbols to their corresponding function pointers
std::unordered_map<std::string, void (*)()> cuFunctionMap;

CUresult cuDriverGetVersion_handler(int *driverVersion) {
    if (driverVersion == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Start the RPC request for cuDriverGetVersion
    int request_id = rpc_start_request(RPC_cuDriverGetVersion);
    if (request_id < 0) {
        return CUDA_ERROR_UNKNOWN;
    }

    // Wait for the server response
    if (rpc_wait_for_response(request_id) < 0) {
        return CUDA_ERROR_UNKNOWN;
    }

    // Read the result code from the server
    CUresult result;
    if (rpc_read(&result, sizeof(CUresult)) < 0) {
        return CUDA_ERROR_UNKNOWN;
    }

    // If the result indicates an error, return it directly
    if (result != CUDA_SUCCESS) {
        return result;
    }

    // Read the driver version from the server
    if (rpc_read(driverVersion, sizeof(int)) < 0) {
        return CUDA_ERROR_UNKNOWN;
    }

    std::cout << "Client: Received driver version from server: " << *driverVersion << std::endl;

    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

void cuGraphInstantiateWithParams_ptszShim() {
    std::cout << "Client: calling cuGraphInstantiateWithParams_ptsz" << std::endl;
}

void cuGraphInstantiateWithFlagsShim() {
    std::cout << "Client: calling cuGraphInstantiateWithFlags" << std::endl;
}

void cuEGLStreamConsumerConnectWithFlagsShim() {
    std::cout << "Client: calling cuEGLStreamConsumerConnectWithFlags" << std::endl;
}

void cuGraphicsResourceGetMappedPointerShim() {
    std::cout << "Client: calling cuGraphicsResourceGetMappedPointerShim" << std::endl;
}

cudaError_t cudaGetDeviceCountShim(int *count) {
    std::cout << "calling cudaGetDeviceCountShim" << std::endl;

    if (sockfd < 0) {
        std::cerr << "Socket not connected." << std::endl;
        return cudaErrorUnknown;
    }

    // Prepare the request ID for the server
    int request_id = rpc_start_request(RPC_cudaGetDeviceCount);
    if (request_id < 0) {
        std::cerr << "Failed to start request for cudaGetDeviceCount" << std::endl;
        return cudaErrorUnknown;
    }

    // Wait for the server's response
    if (rpc_wait_for_response(request_id) < 0) {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return cudaErrorUnknown;
    }

    // Read the result code from the server as cudaError_t
    cudaError_t result;
    if (rpc_read(&result, sizeof(cudaError_t)) < 0) {
        std::cerr << "Failed to read result from server" << std::endl;
        return cudaErrorUnknown;
    }

    // Check if the cudaGetDeviceCount call was successful
    if (result != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount call failed on the server. Error code: " << result << std::endl;
        return result;  // Return the actual error code from the server
    }

    // Read the device count from the server
    if (rpc_read(count, sizeof(int)) < 0) {
        std::cerr << "Failed to read device count from server" << std::endl;
        return cudaErrorUnknown;
    }

    if (read(sockfd, &result, sizeof(int)) < 0) {
        return cudaErrorUnknown;
    }

    pthread_mutex_unlock(&mutex);

    std::cout << "Client: Received device count from server: " << result << std::endl;

    return cudaSuccess;  // Return success if everything else worked
}

void initializeCuFunctionMap() {
    cuFunctionMap["cuInit"] = reinterpret_cast<void (*)()>(cuInit);
    cuFunctionMap["cuGetProcAddress"] = reinterpret_cast<void (*)()>(cuGetProcAddress);
    cuFunctionMap["cuDriverGetVersion"] = reinterpret_cast<void (*)()>(cuDriverGetVersion_handler);
    cuFunctionMap["cudaGetDeviceCount"] = reinterpret_cast<void (*)()>(cudaGetDeviceCountShim);
    cuFunctionMap["cuDeviceGet"] = reinterpret_cast<void (*)()>(cuDeviceGetShim);
    cuFunctionMap["cuDeviceGetCount"] = reinterpret_cast<void (*)()>(cuDeviceGetCountShim);
    cuFunctionMap["cuDeviceGetName"] = reinterpret_cast<void (*)()>(cuDeviceGetNameShim);
    cuFunctionMap["cuDeviceTotalMem"] = reinterpret_cast<void (*)()>(cuDeviceTotalMemShim);
    cuFunctionMap["cuDeviceGetAttribute"] = reinterpret_cast<void (*)()>(cuDeviceGetAttributeShim);
    cuFunctionMap["cuDeviceGetP2PAttribute"] = reinterpret_cast<void (*)()>(cuDeviceGetP2PAttributeShim);
    cuFunctionMap["cuDeviceGetByPCIBusId"] = reinterpret_cast<void (*)()>(cuDeviceGetByPCIBusIdShim);
    cuFunctionMap["cuDeviceGetPCIBusId"] = reinterpret_cast<void (*)()>(cuDeviceGetPCIBusIdShim);
    cuFunctionMap["cuDeviceGetUuid"] = reinterpret_cast<void (*)()>(cuDeviceGetUuidShim);
    cuFunctionMap["cuDeviceGetTexture1DLinearMaxWidth"] = reinterpret_cast<void (*)()>(cuDeviceGetTexture1DLinearMaxWidthShim);
    cuFunctionMap["cuDeviceGetDefaultMemPool"] = reinterpret_cast<void (*)()>(cuDeviceGetDefaultMemPoolShim);
    cuFunctionMap["cuDeviceSetMemPool"] = reinterpret_cast<void (*)()>(cuDeviceSetMemPoolShim);
    cuFunctionMap["cuDeviceGetMemPool"] = reinterpret_cast<void (*)()>(cuDeviceGetMemPoolShim);
    cuFunctionMap["cuFlushGPUDirectRDMAWrites"] = reinterpret_cast<void (*)()>(cuFlushGPUDirectRDMAWritesShim);
    cuFunctionMap["cuDevicePrimaryCtxRetain"] = reinterpret_cast<void (*)()>(cuDevicePrimaryCtxRetainShim);
    cuFunctionMap["cuDevicePrimaryCtxRelease"] = reinterpret_cast<void (*)()>(cuDevicePrimaryCtxReleaseShim);
    cuFunctionMap["cuDevicePrimaryCtxSetFlags"] = reinterpret_cast<void (*)()>(cuDevicePrimaryCtxSetFlagsShim);
    cuFunctionMap["cuDevicePrimaryCtxGetState"] = reinterpret_cast<void (*)()>(cuDevicePrimaryCtxGetStateShim);
    cuFunctionMap["cuDevicePrimaryCtxReset"] = reinterpret_cast<void (*)()>(cuDevicePrimaryCtxResetShim);
    cuFunctionMap["cuCtxCreate"] = reinterpret_cast<void (*)()>(cuCtxCreateShim);
    cuFunctionMap["cuCtxGetFlags"] = reinterpret_cast<void (*)()>(cuCtxGetFlagsShim);
    cuFunctionMap["cuCtxSetCurrent"] = reinterpret_cast<void (*)()>(cuCtxSetCurrentShim);
    cuFunctionMap["cuCtxGetCurrent"] = reinterpret_cast<void (*)()>(cuCtxGetCurrentShim);
    cuFunctionMap["cuCtxDetach"] = reinterpret_cast<void (*)()>(cuCtxDetachShim);
    cuFunctionMap["cuCtxGetApiVersion"] = reinterpret_cast<void (*)()>(cuCtxGetApiVersionShim);
    cuFunctionMap["cuCtxGetDevice"] = reinterpret_cast<void (*)()>(cuCtxGetDeviceShim);
    cuFunctionMap["cuCtxGetLimit"] = reinterpret_cast<void (*)()>(cuCtxGetLimitShim);
    cuFunctionMap["cuCtxSetLimit"] = reinterpret_cast<void (*)()>(cuCtxSetLimitShim);
    cuFunctionMap["cuCtxGetCacheConfig"] = reinterpret_cast<void (*)()>(cuCtxGetCacheConfigShim);
    cuFunctionMap["cuCtxSetCacheConfig"] = reinterpret_cast<void (*)()>(cuCtxSetCacheConfigShim);
    cuFunctionMap["cuCtxGetSharedMemConfig"] = reinterpret_cast<void (*)()>(cuCtxGetSharedMemConfigShim);
    cuFunctionMap["cuCtxGetStreamPriorityRange"] = reinterpret_cast<void (*)()>(cuCtxGetStreamPriorityRangeShim);
    cuFunctionMap["cuCtxSetSharedMemConfig"] = reinterpret_cast<void (*)()>(cuCtxSetSharedMemConfigShim);
    cuFunctionMap["cuCtxSynchronize"] = reinterpret_cast<void (*)()>(cuCtxSynchronizeShim);
    cuFunctionMap["cuCtxResetPersistingL2Cache"] = reinterpret_cast<void (*)()>(cuCtxResetPersistingL2CacheShim);
    cuFunctionMap["cuCtxPopCurrent"] = reinterpret_cast<void (*)()>(cuCtxPopCurrentShim);
    cuFunctionMap["cuCtxPushCurrent"] = reinterpret_cast<void (*)()>(cuCtxPushCurrentShim);
    cuFunctionMap["cuModuleLoad"] = reinterpret_cast<void (*)()>(cuModuleLoadShim);
    cuFunctionMap["cuModuleLoadData"] = reinterpret_cast<void (*)()>(cuModuleLoadDataShim);
    cuFunctionMap["cuModuleLoadFatBinary"] = reinterpret_cast<void (*)()>(cuModuleLoadFatBinaryShim);
    cuFunctionMap["cuModuleUnload"] = reinterpret_cast<void (*)()>(cuModuleUnloadShim);
    cuFunctionMap["cuModuleGetFunction"] = reinterpret_cast<void (*)()>(cuModuleGetFunctionShim);
    cuFunctionMap["cuModuleGetGlobal"] = reinterpret_cast<void (*)()>(cuModuleGetGlobalShim);
    cuFunctionMap["cuModuleGetTexRef"] = reinterpret_cast<void (*)()>(cuModuleGetTexRefShim);
    cuFunctionMap["cuModuleGetSurfRef"] = reinterpret_cast<void (*)()>(cuModuleGetSurfRefShim);
    cuFunctionMap["cuModuleGetLoadingMode"] = reinterpret_cast<void (*)()>(cuModuleGetLoadingModeShim);
    cuFunctionMap["cuLibraryLoadData"] = reinterpret_cast<void (*)()>(cuLibraryLoadDataShim);
    cuFunctionMap["cuLibraryLoadFromFile"] = reinterpret_cast<void (*)()>(cuLibraryLoadFromFileShim);
    cuFunctionMap["cuLibraryUnload"] = reinterpret_cast<void (*)()>(cuLibraryUnloadShim);
    cuFunctionMap["cuLibraryGetKernel"] = reinterpret_cast<void (*)()>(cuLibraryGetKernelShim);
    cuFunctionMap["cuLibraryGetModule"] = reinterpret_cast<void (*)()>(cuLibraryGetModuleShim);
    cuFunctionMap["cuKernelGetFunction"] = reinterpret_cast<void (*)()>(cuKernelGetFunctionShim);
    cuFunctionMap["cuLibraryGetGlobal"] = reinterpret_cast<void (*)()>(cuLibraryGetGlobalShim);
    cuFunctionMap["cuLibraryGetManaged"] = reinterpret_cast<void (*)()>(cuLibraryGetManagedShim);
    cuFunctionMap["cuKernelGetAttribute"] = reinterpret_cast<void (*)()>(cuKernelGetAttributeShim);
    cuFunctionMap["cuKernelSetAttribute"] = reinterpret_cast<void (*)()>(cuKernelSetAttributeShim);
    cuFunctionMap["cuKernelSetCacheConfig"] = reinterpret_cast<void (*)()>(cuKernelSetCacheConfigShim);
    cuFunctionMap["cuLinkCreate"] = reinterpret_cast<void (*)()>(cuLinkCreateShim);
    cuFunctionMap["cuLinkAddData"] = reinterpret_cast<void (*)()>(cuLinkAddDataShim);
    cuFunctionMap["cuLinkAddFile"] = reinterpret_cast<void (*)()>(cuLinkAddFileShim);
    cuFunctionMap["cuLinkComplete"] = reinterpret_cast<void (*)()>(cuLinkCompleteShim);
    cuFunctionMap["cuLinkDestroy"] = reinterpret_cast<void (*)()>(cuLinkDestroyShim);
    cuFunctionMap["cuMemGetInfo"] = reinterpret_cast<void (*)()>(cuMemGetInfoShim);
    cuFunctionMap["cuMemAllocManaged"] = reinterpret_cast<void (*)()>(cuMemAllocManagedShim);
    cuFunctionMap["cuMemAlloc"] = reinterpret_cast<void (*)()>(cuMemAllocShim);
    cuFunctionMap["cuMemAllocPitch"] = reinterpret_cast<void (*)()>(cuMemAllocPitchShim);
    cuFunctionMap["cuMemFree"] = reinterpret_cast<void (*)()>(cuMemFreeShim);
    cuFunctionMap["cuMemGetAddressRange"] = reinterpret_cast<void (*)()>(cuMemGetAddressRangeShim);
    cuFunctionMap["cuMemFreeHost"] = reinterpret_cast<void (*)()>(cuMemFreeHostShim);
    cuFunctionMap["cuMemHostAlloc"] = reinterpret_cast<void (*)()>(cuMemHostAllocShim);
    cuFunctionMap["cuMemHostGetDevicePointer"] = reinterpret_cast<void (*)()>(cuMemHostGetDevicePointerShim);
    cuFunctionMap["cuMemHostGetFlags"] = reinterpret_cast<void (*)()>(cuMemHostGetFlagsShim);
    cuFunctionMap["cuMemHostRegister"] = reinterpret_cast<void (*)()>(cuMemHostRegisterShim);
    cuFunctionMap["cuMemHostUnregister"] = reinterpret_cast<void (*)()>(cuMemHostUnregisterShim);
    cuFunctionMap["cuPointerGetAttribute"] = reinterpret_cast<void (*)()>(cuPointerGetAttributeShim);
    cuFunctionMap["cuPointerGetAttributes"] = reinterpret_cast<void (*)()>(cuPointerGetAttributesShim);
    cuFunctionMap["cuMemAllocAsync"] = reinterpret_cast<void (*)()>(cuMemAllocAsyncShim);
    cuFunctionMap["cuMemAllocFromPoolAsync"] = reinterpret_cast<void (*)()>(cuMemAllocFromPoolAsyncShim);
    cuFunctionMap["cuMemFreeAsync"] = reinterpret_cast<void (*)()>(cuMemFreeAsyncShim);
    cuFunctionMap["cuMemPoolTrimTo"] = reinterpret_cast<void (*)()>(cuMemPoolTrimToShim);
    cuFunctionMap["cuMemPoolSetAttribute"] = reinterpret_cast<void (*)()>(cuMemPoolSetAttributeShim);
    cuFunctionMap["cuMemPoolGetAttribute"] = reinterpret_cast<void (*)()>(cuMemPoolGetAttributeShim);
    cuFunctionMap["cuMemPoolSetAccess"] = reinterpret_cast<void (*)()>(cuMemPoolSetAccessShim);
    cuFunctionMap["cuMemPoolGetAccess"] = reinterpret_cast<void (*)()>(cuMemPoolGetAccessShim);
    cuFunctionMap["cuMemPoolCreate"] = reinterpret_cast<void (*)()>(cuMemPoolCreateShim);
    cuFunctionMap["cuMemPoolDestroy"] = reinterpret_cast<void (*)()>(cuMemPoolDestroyShim);
    cuFunctionMap["cuMemPoolExportToShareableHandle"] = reinterpret_cast<void (*)()>(cuMemPoolExportToShareableHandleShim);
    cuFunctionMap["cuMemPoolImportFromShareableHandle"] = reinterpret_cast<void (*)()>(cuMemPoolImportFromShareableHandleShim);
    cuFunctionMap["cuMemPoolExportPointer"] = reinterpret_cast<void (*)()>(cuMemPoolExportPointerShim);
    cuFunctionMap["cuMemPoolImportPointer"] = reinterpret_cast<void (*)()>(cuMemPoolImportPointerShim);
    cuFunctionMap["cuMemcpy"] = reinterpret_cast<void (*)()>(cuMemcpyShim);
    cuFunctionMap["cuMemcpyAsync"] = reinterpret_cast<void (*)()>(cuMemcpyAsyncShim);
    cuFunctionMap["cuMemcpyPeer"] = reinterpret_cast<void (*)()>(cuMemcpyPeerShim);
    cuFunctionMap["cuMemcpyPeerAsync"] = reinterpret_cast<void (*)()>(cuMemcpyPeerAsyncShim);
    cuFunctionMap["cuMemcpyHtoD"] = reinterpret_cast<void (*)()>(cuMemcpyHtoDShim);
    cuFunctionMap["cuMemcpyHtoDAsync"] = reinterpret_cast<void (*)()>(cuMemcpyHtoDAsyncShim);
    cuFunctionMap["cuMemcpyDtoH"] = reinterpret_cast<void (*)()>(cuMemcpyDtoHShim);
    cuFunctionMap["cuMemcpyDtoHAsync"] = reinterpret_cast<void (*)()>(cuMemcpyDtoHAsyncShim);
    cuFunctionMap["cuMemcpyDtoD"] = reinterpret_cast<void (*)()>(cuMemcpyDtoDShim);
    cuFunctionMap["cuMemcpyDtoDAsync"] = reinterpret_cast<void (*)()>(cuMemcpyDtoDAsyncShim);
    cuFunctionMap["cuMemcpy2DUnaligned"] = reinterpret_cast<void (*)()>(cuMemcpy2DUnalignedShim);
    cuFunctionMap["cuMemcpy2DAsync"] = reinterpret_cast<void (*)()>(cuMemcpy2DAsyncShim);
    cuFunctionMap["cuMemcpy3D"] = reinterpret_cast<void (*)()>(cuMemcpy3DShim);
    cuFunctionMap["cuMemcpy3DAsync"] = reinterpret_cast<void (*)()>(cuMemcpy3DAsyncShim);
    cuFunctionMap["cuMemcpy3DPeer"] = reinterpret_cast<void (*)()>(cuMemcpy3DPeerShim);
    cuFunctionMap["cuMemcpy3DPeerAsync"] = reinterpret_cast<void (*)()>(cuMemcpy3DPeerAsyncShim);
    cuFunctionMap["cuMemsetD8"] = reinterpret_cast<void (*)()>(cuMemsetD8Shim);
    cuFunctionMap["cuMemsetD8Async"] = reinterpret_cast<void (*)()>(cuMemsetD8AsyncShim);
    cuFunctionMap["cuMemsetD2D8"] = reinterpret_cast<void (*)()>(cuMemsetD2D8Shim);
    cuFunctionMap["cuMemsetD2D8Async"] = reinterpret_cast<void (*)()>(cuMemsetD2D8AsyncShim);
    cuFunctionMap["cuFuncSetCacheConfig"] = reinterpret_cast<void (*)()>(cuFuncSetCacheConfigShim);
    cuFunctionMap["cuFuncSetSharedMemConfig"] = reinterpret_cast<void (*)()>(cuFuncSetSharedMemConfigShim);
    cuFunctionMap["cuFuncGetAttribute"] = reinterpret_cast<void (*)()>(cuFuncGetAttributeShim);
    cuFunctionMap["cuFuncSetAttribute"] = reinterpret_cast<void (*)()>(cuFuncSetAttributeShim);
    cuFunctionMap["cuArrayCreate"] = reinterpret_cast<void (*)()>(cuArrayCreateShim);
    cuFunctionMap["cuArrayGetDescriptor"] = reinterpret_cast<void (*)()>(cuArrayGetDescriptorShim);
    cuFunctionMap["cuArrayGetSparseProperties"] = reinterpret_cast<void (*)()>(cuArrayGetSparsePropertiesShim);
    cuFunctionMap["cuArrayGetPlane"] = reinterpret_cast<void (*)()>(cuArrayGetPlaneShim);
    cuFunctionMap["cuArray3DCreate"] = reinterpret_cast<void (*)()>(cuArray3DCreateShim);
    cuFunctionMap["cuArray3DGetDescriptor"] = reinterpret_cast<void (*)()>(cuArray3DGetDescriptorShim);
    cuFunctionMap["cuArrayDestroy"] = reinterpret_cast<void (*)()>(cuArrayDestroyShim);
    cuFunctionMap["cuMipmappedArrayCreate"] = reinterpret_cast<void (*)()>(cuMipmappedArrayCreateShim);
    cuFunctionMap["cuMipmappedArrayGetLevel"] = reinterpret_cast<void (*)()>(cuMipmappedArrayGetLevelShim);
    cuFunctionMap["cuMipmappedArrayGetSparseProperties"] = reinterpret_cast<void (*)()>(cuMipmappedArrayGetSparsePropertiesShim);
    cuFunctionMap["cuMipmappedArrayDestroy"] = reinterpret_cast<void (*)()>(cuMipmappedArrayDestroyShim);
    cuFunctionMap["cuArrayGetMemoryRequirements"] = reinterpret_cast<void (*)()>(cuArrayGetMemoryRequirementsShim);
    cuFunctionMap["cuMipmappedArrayGetMemoryRequirements"] = reinterpret_cast<void (*)()>(cuMipmappedArrayGetMemoryRequirementsShim);
    cuFunctionMap["cuTexObjectCreate"] = reinterpret_cast<void (*)()>(cuTexObjectCreateShim);
    cuFunctionMap["cuTexObjectDestroy"] = reinterpret_cast<void (*)()>(cuTexObjectDestroyShim);
    cuFunctionMap["cuTexObjectGetResourceDesc"] = reinterpret_cast<void (*)()>(cuTexObjectGetResourceDescShim);
    cuFunctionMap["cuTexObjectGetTextureDesc"] = reinterpret_cast<void (*)()>(cuTexObjectGetTextureDescShim);
    cuFunctionMap["cuTexObjectGetResourceViewDesc"] = reinterpret_cast<void (*)()>(cuTexObjectGetResourceViewDescShim);
    cuFunctionMap["cuSurfObjectCreate"] = reinterpret_cast<void (*)()>(cuSurfObjectCreateShim);
    cuFunctionMap["cuSurfObjectDestroy"] = reinterpret_cast<void (*)()>(cuSurfObjectDestroyShim);
    cuFunctionMap["cuSurfObjectGetResourceDesc"] = reinterpret_cast<void (*)()>(cuSurfObjectGetResourceDescShim);
    cuFunctionMap["cuImportExternalMemory"] = reinterpret_cast<void (*)()>(cuImportExternalMemoryShim);
    cuFunctionMap["cuExternalMemoryGetMappedBuffer"] = reinterpret_cast<void (*)()>(cuExternalMemoryGetMappedBufferShim);
    cuFunctionMap["cuExternalMemoryGetMappedMipmappedArray"] = reinterpret_cast<void (*)()>(cuExternalMemoryGetMappedMipmappedArrayShim);
    cuFunctionMap["cuDestroyExternalMemory"] = reinterpret_cast<void (*)()>(cuDestroyExternalMemoryShim);
    cuFunctionMap["cuImportExternalSemaphore"] = reinterpret_cast<void (*)()>(cuImportExternalSemaphoreShim);
    cuFunctionMap["cuSignalExternalSemaphoresAsync"] = reinterpret_cast<void (*)()>(cuSignalExternalSemaphoresAsyncShim);
    cuFunctionMap["cuWaitExternalSemaphoresAsync"] = reinterpret_cast<void (*)()>(cuWaitExternalSemaphoresAsyncShim);
    cuFunctionMap["cuDestroyExternalSemaphore"] = reinterpret_cast<void (*)()>(cuDestroyExternalSemaphoreShim);
    cuFunctionMap["cuDeviceGetNvSciSyncAttributes"] = reinterpret_cast<void (*)()>(cuDeviceGetNvSciSyncAttributesShim);
    cuFunctionMap["cuLaunchKernel"] = reinterpret_cast<void (*)()>(cuLaunchKernelShim);
    cuFunctionMap["cuLaunchCooperativeKernel"] = reinterpret_cast<void (*)()>(cuLaunchCooperativeKernelShim);
    cuFunctionMap["cuLaunchCooperativeKernelMultiDevice"] = reinterpret_cast<void (*)()>(cuLaunchCooperativeKernelMultiDeviceShim);
    cuFunctionMap["cuLaunchHostFunc"] = reinterpret_cast<void (*)()>(cuLaunchHostFuncShim);
    cuFunctionMap["cuLaunchKernelEx"] = reinterpret_cast<void (*)()>(cuLaunchKernelExShim);
    cuFunctionMap["cuEventCreate"] = reinterpret_cast<void (*)()>(cuEventCreateShim);
    cuFunctionMap["cuEventRecord"] = reinterpret_cast<void (*)()>(cuEventRecordShim);
    cuFunctionMap["cuEventRecordWithFlags"] = reinterpret_cast<void (*)()>(cuEventRecordWithFlagsShim);
    cuFunctionMap["cuEventQuery"] = reinterpret_cast<void (*)()>(cuEventQueryShim);
    cuFunctionMap["cuEventSynchronize"] = reinterpret_cast<void (*)()>(cuEventSynchronizeShim);
    cuFunctionMap["cuEventDestroy"] = reinterpret_cast<void (*)()>(cuEventDestroyShim);
    cuFunctionMap["cuEventElapsedTime"] = reinterpret_cast<void (*)()>(cuEventElapsedTimeShim);
    cuFunctionMap["cuStreamWaitValue32"] = reinterpret_cast<void (*)()>(cuStreamWaitValue32Shim);
    cuFunctionMap["cuStreamWriteValue32"] = reinterpret_cast<void (*)()>(cuStreamWriteValue32Shim);
    cuFunctionMap["cuStreamWaitValue64"] = reinterpret_cast<void (*)()>(cuStreamWaitValue64Shim);
    cuFunctionMap["cuStreamWriteValue64"] = reinterpret_cast<void (*)()>(cuStreamWriteValue64Shim);
    cuFunctionMap["cuStreamBatchMemOp"] = reinterpret_cast<void (*)()>(cuStreamBatchMemOpShim);
    cuFunctionMap["cuStreamCreate"] = reinterpret_cast<void (*)()>(cuStreamCreateShim);
    cuFunctionMap["cuStreamCreateWithPriority"] = reinterpret_cast<void (*)()>(cuStreamCreateWithPriorityShim);
    cuFunctionMap["cuStreamGetPriority"] = reinterpret_cast<void (*)()>(cuStreamGetPriorityShim);
    cuFunctionMap["cuStreamGetFlags"] = reinterpret_cast<void (*)()>(cuStreamGetFlagsShim);
    cuFunctionMap["cuStreamGetCtx"] = reinterpret_cast<void (*)()>(cuStreamGetCtxShim);
    cuFunctionMap["cuStreamGetId"] = reinterpret_cast<void (*)()>(cuStreamGetIdShim);
    cuFunctionMap["cuStreamDestroy"] = reinterpret_cast<void (*)()>(cuStreamDestroyShim);
    cuFunctionMap["cuStreamWaitEvent"] = reinterpret_cast<void (*)()>(cuStreamWaitEventShim);
    cuFunctionMap["cuStreamAddCallback"] = reinterpret_cast<void (*)()>(cuStreamAddCallbackShim);
    cuFunctionMap["cuStreamSynchronize"] = reinterpret_cast<void (*)()>(cuStreamSynchronizeShim);
    cuFunctionMap["cuStreamQuery"] = reinterpret_cast<void (*)()>(cuStreamQueryShim);
    cuFunctionMap["cuStreamAttachMemAsync"] = reinterpret_cast<void (*)()>(cuStreamAttachMemAsyncShim);
    cuFunctionMap["cuStreamCopyAttributes"] = reinterpret_cast<void (*)()>(cuStreamCopyAttributesShim);
    cuFunctionMap["cuStreamGetAttribute"] = reinterpret_cast<void (*)()>(cuStreamGetAttributeShim);
    cuFunctionMap["cuStreamSetAttribute"] = reinterpret_cast<void (*)()>(cuStreamSetAttributeShim);
    cuFunctionMap["cuDeviceCanAccessPeer"] = reinterpret_cast<void (*)()>(cuDeviceCanAccessPeerShim);
    cuFunctionMap["cuCtxEnablePeerAccess"] = reinterpret_cast<void (*)()>(cuCtxEnablePeerAccessShim);
    cuFunctionMap["cuCtxDisablePeerAccess"] = reinterpret_cast<void (*)()>(cuCtxDisablePeerAccessShim);
    cuFunctionMap["cuIpcGetEventHandle"] = reinterpret_cast<void (*)()>(cuIpcGetEventHandleShim);
    cuFunctionMap["cuIpcOpenEventHandle"] = reinterpret_cast<void (*)()>(cuIpcOpenEventHandleShim);
    cuFunctionMap["cuIpcGetMemHandle"] = reinterpret_cast<void (*)()>(cuIpcGetMemHandleShim);
    cuFunctionMap["cuIpcOpenMemHandle"] = reinterpret_cast<void (*)()>(cuIpcOpenMemHandleShim);
    cuFunctionMap["cuIpcCloseMemHandle"] = reinterpret_cast<void (*)()>(cuIpcCloseMemHandleShim);
    cuFunctionMap["cuGLCtxCreate"] = reinterpret_cast<void (*)()>(cuGLCtxCreateShim);
    cuFunctionMap["cuGLInit"] = reinterpret_cast<void (*)()>(cuGLInitShim);
    cuFunctionMap["cuGLGetDevices"] = reinterpret_cast<void (*)()>(cuGLGetDevicesShim);
    cuFunctionMap["cuGLRegisterBufferObject"] = reinterpret_cast<void (*)()>(cuGLRegisterBufferObjectShim);
    cuFunctionMap["cuGLMapBufferObject"] = reinterpret_cast<void (*)()>(cuGLMapBufferObjectShim);
    cuFunctionMap["cuGLMapBufferObjectAsync"] = reinterpret_cast<void (*)()>(cuGLMapBufferObjectAsyncShim);
    cuFunctionMap["cuGLUnmapBufferObject"] = reinterpret_cast<void (*)()>(cuGLUnmapBufferObjectShim);
    cuFunctionMap["cuGLUnmapBufferObjectAsync"] = reinterpret_cast<void (*)()>(cuGLUnmapBufferObjectAsyncShim);
    cuFunctionMap["cuGLUnregisterBufferObject"] = reinterpret_cast<void (*)()>(cuGLUnregisterBufferObjectShim);
    cuFunctionMap["cuGLSetBufferObjectMapFlags"] = reinterpret_cast<void (*)()>(cuGLSetBufferObjectMapFlagsShim);
    cuFunctionMap["cuGraphicsGLRegisterImage"] = reinterpret_cast<void (*)()>(cuGraphicsGLRegisterImageShim);
    cuFunctionMap["cuGraphicsGLRegisterBuffer"] = reinterpret_cast<void (*)()>(cuGraphicsGLRegisterBufferShim);
    cuFunctionMap["cuGraphicsEGLRegisterImage"] = reinterpret_cast<void (*)()>(cuGraphicsEGLRegisterImageShim);
    cuFunctionMap["cuEGLStreamConsumerConnect"] = reinterpret_cast<void (*)()>(cuEGLStreamConsumerConnectShim);
    cuFunctionMap["cuEGLStreamConsumerDisconnect"] = reinterpret_cast<void (*)()>(cuEGLStreamConsumerDisconnectShim);
    cuFunctionMap["cuEGLStreamConsumerAcquireFrame"] = reinterpret_cast<void (*)()>(cuEGLStreamConsumerAcquireFrameShim);
    cuFunctionMap["cuEGLStreamConsumerReleaseFrame"] = reinterpret_cast<void (*)()>(cuEGLStreamConsumerReleaseFrameShim);
    cuFunctionMap["cuEGLStreamProducerConnect"] = reinterpret_cast<void (*)()>(cuEGLStreamProducerConnectShim);
    cuFunctionMap["cuEGLStreamProducerDisconnect"] = reinterpret_cast<void (*)()>(cuEGLStreamProducerDisconnectShim);
    cuFunctionMap["cuEGLStreamProducerPresentFrame"] = reinterpret_cast<void (*)()>(cuEGLStreamProducerPresentFrameShim);
    cuFunctionMap["cuEGLStreamProducerReturnFrame"] = reinterpret_cast<void (*)()>(cuEGLStreamProducerReturnFrameShim);
    cuFunctionMap["cuGraphicsResourceGetMappedEglFrame"] = reinterpret_cast<void (*)()>(cuGraphicsResourceGetMappedEglFrameShim);
    cuFunctionMap["cuGraphicsUnregisterResource"] = reinterpret_cast<void (*)()>(cuGraphicsUnregisterResourceShim);
    cuFunctionMap["cuGraphicsMapResources"] = reinterpret_cast<void (*)()>(cuGraphicsMapResourcesShim);
    cuFunctionMap["cuGraphicsUnmapResources"] = reinterpret_cast<void (*)()>(cuGraphicsUnmapResourcesShim);
    cuFunctionMap["cuGraphicsResourceSetMapFlags"] = reinterpret_cast<void (*)()>(cuGraphicsResourceSetMapFlagsShim);
    cuFunctionMap["cuGraphicsSubResourceGetMappedArray"] = reinterpret_cast<void (*)()>(cuGraphicsSubResourceGetMappedArrayShim);
    cuFunctionMap["cuGraphicsResourceGetMappedMipmappedArray"] = reinterpret_cast<void (*)()>(cuGraphicsResourceGetMappedMipmappedArrayShim);
    cuFunctionMap["cuProfilerInitialize"] = reinterpret_cast<void (*)()>(cuProfilerInitializeShim);
    cuFunctionMap["cuProfilerStart"] = reinterpret_cast<void (*)()>(cuProfilerStartShim);
    cuFunctionMap["cuProfilerStop"] = reinterpret_cast<void (*)()>(cuProfilerStopShim);
    cuFunctionMap["cuVDPAUGetDevice"] = reinterpret_cast<void (*)()>(cuVDPAUGetDeviceShim);
    cuFunctionMap["cuVDPAUCtxCreate"] = reinterpret_cast<void (*)()>(cuVDPAUCtxCreateShim);
    cuFunctionMap["cuGraphicsVDPAURegisterVideoSurface"] = reinterpret_cast<void (*)()>(cuGraphicsVDPAURegisterVideoSurfaceShim);
    cuFunctionMap["cuGraphicsVDPAURegisterOutputSurface"] = reinterpret_cast<void (*)()>(cuGraphicsVDPAURegisterOutputSurfaceShim);
    cuFunctionMap["cuGetExportTable"] = reinterpret_cast<void (*)()>(cuGetExportTableShim);
    cuFunctionMap["cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"] = reinterpret_cast<void (*)()>(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsShim);
    cuFunctionMap["cuOccupancyAvailableDynamicSMemPerBlock"] = reinterpret_cast<void (*)()>(cuOccupancyAvailableDynamicSMemPerBlockShim);
    cuFunctionMap["cuOccupancyMaxPotentialClusterSize"] = reinterpret_cast<void (*)()>(cuOccupancyMaxPotentialClusterSizeShim);
    cuFunctionMap["cuOccupancyMaxActiveClusters"] = reinterpret_cast<void (*)()>(cuOccupancyMaxActiveClustersShim);
    cuFunctionMap["cuMemAdvise"] = reinterpret_cast<void (*)()>(cuMemAdviseShim);
    cuFunctionMap["cuMemPrefetchAsync"] = reinterpret_cast<void (*)()>(cuMemPrefetchAsyncShim);
    cuFunctionMap["cuMemRangeGetAttribute"] = reinterpret_cast<void (*)()>(cuMemRangeGetAttributeShim);
    cuFunctionMap["cuMemRangeGetAttributes"] = reinterpret_cast<void (*)()>(cuMemRangeGetAttributesShim);
    cuFunctionMap["cuGetErrorString"] = reinterpret_cast<void (*)()>(cuGetErrorStringShim);
    cuFunctionMap["cuGetErrorName"] = reinterpret_cast<void (*)()>(cuGetErrorNameShim);
    cuFunctionMap["cuGraphCreate"] = reinterpret_cast<void (*)()>(cuGraphCreateShim);
    cuFunctionMap["cuGraphAddKernelNode"] = reinterpret_cast<void (*)()>(cuGraphAddKernelNodeShim);
    cuFunctionMap["cuGraphKernelNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphKernelNodeGetParamsShim);
    cuFunctionMap["cuGraphKernelNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphKernelNodeSetParamsShim);
    cuFunctionMap["cuGraphAddMemcpyNode"] = reinterpret_cast<void (*)()>(cuGraphAddMemcpyNodeShim);
    cuFunctionMap["cuGraphMemcpyNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphMemcpyNodeGetParamsShim);
    cuFunctionMap["cuGraphMemcpyNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphMemcpyNodeSetParamsShim);
    cuFunctionMap["cuGraphAddMemsetNode"] = reinterpret_cast<void (*)()>(cuGraphAddMemsetNodeShim);
    cuFunctionMap["cuGraphMemsetNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphMemsetNodeGetParamsShim);
    cuFunctionMap["cuGraphMemsetNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphMemsetNodeSetParamsShim);
    cuFunctionMap["cuGraphAddHostNode"] = reinterpret_cast<void (*)()>(cuGraphAddHostNodeShim);
    cuFunctionMap["cuGraphHostNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphHostNodeGetParamsShim);
    cuFunctionMap["cuGraphHostNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphHostNodeSetParamsShim);
    cuFunctionMap["cuGraphAddChildGraphNode"] = reinterpret_cast<void (*)()>(cuGraphAddChildGraphNodeShim);
    cuFunctionMap["cuGraphChildGraphNodeGetGraph"] = reinterpret_cast<void (*)()>(cuGraphChildGraphNodeGetGraphShim);
    cuFunctionMap["cuGraphAddEmptyNode"] = reinterpret_cast<void (*)()>(cuGraphAddEmptyNodeShim);
    cuFunctionMap["cuGraphAddEventRecordNode"] = reinterpret_cast<void (*)()>(cuGraphAddEventRecordNodeShim);
    cuFunctionMap["cuGraphEventRecordNodeGetEvent"] = reinterpret_cast<void (*)()>(cuGraphEventRecordNodeGetEventShim);
    cuFunctionMap["cuGraphEventRecordNodeSetEvent"] = reinterpret_cast<void (*)()>(cuGraphEventRecordNodeSetEventShim);
    cuFunctionMap["cuGraphAddEventWaitNode"] = reinterpret_cast<void (*)()>(cuGraphAddEventWaitNodeShim);
    cuFunctionMap["cuGraphEventWaitNodeGetEvent"] = reinterpret_cast<void (*)()>(cuGraphEventWaitNodeGetEventShim);
    cuFunctionMap["cuGraphEventWaitNodeSetEvent"] = reinterpret_cast<void (*)()>(cuGraphEventWaitNodeSetEventShim);
    cuFunctionMap["cuGraphAddExternalSemaphoresSignalNode"] = reinterpret_cast<void (*)()>(cuGraphAddExternalSemaphoresSignalNodeShim);
    cuFunctionMap["cuGraphExternalSemaphoresSignalNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphExternalSemaphoresSignalNodeGetParamsShim);
    cuFunctionMap["cuGraphExternalSemaphoresSignalNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExternalSemaphoresSignalNodeSetParamsShim);
    cuFunctionMap["cuGraphAddExternalSemaphoresWaitNode"] = reinterpret_cast<void (*)()>(cuGraphAddExternalSemaphoresWaitNodeShim);
    cuFunctionMap["cuGraphExternalSemaphoresWaitNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphExternalSemaphoresWaitNodeGetParamsShim);
    cuFunctionMap["cuGraphExternalSemaphoresWaitNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExternalSemaphoresWaitNodeSetParamsShim);
    cuFunctionMap["cuGraphExecExternalSemaphoresSignalNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecExternalSemaphoresSignalNodeSetParamsShim);
    cuFunctionMap["cuGraphExecExternalSemaphoresWaitNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecExternalSemaphoresWaitNodeSetParamsShim);
    cuFunctionMap["cuGraphAddMemAllocNode"] = reinterpret_cast<void (*)()>(cuGraphAddMemAllocNodeShim);
    cuFunctionMap["cuGraphMemAllocNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphMemAllocNodeGetParamsShim);
    cuFunctionMap["cuGraphAddMemFreeNode"] = reinterpret_cast<void (*)()>(cuGraphAddMemFreeNodeShim);
    cuFunctionMap["cuGraphMemFreeNodeGetParams"] = reinterpret_cast<void (*)()>(cuGraphMemFreeNodeGetParamsShim);
    cuFunctionMap["cuDeviceGraphMemTrim"] = reinterpret_cast<void (*)()>(cuDeviceGraphMemTrimShim);
    cuFunctionMap["cuDeviceGetGraphMemAttribute"] = reinterpret_cast<void (*)()>(cuDeviceGetGraphMemAttributeShim);
    cuFunctionMap["cuDeviceSetGraphMemAttribute"] = reinterpret_cast<void (*)()>(cuDeviceSetGraphMemAttributeShim);
    cuFunctionMap["cuGraphClone"] = reinterpret_cast<void (*)()>(cuGraphCloneShim);
    cuFunctionMap["cuGraphNodeFindInClone"] = reinterpret_cast<void (*)()>(cuGraphNodeFindInCloneShim);
    cuFunctionMap["cuGraphNodeGetType"] = reinterpret_cast<void (*)()>(cuGraphNodeGetTypeShim);
    cuFunctionMap["cuGraphGetNodes"] = reinterpret_cast<void (*)()>(cuGraphGetNodesShim);
    cuFunctionMap["cuGraphGetRootNodes"] = reinterpret_cast<void (*)()>(cuGraphGetRootNodesShim);
    cuFunctionMap["cuGraphGetEdges"] = reinterpret_cast<void (*)()>(cuGraphGetEdgesShim);
    cuFunctionMap["cuGraphNodeGetDependencies"] = reinterpret_cast<void (*)()>(cuGraphNodeGetDependenciesShim);
    cuFunctionMap["cuGraphNodeGetDependentNodes"] = reinterpret_cast<void (*)()>(cuGraphNodeGetDependentNodesShim);
    cuFunctionMap["cuGraphAddDependencies"] = reinterpret_cast<void (*)()>(cuGraphAddDependenciesShim);
    cuFunctionMap["cuGraphRemoveDependencies"] = reinterpret_cast<void (*)()>(cuGraphRemoveDependenciesShim);
    cuFunctionMap["cuGraphDestroyNode"] = reinterpret_cast<void (*)()>(cuGraphDestroyNodeShim);
    cuFunctionMap["cuGraphInstantiate"] = reinterpret_cast<void (*)()>(cuGraphInstantiateShim);
    cuFunctionMap["cuGraphUpload"] = reinterpret_cast<void (*)()>(cuGraphUploadShim);
    cuFunctionMap["cuGraphLaunch"] = reinterpret_cast<void (*)()>(cuGraphLaunchShim);
    cuFunctionMap["cuGraphExecDestroy"] = reinterpret_cast<void (*)()>(cuGraphExecDestroyShim);
    cuFunctionMap["cuGraphDestroy"] = reinterpret_cast<void (*)()>(cuGraphDestroyShim);
    cuFunctionMap["cuStreamBeginCapture"] = reinterpret_cast<void (*)()>(cuStreamBeginCaptureShim);
    cuFunctionMap["cuStreamEndCapture"] = reinterpret_cast<void (*)()>(cuStreamEndCaptureShim);
    cuFunctionMap["cuStreamIsCapturing"] = reinterpret_cast<void (*)()>(cuStreamIsCapturingShim);
    cuFunctionMap["cuStreamGetCaptureInfo"] = reinterpret_cast<void (*)()>(cuStreamGetCaptureInfoShim);
    cuFunctionMap["cuStreamUpdateCaptureDependencies"] = reinterpret_cast<void (*)()>(cuStreamUpdateCaptureDependenciesShim);
    cuFunctionMap["cuGraphExecKernelNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecKernelNodeSetParamsShim);
    cuFunctionMap["cuGraphExecMemcpyNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecMemcpyNodeSetParamsShim);
    cuFunctionMap["cuGraphExecMemsetNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecMemsetNodeSetParamsShim);
    cuFunctionMap["cuGraphExecHostNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecHostNodeSetParamsShim);
    cuFunctionMap["cuGraphExecChildGraphNodeSetParams"] = reinterpret_cast<void (*)()>(cuGraphExecChildGraphNodeSetParamsShim);
    cuFunctionMap["cuGraphExecEventRecordNodeSetEvent"] = reinterpret_cast<void (*)()>(cuGraphExecEventRecordNodeSetEventShim);
    cuFunctionMap["cuGraphExecEventWaitNodeSetEvent"] = reinterpret_cast<void (*)()>(cuGraphExecEventWaitNodeSetEventShim);
    cuFunctionMap["cuThreadExchangeStreamCaptureMode"] = reinterpret_cast<void (*)()>(cuThreadExchangeStreamCaptureModeShim);
    cuFunctionMap["cuGraphExecUpdate"] = reinterpret_cast<void (*)()>(cuGraphExecUpdateShim);
    cuFunctionMap["cuGraphKernelNodeCopyAttributes"] = reinterpret_cast<void (*)()>(cuGraphKernelNodeCopyAttributesShim);
    cuFunctionMap["cuGraphKernelNodeGetAttribute"] = reinterpret_cast<void (*)()>(cuGraphKernelNodeGetAttributeShim);
    cuFunctionMap["cuGraphKernelNodeSetAttribute"] = reinterpret_cast<void (*)()>(cuGraphKernelNodeSetAttributeShim);
    cuFunctionMap["cuGraphDebugDotPrint"] = reinterpret_cast<void (*)()>(cuGraphDebugDotPrintShim);
    cuFunctionMap["cuUserObjectCreate"] = reinterpret_cast<void (*)()>(cuUserObjectCreateShim);
    cuFunctionMap["cuUserObjectRetain"] = reinterpret_cast<void (*)()>(cuUserObjectRetainShim);
    cuFunctionMap["cuUserObjectRelease"] = reinterpret_cast<void (*)()>(cuUserObjectReleaseShim);
    cuFunctionMap["cuGraphRetainUserObject"] = reinterpret_cast<void (*)()>(cuGraphRetainUserObjectShim);
    cuFunctionMap["cuGraphReleaseUserObject"] = reinterpret_cast<void (*)()>(cuGraphReleaseUserObjectShim);
    cuFunctionMap["cuGraphNodeSetEnabled"] = reinterpret_cast<void (*)()>(cuGraphNodeSetEnabledShim);
    cuFunctionMap["cuGraphNodeGetEnabled"] = reinterpret_cast<void (*)()>(cuGraphNodeGetEnabledShim);
    cuFunctionMap["cuGraphInstantiateWithParams"] = reinterpret_cast<void (*)()>(cuGraphInstantiateWithParamsShim);
    cuFunctionMap["cuGraphExecGetFlags"] = reinterpret_cast<void (*)()>(cuGraphExecGetFlagsShim);

    cuFunctionMap["cuGraphInstantiateWithParams_ptsz"] = reinterpret_cast<void (*)()>(cuGraphInstantiateWithParams_ptszShim);
    cuFunctionMap["cuGraphInstantiateWithFlags"] = reinterpret_cast<void (*)()>(cuGraphInstantiateWithFlagsShim);
    cuFunctionMap["cuEGLStreamConsumerConnectWithFlags"] = reinterpret_cast<void (*)()>(cuEGLStreamConsumerConnectWithFlagsShim);
    cuFunctionMap["cuGraphicsResourceGetMappedPointer"] = reinterpret_cast<void (*)()>(cuGraphicsResourceGetMappedPointerShim);
}

void noOpFunction() {
    // Do nothing
}

CUresult cuGetProcAddress_v2_handler(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
{
    initializeCuFunctionMap();
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

    // Read the function pointer from the response
    if (rpc_read(pfn, sizeof(void *)) < 0 || rpc_read(symbolStatus, sizeof(CUdriverProcAddressQueryResult)) < 0)
    {
        return CUDA_ERROR_UNKNOWN;
    }

    std::string symbolName(symbol);
    auto it = cuFunctionMap.find(symbolName);
    if (it != cuFunctionMap.end()) {
        *pfn = reinterpret_cast<void *>(it->second);
        std::cout << "Mapped symbol: " << symbolName << " to function: " << *pfn << std::endl;
    } else {
        void *fn = nullptr;
        std::cerr << "Function for symbol: " << symbolName << " not found!" << std::endl;
        *pfn = reinterpret_cast<void *>(noOpFunction); 
    }

    // Retrieve and return the result from the response
    return rpc_get_return<CUresult>(request_id, CUDA_ERROR_UNKNOWN);
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
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

    // Read the function pointer from the response
    if (rpc_read(pfn, sizeof(void *)) < 0 || rpc_read(symbolStatus, sizeof(CUdriverProcAddressQueryResult)) < 0)
    {
        return CUDA_ERROR_UNKNOWN;
    }

    std::string symbolName(symbol);
    auto it = cuFunctionMap.find(symbolName);
    if (it != cuFunctionMap.end()) {
        *pfn = reinterpret_cast<void *>(it->second);
        std::cout << "Mapped symbol: " << symbolName << " to function: " << *pfn << std::endl;
    } else {
        void *fn = nullptr;
        std::cerr << "Function for symbol: " << symbolName << " not found!" << std::endl;
        *pfn = reinterpret_cast<void *>(noOpFunction); 
    }

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

    std::cout << "Functionnnnnn: " << name << std::endl;

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
