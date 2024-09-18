#include <arpa/inet.h>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <netdb.h>
#include <nvml.h>
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
        std::cout << "socket already opened" << std::endl;

        return sockfd;
    }

    std::cout << "opening tcp socket..." << std::endl;

    char *server_ip = getenv("SCUDA_SERVER");
    if (server_ip == NULL)
    {
        printf("SCUDA_SERVER environment variable not set\n");
        std::exit(1);
    }

    char *p = getenv("SCUDA_PORT");

    if (p == NULL)
    {
        std::cout << "SCUDA_PORT not defined, defaulting to: " << "14833"
                  << std::endl;
        port = (char *)"14833";
    }
    else
    {
        port = p;
        std::cout << "using SCUDA_PORT: " << port << std::endl;
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

    // write the request atomically
    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++;

    if (write(sockfd, &request_id, sizeof(int)) < 0 ||
        write(sockfd, &op, sizeof(unsigned int)) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return -1;
    }

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

nvmlReturn_t rpc_get_return(int request_id)
{
    nvmlReturn_t result;
    if (read(sockfd, &result, sizeof(nvmlReturn_t)) < 0)
        result = NVML_ERROR_GPU_IS_LOST;

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
    if (request_id < 0 || rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlInit_v2()
{
    if (open_rpc_client() < 0)
        return NVML_ERROR_GPU_IS_LOST;
    int request_id = rpc_start_request(RPC_nvmlInit_v2);
    if (request_id < 0 || rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlShutdown()
{
    int request_id = rpc_start_request(RPC_nvmlShutdown);
    if (request_id < 0 || rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    nvmlReturn_t result = rpc_get_return(request_id);
    close_rpc_client();
    return result;
}

// 4.14 System Queries
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    if (request_id < 0 || rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(version, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount,
                                     nvmlHwbcEntry_t *hwbcEntries)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    if (request_id < 0 || rpc_write(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_read(hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    nvmlReturn_t result = rpc_get_return(request_id);
    close_rpc_client();
    return result;
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetNVMLVersion);
    if (request_id < 0 || rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(version, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name,
                                      unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetProcessName);
    if (request_id < 0 || rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(name, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber,
                                         unsigned int *count,
                                         nvmlDevice_t *deviceArray)
{
    int request_id = rpc_start_request(RPC_nvmlSystemGetTopologyGpuSet);
    if (request_id < 0 || rpc_write(&cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

// 4.15 Unit Queries
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetCount);
    if (request_id < 0 || rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unitCount, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount,
                                nvmlDevice_t *devices)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetDevices);
    if (request_id < 0 || rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_read(devices, *deviceCount * sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit,
                                     nvmlUnitFanSpeeds_t *fanSpeeds)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetFanSpeedInfo);
    if (request_id < 0 || rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetHandleByIndex);
    if (request_id < 0 || rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unit, sizeof(nvmlUnit_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetLedState);
    if (request_id < 0 || rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(state, sizeof(nvmlLedState_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetPsuInfo);
    if (request_id < 0 || rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(psu, sizeof(nvmlPSUInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type,
                                    unsigned int *temp)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetTemperature);
    if (request_id < 0 || rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&type, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info)
{
    int request_id = rpc_start_request(RPC_nvmlUnitGetUnitInfo);
    if (request_id < 0 || rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlUnitInfo_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
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
    return rpc_get_return(request_id);
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
    return rpc_get_return(request_id);
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
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetCreate);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(set, sizeof(nvmlEventSet_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set)
{
    int request_id = rpc_start_request(RPC_nvmlEventSetFree);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
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
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCount_v2);
    if (request_id < 0 || rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                               unsigned int length)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetName);
    if (request_id < 0 || rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 || rpc_read(name, length) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                           nvmlDevice_t *device)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    if (request_id < 0 || rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return rpc_get_return(request_id);
}

// 4.16 Device Queries
nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device,
                                        nvmlMemory_v2_t *memoryInfo)
{
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo_v2);
    if (request_id < 0 || rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memoryInfo, sizeof(nvmlMemory_v2_t)) < 0)
    {
        std::cerr << "Failed to start RPC request" << std::endl;
        return NVML_ERROR_GPU_IS_LOST;
    }

    return rpc_get_return(request_id);
}

std::unordered_map<std::string, void *> functionMap;

void initializeFunctionMap()
{
    // simple cache check to make sure we only init handlers on the first run
    if (functionMap.find("nvmlInit_v2") != functionMap.end())
    {
        std::cout << "handlers already initialized" << std::endl;
    }
    else
    {
        std::cout << "initializing handlers" << std::endl;

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
        functionMap["nvmlDeviceGetCount_v2"] = (void *)nvmlDeviceGetCount_v2;
        functionMap["nvmlDeviceGetName"] = (void *)nvmlDeviceGetName;
        functionMap["nvmlDeviceGetHandleByIndex_v2"] =
            (void *)nvmlDeviceGetHandleByIndex_v2;
        functionMap["nvmlDeviceGetMemoryInfo_v2"] =
            (void *)nvmlDeviceGetMemoryInfo_v2;
    }
}

// Lookup function similar to dlsym
void *getFunctionByName(const char *name)
{
    auto it = functionMap.find(name);

    if (it != functionMap.end())
    {
        return it->second;
    }
    else
    {
        std::cout << "handler not found: " << name << std::endl;
        return nullptr;
    }
}

void *dlsym(void *handle, const char *name) __THROW
{
    initializeFunctionMap();

    std::cout << "Resolving function symbol: " << name << std::endl;

    void *func = getFunctionByName(name);

    if (func != nullptr)
    {
        return func;
    }

    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
    {
        // avoid calling dlsym recursively; use dlvsym to resolve dlsym itself
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                             "GLIBC_2.2.5");
    }

    if (!strcmp(name, "dlsym"))
    {
        return (void *)dlsym;
    }

    // if func symbol is not found in the handler mappings, return the real dlsym
    // resolution
    return real_dlsym(handle, name);
}
