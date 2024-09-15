#include <arpa/inet.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <nvml.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include "api.h"

int sockfd;

int open_rpc_client()
{
    struct sockaddr_in servaddr;

    char *server_ip = getenv("SCUDA_SERVER");
    if (server_ip == NULL)
    {
        printf("SCUDA_SERVER environment variable not set\n");
        return -1;
    }

    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(server_ip, "14833", &hints, &res) != 0)
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

// TODO: can this be done via a template?
nvmlReturn_t send_rpc_message(
    const unsigned int op,
    std::vector<std::pair<const void *, const int>> requests = {},
    std::vector<std::pair<void *, const int>> responses = {})
{
    static int next_request_id = 0, active_response_id = -1;
    static pthread_mutex_t mutex;
    static pthread_cond_t cond;

    // write the request atomically
    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++;

    if (write(sockfd, &request_id, sizeof(int)) < 0 ||
        write(sockfd, &op, sizeof(unsigned int)) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return NVML_ERROR_GPU_IS_LOST;
    }

    for (auto r : requests)
        if (write(sockfd, r.first, r.second) < 0)
        {
            pthread_mutex_unlock(&mutex);
            return NVML_ERROR_GPU_IS_LOST;
        }

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
                return NVML_ERROR_GPU_IS_LOST;
            }

            if (active_response_id != request_id)
            {
                pthread_cond_broadcast(&cond);
                continue;
            }
        }

        active_response_id = -1;

        // it's our turn to read the response.
        for (auto r : responses)
            if (read(sockfd, r.first, r.second) < 0)
            {
                pthread_mutex_unlock(&mutex);
                return NVML_ERROR_GPU_IS_LOST;
            }

        nvmlReturn_t ret;
        if (read(sockfd, &ret, sizeof(nvmlReturn_t)) < 0)
            ret = NVML_ERROR_GPU_IS_LOST;

        // we are done, unlock and return.
        pthread_mutex_unlock(&mutex);
        return ret;
    }
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
    return send_rpc_message(RPC_nvmlInitWithFlags, {{&flags, sizeof(unsigned int)}});
}

nvmlReturn_t nvmlInit_v2()
{
    if (open_rpc_client() < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return send_rpc_message(RPC_nvmlInit_v2);
}

nvmlReturn_t nvmlShutdown()
{
    nvmlReturn_t result = send_rpc_message(RPC_nvmlShutdown);
    close_rpc_client();
    return result;
}

// 4.14 System Queries
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length)
{
    return send_rpc_message(RPC_nvmlSystemGetDriverVersion, {}, {{version, length}});
}

nvmlReturn_t nvmlSystemGetDriverVersion_v2(char *version, unsigned int length)
{
    return send_rpc_message(RPC_nvmlSystemGetDriverVersion_v2, {}, {{version, length}});
}

nvmlReturn_t nvmlSystemGetDriverBranch(char *branch, unsigned int length)
{
    return send_rpc_message(RPC_nvmlSystemGetDriverBranch, {}, {{branch, length}});
}

nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount, nvmlHwbcEntry_t *hwbcEntries)
{
    return send_rpc_message(RPC_nvmlSystemGetHicVersion, {}, {{hwbcCount, sizeof(unsigned int)}, {hwbcEntries, sizeof(nvmlHwbcEntry_t)}});
}

nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length)
{
    return send_rpc_message(RPC_nvmlSystemGetNVMLVersion, {}, {{version, length}});
}

nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length)
{
    return send_rpc_message(RPC_nvmlSystemGetProcessName, {{&pid, sizeof(unsigned int)}}, {{name, length}});
}

nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, nvmlDevice_t *deviceArray)
{
    return send_rpc_message(RPC_nvmlSystemGetTopologyGpuSet, {{&cpuNumber, sizeof(unsigned int)}}, {{count, sizeof(unsigned int)}, {deviceArray, sizeof(nvmlDevice_t)}});
}

// 4.15 Unit Queries
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount)
{
    return send_rpc_message(RPC_nvmlUnitGetCount, {}, {{unitCount, sizeof(unsigned int)}});
}

nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount, nvmlDevice_t *devices)
{
    return send_rpc_message(RPC_nvmlUnitGetDevices, {{&unit, sizeof(nvmlUnit_t)}}, {{deviceCount, sizeof(unsigned int)}, {devices, sizeof(nvmlDevice_t)}});
}

nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds)
{
    return send_rpc_message(RPC_nvmlUnitGetFanSpeedInfo, {{&unit, sizeof(nvmlUnit_t)}}, {{fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)}});
}

nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit)
{
    return send_rpc_message(RPC_nvmlUnitGetHandleByIndex, {{&index, sizeof(unsigned int)}}, {{unit, sizeof(nvmlUnit_t)}});
}

nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state)
{
    return send_rpc_message(RPC_nvmlUnitGetLedState, {{&unit, sizeof(nvmlUnit_t)}}, {{state, sizeof(nvmlLedState_t)}});
}

nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu)
{
    return send_rpc_message(RPC_nvmlUnitGetPsuInfo, {{&unit, sizeof(nvmlUnit_t)}}, {{psu, sizeof(nvmlPSUInfo_t)}});
}

nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int *temp)
{
    return send_rpc_message(RPC_nvmlUnitGetTemperature, {{&unit, sizeof(nvmlUnit_t)}, {&type, sizeof(unsigned int)}}, {{temp, sizeof(unsigned int)}});
}

nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info)
{
    return send_rpc_message(RPC_nvmlUnitGetUnitInfo, {{&unit, sizeof(nvmlUnit_t)}}, {{info, sizeof(nvmlUnitInfo_t)}});
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    return send_rpc_message(RPC_nvmlDeviceGetCount_v2, {}, {{deviceCount, sizeof(unsigned int)}});
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    return send_rpc_message(RPC_nvmlDeviceGetName, {{&device, sizeof(nvmlDevice_t)}, {&length, sizeof(int)}}, {{name, length}});
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    return send_rpc_message(RPC_nvmlDeviceGetHandleByIndex_v2, {{&index, sizeof(unsigned int)}}, {{device, sizeof(nvmlDevice_t)}});
}

void *dlsym(void *handle, const char *name) __THROW
{
    printf("Resolving symbol: %s\n", name);

    // 4.11 Initialization and Cleanup
    if (!strcmp(name, "nvmlInitWithFlags"))
        return (void *)nvmlInitWithFlags;
    if (!strcmp(name, "nvmlInit_v2"))
        return (void *)nvmlInit_v2;
    if (!strcmp(name, "nvmlShutdown"))
        return (void *)nvmlShutdown;

    // 4.14 System Queries
    if (!strcmp(name, "nvmlSystemGetDriverVersion"))
        return (void *)nvmlSystemGetDriverVersion;
    if (!strcmp(name, "nvmlSystemGetDriverVersion_v2"))
        return (void *)nvmlSystemGetDriverVersion_v2;
    if (!strcmp(name, "nvmlSystemGetDriverBranch"))
        return (void *)nvmlSystemGetDriverBranch;
    if (!strcmp(name, "nvmlSystemGetHicVersion"))
        return (void *)nvmlSystemGetHicVersion;
    if (!strcmp(name, "nvmlSystemGetNVMLVersion"))
        return (void *)nvmlSystemGetNVMLVersion;
    if (!strcmp(name, "nvmlSystemGetProcessName"))
        return (void *)nvmlSystemGetProcessName;
    if (!strcmp(name, "nvmlSystemGetTopologyGpuSet"))
        return (void *)nvmlSystemGetTopologyGpuSet;

    // 4.15 Unit Queries
    if (!strcmp(name, "nvmlUnitGetCount"))
        return (void *)nvmlUnitGetCount;
    if (!strcmp(name, "nvmlUnitGetDevices"))
        return (void *)nvmlUnitGetDevices;
    if (!strcmp(name, "nvmlUnitGetFanSpeedInfo"))
        return (void *)nvmlUnitGetFanSpeedInfo;
    if (!strcmp(name, "nvmlUnitGetHandleByIndex"))
        return (void *)nvmlUnitGetHandleByIndex;
    if (!strcmp(name, "nvmlUnitGetLedState"))
        return (void *)nvmlUnitGetLedState;
    if (!strcmp(name, "nvmlUnitGetPsuInfo"))
        return (void *)nvmlUnitGetPsuInfo;
    if (!strcmp(name, "nvmlUnitGetTemperature"))
        return (void *)nvmlUnitGetTemperature;
    if (!strcmp(name, "nvmlUnitGetUnitInfo"))
        return (void *)nvmlUnitGetUnitInfo;

    if (!strcmp(name, "nvmlDeviceGetCount_v2"))
        return (void *)nvmlDeviceGetCount_v2;
    if (!strcmp(name, "nvmlDeviceGetName"))
        return (void *)nvmlDeviceGetName;
    if (!strcmp(name, "nvmlDeviceGetHandleByIndex_v2"))
        return (void *)nvmlDeviceGetHandleByIndex_v2;

    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    /* my target binary is even asking for dlsym() via dlsym()... */
    if (!strcmp(name, "dlsym"))
        return (void *)dlsym;
    return real_dlsym(handle, name);
}
