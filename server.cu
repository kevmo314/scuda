#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <nvml.h>
#include <sys/socket.h>

#include "api.h"

#define PORT 14833
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
    case RPC_nvmlDeviceGetCount_v2:
    {
        unsigned int deviceCount = 0;
        nvmlReturn_t result = nvmlDeviceGetCount_v2(&deviceCount);
        if (write(connfd, &deviceCount, sizeof(unsigned int)) < 0)
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
    default:
        printf("Unknown operation %d\n", op);
        break;
    }
    return -1;
}

void *client_handler(void *arg)
{
    int connfd = *(int *)arg;
    free(arg);

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

        if (write(connfd, &request_id, sizeof(int)) < 0) {
            printf("error writing to client.\n");
            break;
        }

        int res = request_handler(connfd);
        if (write(connfd, &res, sizeof(int)) < 0) {
            printf("error writing result to client.\n");
            break;
        }
    }

    close(connfd);
    // dont need pthread_exit, can return when complete instead
    return NULL;
}

int main()
{
    struct sockaddr_in servaddr, cli;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("Socket creation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Bind the socket
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

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

    // Listen for clients
    if (listen(sockfd, MAX_CLIENTS) != 0)
    {
        printf("Listen failed.\n");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", PORT);

    // Server loop
    while (1)
    {
        socklen_t len = sizeof(cli);
        int *connfd = (int *)malloc(sizeof(int));
        *connfd = accept(sockfd, (struct sockaddr *)&cli, &len);
        if (*connfd < 0)
        {
            printf("Server accept failed.\n");
            free(connfd);
            continue;
        }

        printf("client connected, spawning thread.\n");

        pthread_t thread_id;
        pthread_create(&thread_id, NULL, client_handler, connfd);
    }

    close(sockfd);
    return 0;
}
