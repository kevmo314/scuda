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

int sockfd;

void nvmlInitWithFlagsHandler(int connfd, unsigned int flags)
{
    nvmlReturn_t result = nvmlInitWithFlags(flags);
    write(connfd, &result, sizeof(result));
}

void nvmlShutdownHandler(int connfd)
{
    nvmlReturn_t result = nvmlShutdown();
    write(connfd, &result, sizeof(result));
}

void nvmlDeviceGetNameHandler(int connfd, nvmlDevice_t device)
{
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlReturn_t result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    write(connfd, &result, sizeof(result));
    if (result == NVML_SUCCESS)
    {
        int len = strlen(name) + 1; // including null terminator
        write(connfd, &len, sizeof(len));
        write(connfd, name, len);
    }
}

void *client_handler(void *arg)
{
    int connfd = *(int *)arg;
    free(arg);

    unsigned int op;
    int request_id;
    int argslen;

    while (read(connfd, &request_id, sizeof(int)) >= 0)
    {
        if (read(connfd, &op, sizeof(unsigned int)) < 0)
            goto exit;

        if (write(connfd, &request_id, sizeof(int)) < 0)
            goto exit;

        // Handle different NVML operations
        switch (op)
        {
        case RPC_nvmlInitWithFlags:
        {
            unsigned int flags;
            if (read(connfd, &flags, sizeof(unsigned int)) < 0)
                goto exit;
            printf("Received nvmlInitWithFlags request %d %d\n", request_id, flags);
            nvmlReturn_t result = nvmlInitWithFlags(flags);
            printf("Sending nvmlInitWithFlags response %d %d\n", request_id, result);
            if (write(connfd, &result, sizeof(nvmlReturn_t)) < 0)
                goto exit;
            break;
        }
        }
    }

exit:
    close(connfd);
    pthread_exit(NULL);
}

int main()
{
    struct sockaddr_in servaddr, cli;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
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

        pthread_t thread_id;
        pthread_create(&thread_id, NULL, client_handler, connfd);
    }

    close(sockfd);
    return 0;
}
