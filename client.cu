#define _GNU_SOURCE

#include <arpa/inet.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <nvml.h>
#include <unistd.h>
#include <pthread.h>

int sockfd;

int open_rpc_client()
{
    int connfd;
    struct sockaddr_in servaddr, cli;

    if (sockfd != 0)
    {
        return sockfd;
    }

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("socket creation failed...\n");
        exit(0);
    }

    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    servaddr.sin_port = htons(14833);
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0)
    {
        printf("connection with the server failed...\n");
        exit(0);
    }
    return sockfd;
}

nvmlReturn_t send_rpc_message(void **response, int *len, const char *op, const void *args, const int argslen)
{
    static int next_request_id = 0, active_response_id = -1;
    static pthread_mutex_t mutex;
    static pthread_cond_t cond;

    // write the request atomically
    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++;

    uint8_t oplen = (uint8_t)strlen(op);
    if (write(sockfd, (void *)&request_id, sizeof(int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;

    if (write(sockfd, (void *)&oplen, sizeof(uint8_t)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    if (write(sockfd, op, oplen) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    if (write(sockfd, (void *)&argslen, sizeof(int)) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    if (write(sockfd, args, argslen) < 0)
        return NVML_ERROR_GPU_IS_LOST;

    // wait for the response
    while (true)
    {
        while (active_response_id != request_id && active_response_id != -1)
            pthread_cond_wait(&cond, &mutex);

        // we currently own mutex. if active response id is -1, read the response id
        if (active_response_id == -1)
        {
            if (read(sockfd, (void *)&active_response_id, sizeof(int)) < 0)
                return NVML_ERROR_GPU_IS_LOST;
            continue;
        }
        else
        {
            // it's our turn to read the response.
            nvmlReturn_t ret;
            if (read(sockfd, (void *)&ret, sizeof(nvmlReturn_t)) < 0)
                return NVML_ERROR_GPU_IS_LOST;
            if (ret != NVML_SUCCESS || response == NULL)
            {
                pthread_mutex_unlock(&mutex);
                return ret;
            }

            if (read(sockfd, (void *)len, sizeof(int)) < 0)
                return NVML_ERROR_GPU_IS_LOST;
            if (len > 0)
            {
                *response = malloc(*len);
                if (read(sockfd, *response, *len) < 0)
                    return NVML_ERROR_GPU_IS_LOST;
            }

            // we are done, unlock and return.
            pthread_mutex_unlock(&mutex);
            return ret;
        }
    }
}

void close_rpc_client()
{
    close(sockfd);
    sockfd = 0;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    open_rpc_client();
    return send_rpc_message(NULL, NULL, "nvmlInitWithFlags", (void *)&flags, sizeof(unsigned int));
}

nvmlReturn_t nvmlInit_v2()
{
    open_rpc_client();
    return send_rpc_message(NULL, NULL, "nvmlInit_v2", NULL, 0);
}

nvmlReturn_t nvmlShutdown()
{
    open_rpc_client();
    return send_rpc_message(NULL, NULL, "nvmlShutdown", NULL, 0);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    open_rpc_client();
    return send_rpc_message((void **)&name, (int *)&length, "nvmlDeviceGetName", (void *)&device, sizeof(nvmlDevice_t));
}

void *dlsym(void *handle, const char *name)
{
    printf("Resolving symbol: %s\n", name);

    if (!strcmp(name, "nvmlInitWithFlags"))
        return (void *)nvmlInitWithFlags;
    if (!strcmp(name, "nvmlInit_v2"))
        return (void *)nvmlInit_v2;
    if (!strcmp(name, "nvmlShutdown"))
        return (void *)nvmlShutdown;
    if (!strcmp(name, "nvmlDeviceGetName"))
        return (void *)nvmlDeviceGetName;

    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    /* my target binary is even asking for dlsym() via dlsym()... */
    if (!strcmp(name, "dlsym"))
        return (void *)dlsym;
    return real_dlsym(handle, name);
}
