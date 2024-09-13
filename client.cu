#include <arpa/inet.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <nvml.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>

#include "api.h"

int sockfd;

int open_rpc_client()
{
    struct sockaddr_in servaddr;

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

    printf("Sending request %d\n", request_id);
    if (write(sockfd, &request_id, sizeof(int)) < 0 ||
        write(sockfd, &op, sizeof(unsigned int)) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return NVML_ERROR_GPU_IS_LOST;
    }

    printf("Sending %lu requests\n", requests.size());

    for (auto r : requests)
        if (write(sockfd, r.first, r.second) < 0)
        {
            pthread_mutex_unlock(&mutex);
            return NVML_ERROR_GPU_IS_LOST;
        }

    // wait for the response
    while (true)
    {
        printf("Waiting for response %d\n", request_id);
        while (active_response_id != request_id && active_response_id != -1)
            pthread_cond_wait(&cond, &mutex);

        printf("Got response active %d\n", active_response_id);

        // we currently own mutex. if active response id is -1, read the response id
        if (active_response_id == -1)
        {
            if (read(sockfd, &active_response_id, sizeof(int)) < 0)
            {
                pthread_mutex_unlock(&mutex);
                return NVML_ERROR_GPU_IS_LOST;
            }

            printf("Got response id %d\n", active_response_id);

            if (active_response_id != request_id)
            {
                pthread_cond_broadcast(&cond);
                continue;
            }
        }

        active_response_id = -1;

        printf("Reading response %d\n", request_id);

        // it's our turn to read the response.
        nvmlReturn_t ret;
        if (read(sockfd, &ret, sizeof(nvmlReturn_t)) < 0 || ret != NVML_SUCCESS)
        {
            pthread_mutex_unlock(&mutex);
            return NVML_ERROR_GPU_IS_LOST;
        }

        printf("Reading %lu responses\n", responses.size());

        for (auto r : responses)
            if (read(sockfd, r.first, r.second) < 0)
            {
                pthread_mutex_unlock(&mutex);
                return NVML_ERROR_GPU_IS_LOST;
            }

        printf("done!\n");

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

nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    open_rpc_client();
    return send_rpc_message(RPC_nvmlInitWithFlags, {{&flags, sizeof(unsigned int)}});
}

nvmlReturn_t nvmlInit_v2()
{
    open_rpc_client();
    return send_rpc_message(RPC_nvmlInit_v2);
}

nvmlReturn_t nvmlShutdown()
{
    open_rpc_client();
    return send_rpc_message(RPC_nvmlShutdown);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    open_rpc_client();
    return send_rpc_message(RPC_nvmlDeviceGetName, {{&device, sizeof(nvmlDevice_t)}, {&length, sizeof(int)}}, {{name, length}});
}

void *dlsym(void *handle, const char *name) __THROW
{
    printf("Resolving symbol: %s\n", name);

    if (!strcmp(name, "nvmlInitWithFlags"))
        return (void *)nvmlInitWithFlags;
    // if (!strcmp(name, "nvmlInit_v2"))
    //     return (void *)nvmlInit_v2;
    // if (!strcmp(name, "nvmlShutdown"))
    //     return (void *)nvmlShutdown;
    // if (!strcmp(name, "nvmlDeviceGetName"))
    //     return (void *)nvmlDeviceGetName;

    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    /* my target binary is even asking for dlsym() via dlsym()... */
    if (!strcmp(name, "dlsym"))
        return (void *)dlsym;
    return real_dlsym(handle, name);
}
