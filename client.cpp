
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
#include <cuda.h>
#include <sys/uio.h>

#include <unordered_map>

#include "codegen/gen_client.h"

int sockfd = -1;
char *port;

int open_rpc_client()
{
    // if socket is already opened, return our socket.
    if (sockfd != -1)
    {
        return sockfd;
    }

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

struct iovec write_iov[128];
int write_iov_count = 0;

int rpc_start_request(const unsigned int op)
{
    static int next_request_id = 1;

    open_rpc_client();

    // write the request atomically
    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++;

    write_iov_count = 0;
    write_iov[write_iov_count++] = {&request_id, sizeof(int)};
    write_iov[write_iov_count++] = {&op, sizeof(unsigned int)};
    return request_id;
}

int rpc_write(const void *data, const size_t size)
{
    write_iov[write_iov_count++] = {const_cast<void *>(data), size};
    return 0;
}

int rpc_end_request(void *result, const unsigned int request_id)
{
    if (read(sockfd, result, sizeof(nvmlReturn_t)) < 0)
        return -1;

    pthread_mutex_unlock(&mutex);
    return 0;
}

int rpc_read(void *data, size_t size)
{
    if (data == nullptr)
    {
        // temp buffer to discard data
        char tempBuffer[256];
        while (size > 0)
        {
            ssize_t bytesRead = read(sockfd, tempBuffer, std::min(size, sizeof(tempBuffer)));
            if (bytesRead < 0)
            {
                pthread_mutex_unlock(&mutex);
                return -1; // error if reading fails
            }
            size -= bytesRead;
        }
    }
    else if (read(sockfd, data, size) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    return 0;
}

int rpc_wait_for_response(const unsigned int request_id)
{
    static int active_response_id = -1;

    // write the request to the server
    if (writev(sockfd, write_iov, write_iov_count) < 0)
    {
        pthread_mutex_unlock(&mutex);
        return -1;
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

void close_rpc_client()
{
    close(sockfd);
    sockfd = 0;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)
{
    std::cout << "cuGetProcAddress getting symbol: " << symbol << std::endl;

    auto it = get_function_pointer(symbol);
    if (it != nullptr)
    {
        *pfn = (void *)(&it);
        std::cout << "cuGetProcAddress: Mapped symbol '" << symbol << "' to function: " << *pfn << std::endl;
        return CUDA_SUCCESS;
    }

    if (strcmp(symbol, "cuGetProcAddress_v2") == 0 || strcmp(symbol, "cuGetProcAddress") == 0)
    {
        *pfn = (void *)&cuGetProcAddress_v2;
        return CUDA_SUCCESS;
    }

    std::cout << "cuGetProcAddress: Symbol '" << symbol << "' not found in cudaFunctionMap." << std::endl;

    // fall back to dlsym
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
    {
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!libCudaHandle)
    {
        std::cerr << "Error: Failed to open libcuda.so" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    *pfn = real_dlsym(libCudaHandle, symbol);
    if (!(*pfn))
    {
        std::cerr << "Error: Could not resolve symbol '" << symbol << "' using dlsym." << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    return CUDA_SUCCESS;
}

void *dlsym(void *handle, const char *name) __THROW
{
    open_rpc_client();

    void *func = get_function_pointer(name);

    /** proc address function calls are basically dlsym; we should handle this differently at the top level. */
    if (strcmp(name, "cuGetProcAddress_v2") == 0 || strcmp(name, "cuGetProcAddress") == 0)
    {
        return (void *)&cuGetProcAddress_v2;
    }

    if (func != nullptr)
    {
        std::cout << "[dlsym] Function address from cudaFunctionMap: " << func << " " << name << std::endl;
        return func;
    }

    // Real dlsym lookup
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
    {
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    std::cout << "[dlsym] Falling back to real_dlsym for name: " << name << std::endl;
    return real_dlsym(handle, name);
}
