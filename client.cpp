
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
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "spdlog/spdlog.h"

#include <unordered_map>

#include "codegen/gen_client.h"

int sockfd = -1;
char *port;

int open_rpc_client() {
    // if socket is already opened, return our socket.
    if (sockfd != -1) {
        return sockfd;
    }

    char *server_ip = getenv("SCUDA_SERVER");
    if (server_ip == NULL) {
        printf("SCUDA_SERVER environment variable not set\n");
        std::exit(1);
    }

    char *p = getenv("SCUDA_PORT");

    if (p == NULL) {
        port = (char *)"14833";
    } else {
        port = p;
    }

    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(server_ip, port, &hints, &res) != 0) {
        printf("getaddrinfo failed\n");
        return -1;
    }

    sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        return -1;
    }

    // Set the socket to non-blocking mode
    fcntl(sockfd, F_SETFL, O_NONBLOCK);

    // Attempt to connect (non-blocking)
    int ret = connect(sockfd, res->ai_addr, res->ai_addrlen);
    if (ret == -1 && errno != EINPROGRESS) {
        // Immediate failure, not just "in progress"
        printf("connection with the server failed immediately...\n");
        return -1;
    }

    // If connect() returned EINPROGRESS, we need to wait for the connection to complete
    if (errno == EINPROGRESS) {
        printf("Connection in progress, waiting for completion...\n");

        struct pollfd fds;
        fds.fd = sockfd;
        fds.events = POLLOUT;  // Waiting for socket to be writable (connected)

        ret = poll(&fds, 1, 5000);  // Wait for 5 seconds
        if (ret == -1) {
            printf("poll() error during connect...\n");
            return -1;
        } else if (ret == 0) {
            printf("Connection timed out...\n");
            return -1;
        }

        // Check for socket errors after poll() indicates readiness
        int so_error;
        socklen_t len = sizeof(so_error);
        getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &so_error, &len);
        if (so_error != 0) {
            printf("Connection failed: %s\n", strerror(so_error));
            return -1;
        }
    }

    printf("Connected successfully!\n");
    return sockfd;
}

pthread_mutex_t mutex;
pthread_cond_t cond;

int rpc_start_request(const unsigned int op)
{
    static int next_request_id = 1;

    open_rpc_client();

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

int rpc_write(const void *data, const size_t size)
{
    fd_set writefds;
    FD_ZERO(&writefds);
    FD_SET(sockfd, &writefds);

    struct timeval timeout;
    timeout.tv_sec = 5;  // 5 seconds timeout for write operation
    timeout.tv_usec = 0;

    std::cout << "writing data..." << std::endl;

    int ret = select(sockfd + 1, NULL, &writefds, NULL, &timeout);
    if (ret < 0) {
        pthread_mutex_unlock(&mutex);
        return -1;
    } else if (ret == 0) {
        std::cerr << "rpc_write: Timeout waiting for socket to be ready for write." << std::endl;
        pthread_mutex_unlock(&mutex);
        return -1;
    }

    if (FD_ISSET(sockfd, &writefds)) {
        if (write(sockfd, data, size) < 0) {
            std::cerr << "FAIL" << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }
    }

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
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);

    // Timeout structure
    struct timeval timeout;
    timeout.tv_sec = 5;  // Wait for up to 5 seconds
    timeout.tv_usec = 0;

    int ret = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
    if (ret < 0) {
        pthread_mutex_unlock(&mutex);
        return -1;
    } else if (ret == 0) {
        // Timeout occurred
        std::cerr << "rpc_read: Timeout waiting for server response." << std::endl;
        pthread_mutex_unlock(&mutex);
        return -1;
    }

    if (FD_ISSET(sockfd, &readfds)) {
        if (data == nullptr) {
            char tempBuffer[256];
            while (size > 0) {
                ssize_t bytesRead = read(sockfd, tempBuffer, std::min(size, sizeof(tempBuffer)));
                if (bytesRead < 0) {
                    pthread_mutex_unlock(&mutex);
                    return -1;
                }
                size -= bytesRead;
            }
        } else if (read(sockfd, data, size) < 0) {
            pthread_mutex_unlock(&mutex);
            return -1;
        }
    }

    return 0;
}

int rpc_wait_for_response(const unsigned int request_id)
{
    fd_set readfds;
    struct timeval timeout;
    int retval;
    int response_id = -1;

    // Initialize timeout to 5 seconds
    timeout.tv_sec = 5; 
    timeout.tv_usec = 0;

    // Set up the file descriptor set
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);

    while (true) {
        // Wait for the socket to become readable using select (multiplexing)
        retval = select(sockfd + 1, &readfds, NULL, NULL, &timeout);

        if (retval == -1) {
            // An error occurred during select
            std::cerr << "Error during select: " << strerror(errno) << std::endl;
            return -1;
        } else if (retval == 0) {
            // Timeout occurred
            std::cerr << "Timeout waiting for response for request ID: " << request_id << std::endl;
            return -1;
        }

        // If socket is ready to be read, proceed
        if (FD_ISSET(sockfd, &readfds)) {
            // Read the response ID from the server
            ssize_t bytes_read = read(sockfd, &response_id, sizeof(int));

            if (bytes_read == 0) {
                std::cerr << "Server closed the connection." << std::endl;
                return -1;
            } else if (bytes_read < 0 && errno == EAGAIN) {
                std::cerr << "Non-blocking read failed: " << strerror(errno) << std::endl;
                return -1;
            } else if (bytes_read < 0) {
                std::cerr << "Error reading from socket: " << strerror(errno) << std::endl;
                return -1;
            }

            // Check if this is the expected response for the current request
            if (response_id != request_id) {
                std::cerr << "Unexpected response ID: " << response_id << ", expected: " << request_id << std::endl;
                return -1;
            }

            // If the response ID matches, break out of the loop
            break;
        }
    }

    return 0;  // Successfully received the correct response
}


void close_rpc_client()
{
    close(sockfd);
    sockfd = 0;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    std::cout << "cuGetProcAddress getting symbol: " << symbol << std::endl;

    auto it = get_function_pointer(symbol);
    if (it != nullptr) {
        *pfn = (void *)(&it);
        std::cout << "cuGetProcAddress: Mapped symbol '" << symbol << "' to function: " << *pfn << std::endl;
        return CUDA_SUCCESS;
    }

    // fall back to dlsym
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL) {
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!libCudaHandle) {
        std::cerr << "Error: Failed to open libcuda.so" << std::endl;
        return CUDA_ERROR_UNKNOWN;
    }

    *pfn = real_dlsym(libCudaHandle, symbol);
    if (!(*pfn)) {
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
        return (void *)&cuGetProcAddress;
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
