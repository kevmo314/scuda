
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
        freeaddrinfo(res);
        return -1;
    }

    // Set the socket to non-blocking mode
    if (fcntl(sockfd, F_SETFL, O_NONBLOCK) == -1) {
        printf("Failed to set socket to non-blocking mode: %s\n", strerror(errno));
        close(sockfd);
        freeaddrinfo(res);
        return -1;
    }

    // Attempt to connect (non-blocking)
    int ret = connect(sockfd, res->ai_addr, res->ai_addrlen);
    if (ret == -1 && errno != EINPROGRESS) {
        // Immediate failure, not just "in progress"
        printf("connection with the server failed immediately: %s\n", strerror(errno));
        close(sockfd);
        freeaddrinfo(res);
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
            printf("poll() error during connect: %s\n", strerror(errno));
            close(sockfd);
            freeaddrinfo(res);
            return -1;
        } else if (ret == 0) {
            printf("Connection timed out...\n");
            close(sockfd);
            freeaddrinfo(res);
            return -1;
        }

        // Check for socket errors after poll() indicates readiness
        int so_error;
        socklen_t len = sizeof(so_error);
        if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &so_error, &len) == -1) {
            printf("getsockopt failed: %s\n", strerror(errno));
            close(sockfd);
            freeaddrinfo(res);
            return -1;
        }
        if (so_error != 0) {
            printf("Connection failed: %s\n", strerror(so_error));
            close(sockfd);
            freeaddrinfo(res);
            return -1;
        }
    }

    freeaddrinfo(res);
    printf("Connected successfully!\n");
    return sockfd;
}

pthread_mutex_t mutex;
pthread_cond_t cond;

int rpc_start_request(const unsigned int op)
{
    static int next_request_id = 1;

    open_rpc_client();

    pthread_mutex_lock(&mutex);

    int request_id = next_request_id++;

    std::cout << "[Client] Sending request ID: " << request_id << ", operation: " << op << std::endl;

    // Write the request ID to the socket with partial write handling
    ssize_t total_written = 0;
    ssize_t bytes_to_write = sizeof(request_id);
    while (total_written < bytes_to_write) {
        ssize_t n = write(sockfd, ((char*)&request_id) + total_written, bytes_to_write - total_written);
        if (n < 0) {
            std::cerr << "[Client] Failed to write request ID: " << strerror(errno) << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }
        total_written += n;
    }

    // Write the operation code to the socket with partial write handling
    total_written = 0;
    bytes_to_write = sizeof(op);
    while (total_written < bytes_to_write) {
        ssize_t n = write(sockfd, ((char*)&op) + total_written, bytes_to_write - total_written);
        if (n < 0) {
            std::cerr << "[Client] Failed to write operation: " << strerror(errno) << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }
        total_written += n;
    }

    pthread_mutex_unlock(&mutex);
    return request_id;
}

int rpc_write(const void *data, const size_t size)
{
    fd_set writefds;
    FD_ZERO(&writefds);
    FD_SET(sockfd, &writefds);

    struct timeval timeout;
    timeout.tv_sec = 5;  // 5 seconds timeout
    timeout.tv_usec = 0;

    pthread_mutex_lock(&mutex); // Lock before writing

    size_t total_written = 0;
    ssize_t n;

    while (total_written < size) {
        // Check if the socket is ready for writing using select
        int ret = select(sockfd + 1, NULL, &writefds, NULL, &timeout);
        if (ret < 0 || ret == 0 || !FD_ISSET(sockfd, &writefds)) {
            std::cerr << "rpc_write: Timeout or error." << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }

        // Try writing the remaining data
        n = write(sockfd, (char*)data + total_written, size - total_written);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // The socket is not ready, continue looping
                continue;
            } else {
                std::cerr << "rpc_write: Write error: " << strerror(errno) << std::endl;
                pthread_mutex_unlock(&mutex);
                return -1;
            }
        }

        // Accumulate the total written bytes
        total_written += n;
    }

    pthread_mutex_unlock(&mutex); // Unlock after writing
    return 0;
}

int rpc_end_request(void *result, const unsigned int request_id)
{
    pthread_mutex_lock(&mutex); // Lock before reading result

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);

    struct timeval timeout;
    timeout.tv_sec = 5;  // 5 seconds timeout
    timeout.tv_usec = 0;

    ssize_t total_read = 0;
    ssize_t bytes_to_read = sizeof(nvmlReturn_t);

    while (total_read < bytes_to_read) {
        // Use select to check if socket is ready for reading
        int ret = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
        if (ret < 0 || ret == 0 || !FD_ISSET(sockfd, &readfds)) {
            std::cerr << "rpc_end_request: Timeout or error while waiting for socket to be readable." << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }

        // Attempt to read the result from the socket
        ssize_t n = read(sockfd, (char*)result + total_read, bytes_to_read - total_read);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Socket is not ready yet, continue looping
                continue;
            } else {
                std::cerr << "rpc_end_request: Read error: " << strerror(errno) << std::endl;
                pthread_mutex_unlock(&mutex);
                return -1;
            }
        } else if (n == 0) {
            std::cerr << "rpc_end_request: Server closed the connection." << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }

        total_read += n;
    }

    pthread_mutex_unlock(&mutex); // Unlock after reading result
    return 0;
}

int rpc_read(void *data, size_t size)
{
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);

    struct timeval timeout;
    timeout.tv_sec = 5;  // 5 seconds timeout
    timeout.tv_usec = 0;

    pthread_mutex_lock(&mutex);  // Lock before reading

    size_t total_read = 0;
    ssize_t n;

    while (total_read < size) {
        // Use select to check if socket is ready for reading
        int ret = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
        if (ret < 0 || ret == 0 || !FD_ISSET(sockfd, &readfds)) {
            std::cerr << "rpc_read: Timeout or error while waiting for socket to be readable." << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }

        // Try reading the remaining data from the socket
        n = read(sockfd, (char*)data + total_read, size - total_read);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // The socket is not ready yet, continue looping
                continue;
            } else {
                std::cerr << "rpc_read: Read error: " << strerror(errno) << std::endl;
                pthread_mutex_unlock(&mutex);
                return -1;
            }
        } else if (n == 0) {
            std::cerr << "rpc_read: Server closed the connection." << std::endl;
            pthread_mutex_unlock(&mutex);
            return -1;
        }

        total_read += n;
    }

    pthread_mutex_unlock(&mutex);  // Unlock after reading
    return 0;
}

int rpc_wait_for_response(const unsigned int request_id)
{
    fd_set readfds;
    struct timeval timeout;
    int retval;
    int response_id = -1;

    // Timeout for select() to wait for 5 seconds
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;

    while (true)
    {
        // Setup file descriptor set
        FD_ZERO(&readfds);
        FD_SET(sockfd, &readfds);

        // Wait for the socket to become readable
        retval = select(sockfd + 1, &readfds, NULL, NULL, &timeout);

        if (retval == -1) // Error during select()
        {
            std::cerr << "Error during select: " << strerror(errno) << std::endl;
            return -1;
        }
        else if (retval == 0) // Timeout occurred
        {
            std::cerr << "Timeout waiting for response for request ID: " << request_id << std::endl;
            return -1;
        }

        if (FD_ISSET(sockfd, &readfds)) // Socket is ready for reading
        {
            pthread_mutex_lock(&mutex);  // Lock the mutex before reading and condition check

            // Read the response ID from the socket
            ssize_t bytes_read = read(sockfd, &response_id, sizeof(int));

            if (bytes_read == 0) // Connection closed by the server
            {
                std::cerr << "Server closed the connection." << std::endl;
                pthread_mutex_unlock(&mutex);
                return -1;
            }
            else if (bytes_read < 0) // Error reading the response ID
            {
                std::cerr << "Error reading from socket: " << strerror(errno) << std::endl;
                pthread_mutex_unlock(&mutex);
                return -1;
            }

            // Debug output to verify response ID read
            std::cout << "Received response ID: " << response_id << std::endl;

            // while (response_id != request_id && response_id != -1)
            //     pthread_cond_wait(&cond, &mutex);

            // Check if this is the expected response for the current request
            if (response_id == request_id)
            {
                pthread_cond_broadcast(&cond);  // Notify other threads waiting on responses
                pthread_mutex_unlock(&mutex);   // Unlock the mutex after processing
                return 0;  // Successfully received the correct response
            }
            else
            {
                // Unexpected response ID; continue waiting
                std::cerr << "Unexpected response ID: " << response_id << ", expected: " << request_id << std::endl;
                pthread_cond_broadcast(&cond);  // Notify others, in case they are waiting for this ID
                pthread_mutex_unlock(&mutex);   // Unlock before retrying
            }
        }
    }

    return 0;  // Should not reach here unless something goes wrong
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
