
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
#include <netinet/tcp.h>
#include <cuda_runtime.h>

#include <unordered_map>

#include <setjmp.h>
#include <signal.h>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

#include "codegen/gen_client.h"

typedef struct
{
    int connfd;
    int read_request_id;
    int active_response_id;
    int write_request_id;
    unsigned int write_request_op;
    pthread_mutex_t read_mutex, write_mutex;
    pthread_cond_t read_cond;
    struct iovec write_iov[128];
    int write_iov_count = 0;

    std::unordered_map<void*, size_t> unified_devices;
} conn_t;

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;

const char *DEFAULT_PORT = "14833";

static int init = 0;
static jmp_buf catch_segfault;
static void* faulting_address = nullptr;

static void segfault(int sig, siginfo_t* info, void* unused) {
    faulting_address = info->si_addr; 

    std::cout << "Caught segfault at address: " << faulting_address << std::endl;

    for (const auto & [ ptr, sz ] : conns[0].unified_devices)
    {
        if (ptr <= faulting_address && faulting_address < (ptr + sz))
        {
            // ensure we assign memory as close to the faulting address as possible...
            // by masking via the allocated unified memory size.
            void* aligned_address = (void*)((uintptr_t)faulting_address & ~(sz - 1));

            // Allocate memory at the faulting address
            void* allocated = mmap(aligned_address, sz, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (allocated == MAP_FAILED) {
                perror("Failed to allocate memory at faulting address");
                _exit(1);
            }

            std::cout << "allocated dynamic memory at address: " << allocated << std::endl;

            return;
        }
    }

    // raise our original segfault handler
    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGSEGV, &sa, nullptr) == -1) {
        perror("Failed to reset SIGSEGV handler");
        _exit(EXIT_FAILURE);
    }

    raise(SIGSEGV);
}

void maybe_copy_unified_arg(const int index, void* arg, enum cudaMemcpyKind kind)
{
    auto& unified_devices = conns[index].unified_devices;
    auto found = unified_devices.find(arg);
    if (found != unified_devices.end())
    {
        std::cout << "found unified arg pointer; copying..." << std::endl;

        void* ptr = found->first;
        size_t size = found->second;

        cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);

        if (res != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(res) << std::endl;
        } else {
            std::cout << "Successfully copied " << size << " bytes" << std::endl;
        }
    }
}


static void set_segfault_handlers() {
    if (init > 0) {
        return;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = segfault;

    if (sigaction(SIGSEGV, &sa, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    std::cout << "Segfault handler installed." << std::endl;

    init = 1;
}

int rpc_open()
{
    set_segfault_handlers();

    sigsetjmp(catch_segfault, 1);

    if (pthread_mutex_lock(&conn_mutex) < 0)
        return -1;

    if (nconns > 0)
    {
        if (pthread_mutex_unlock(&conn_mutex) < 0)
            return -1;
        return 0;
    }

    char *server_ips = getenv("SCUDA_SERVER");
    if (server_ips == NULL)
    {
        printf("SCUDA_SERVER environment variable not set\n");
        std::exit(1);
    }

    char *server_ip = strdup(server_ips);
    char *token;
    while ((token = strsep(&server_ip, ",")))
    {
        char *host;
        char *port;

        // Split the string into IP address and port
        char *colon = strchr(token, ':');
        if (colon == NULL)
        {
            host = token;
            port = const_cast<char *>(DEFAULT_PORT);
        }
        else
        {
            *colon = '\0';
            host = token;
            port = colon + 1;
        }

        addrinfo hints, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host, port, &hints, &res) != 0)
        {
            std::cout << "getaddrinfo of " << host << " port " << port << " failed" << std::endl;
            continue;
        }

        int flag = 1;
        int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (sockfd == -1)
        {
            printf("socket creation failed...\n");
            exit(1);
        }

        int opts = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
        if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0)
        {
            std::cerr << "Connecting to " << host << " port " << port << " failed: "
                      << strerror(errno) << std::endl;
            exit(1);
        }

        conns[nconns++] = {sockfd, 0, 0, 0, 0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
    }

    if (pthread_mutex_unlock(&conn_mutex) < 0)
        return -1;
    if (nconns == 0)
        return -1;
    return 0;
}

int rpc_size()
{
    return nconns;
}

int rpc_start_request(const int index, const unsigned int op)
{
    if (rpc_open() < 0 ||
        pthread_mutex_lock(&conns[index].write_mutex) < 0)
    {
#ifdef VERBOSE
        std::cout << "rpc_start_request failed due to rpc_open() < 0 || conns[index].write_mutex lock" << std::endl;
#endif
        return -1;
    }

    conns[index].write_iov_count = 2;
    conns[index].write_request_op = op;
    return 0;
}

int rpc_write(const int index, const void *data, const size_t size)
{
    conns[index].write_iov[conns[index].write_iov_count++] = {const_cast<void *>(data), size};
    return 0;
}

int rpc_end_request(const int index)
{
    int write_request_id = ++(conns[index].write_request_id);

    conns[index].write_iov[0] = {&write_request_id, sizeof(int)};
    conns[index].write_iov[1] = {&conns[index].write_request_op, sizeof(unsigned int)};

    // write the request to the server
    if (writev(conns[index].connfd, conns[index].write_iov, conns[index].write_iov_count) < 0 ||
        pthread_mutex_unlock(&conns[index].write_mutex) < 0)
        return -1;
    return write_request_id;
}

int rpc_wait_for_response(const int index)
{
    int wait_for_request_id = rpc_end_request(index);
    if (wait_for_request_id < 0)
        return -1;

    if (pthread_mutex_lock(&conns[index].read_mutex) < 0)
        return -1;

    // wait for the response
    while (true)
    {
        while (conns[index].active_response_id != wait_for_request_id && conns[index].active_response_id != 0)
            pthread_cond_wait(&conns[index].read_cond, &conns[index].read_mutex);

        // we currently own mutex. if active response id is 0, read the response id
        if (conns[index].active_response_id == 0)
        {
            if (read(conns[index].connfd, &conns[index].active_response_id, sizeof(int)) < 0)
            {
                pthread_mutex_unlock(&conns[index].read_mutex);
                return -1;
            }

            if (conns[index].active_response_id != wait_for_request_id)
            {
                pthread_cond_broadcast(&conns[index].read_cond);
                continue;
            }
        }

        conns[index].active_response_id = 0;
        return 0;
    }
}

int rpc_read(const int index, void *data, size_t size)
{
    if (data == nullptr)
    {
        // temp buffer to discard data
        char tempBuffer[256];
        while (size > 0)
        {
            ssize_t bytesRead = read(conns[index].connfd, tempBuffer, std::min(size, sizeof(tempBuffer)));
            if (bytesRead < 0)
            {
                pthread_mutex_unlock(&conns[index].read_mutex);
                return -1; // error if reading fails
            }
            size -= bytesRead;
        }
        return size;
    }

    ssize_t n = recv(conns[index].connfd, data, size, MSG_WAITALL);
    if (n < 0)
        pthread_mutex_unlock(&conns[index].read_mutex);
    return n;
}

void allocate_unified_mem_pointer(const int index, void *dev_ptr, size_t size)
{
    // allocate new space for pointer mapping
    conns[index].unified_devices.insert({ dev_ptr, size });
}

void cuda_memcpy_unified_ptrs(const int index, cudaMemcpyKind kind)
{
    for (const auto & [ ptr, sz ] : conns[index].unified_devices) {
        size_t size = reinterpret_cast<size_t>(sz);

        // ptr is the same on both host/device
        cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);
        if (res != cudaSuccess) {
            std::cerr << "cudaMemcpy failed :" << cudaGetErrorString(res) << std::endl;
        } else {
            std::cout << "Successfully copied " << size << " bytes" << std::endl;
        }
    }
}

void maybe_free_unified_mem(const int index, void *ptr)
{
    for (const auto & [ dev_ptr, sz ] : conns[index].unified_devices) {
        size_t size = reinterpret_cast<size_t>(sz);

        if (dev_ptr == ptr) {
            munmap(dev_ptr, size);
            return;
        }
    }
}

int rpc_end_response(const int index, void *result)
{
    if (read(conns[index].connfd, result, sizeof(int)) < 0 ||
        pthread_mutex_unlock(&conns[index].read_mutex) < 0)
        return -1;
    return 0;
}

void rpc_close()
{
    if (pthread_mutex_lock(&conn_mutex) < 0)
        return;
    while (--nconns >= 0)
        close(conns[nconns].connfd);
    pthread_mutex_unlock(&conn_mutex);
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
    void *func = get_function_pointer(name);

    /** proc address function calls are basically dlsym; we should handle this differently at the top level. */
    if (strcmp(name, "cuGetProcAddress_v2") == 0 || strcmp(name, "cuGetProcAddress") == 0)
    {
        return (void *)&cuGetProcAddress_v2;
    }

    if (func != nullptr)
    {
        // std::cout << "[dlsym] Function address from cudaFunctionMap: " << func << " " << name << std::endl;
        return func;
    }

    // Real dlsym lookup
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
    {
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    // std::cout << "[dlsym] Falling back to real_dlsym for name: " << name << std::endl;
    return real_dlsym(handle, name);
}
