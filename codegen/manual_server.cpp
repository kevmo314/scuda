#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <dlfcn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"
#include "ptx_fatbin.hpp"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_response(const void *conn, void *return_value);

extern std::unordered_map<void *, void *> registered_pointers;

FILE *__cudart_trace_output_stream = stdout;

int handle_cudaHostRegister(void *conn)
{
    cudaError_t result;
    int request_id;
    void *ptr;
    std::size_t size;
    unsigned int flags;

    printf("got request\n");

    if (rpc_read(conn, &ptr, sizeof(void *)) < 0 ||
        rpc_read(conn, &size, sizeof(size_t)) < 0 ||
        rpc_read(conn, &flags, sizeof(unsigned int)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;

    printf("registering %p %zu %u\n", ptr, size, flags);

    result = cudaHostRegister(ptr, size, flags);

    if (result == cudaSuccess)
        registered_pointers[ptr] = conn;

    printf("request_id: %d\n", request_id);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    printf("registered %p\n", ptr);

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaHostUnregister(void *conn)
{
    cudaError_t result;
    int request_id;
    void *ptr;

    if (rpc_read(conn, &ptr, sizeof(void *)) < 0)
        goto ERROR_0;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_0;

    result = cudaHostUnregister(ptr);

    if (result == cudaSuccess)
        registered_pointers.erase(ptr);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_0;

    return 0;
ERROR_0:
    return -1;
}

int handle_cudaMemcpy(void *conn)
{
    int request_id;
    cudaError_t result;
    void *src;
    void *dst;
    void *host_data;
    std::size_t count;
    enum cudaMemcpyKind kind;

    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
        goto ERROR_0;

    switch (kind)
    {
    case cudaMemcpyDeviceToHost:
        if (rpc_read(conn, &src, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0)
            goto ERROR_0;

        host_data = malloc(count);
        if (host_data == NULL)
            goto ERROR_0;

        request_id = rpc_end_request(conn);
        if (request_id < 0)
            goto ERROR_1;

        result = cudaMemcpy(host_data, src, count, cudaMemcpyDeviceToHost);

        if (rpc_start_response(conn, request_id) < 0 ||
            rpc_write(conn, host_data, count) < 0)
            goto ERROR_1;
        break;
    case cudaMemcpyHostToDevice:
        if (rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0)
            goto ERROR_0;

        host_data = malloc(count);
        if (host_data == NULL)
            goto ERROR_0;

        if (rpc_read(conn, host_data, count) < 0)
            goto ERROR_1;

        request_id = rpc_end_request(conn);
        if (request_id < 0)
            goto ERROR_1;

        result = cudaMemcpy(dst, host_data, count, kind);

        if (rpc_start_response(conn, request_id) < 0)
            goto ERROR_1;
        break;
    case cudaMemcpyDeviceToDevice:
        if (rpc_read(conn, &src, sizeof(void *)) < 0 ||
            rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0)
            goto ERROR_0;

        request_id = rpc_end_request(conn);
        if (request_id < 0)
            goto ERROR_0;

        result = cudaMemcpy(dst, src, count, kind);

        if (rpc_start_response(conn, request_id) < 0)
            goto ERROR_0;
        break;
    }

    if (rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    free((void *)host_data);
ERROR_0:
    return -1;
}

int handle_cudaMemcpyAsync(void *conn)
{
    cudaError_t result;
    void *dst;

    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
    {
        return -1;
    }

    if (kind == cudaMemcpyDeviceToHost)
    {
        if (rpc_read(conn, &dst, sizeof(void *)) < 0)
            return -1;

        std::size_t count;
        if (rpc_read(conn, &count, sizeof(size_t)) < 0)
            return -1;

        cudaStream_t stream;
        if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        {
            return -1;
        }

        void *host_data = malloc(count);
        if (host_data == NULL)
        {
            std::cerr << "Failed to allocate host memory for device-to-host transfer." << std::endl;
            return -1;
        }

        int request_id = rpc_end_request(conn);
        if (request_id < 0)
        {
            return -1;
        }

        result = cudaMemcpyAsync(host_data, dst, count, cudaMemcpyDeviceToHost, stream);
        if (result != cudaSuccess)
        {
            free(host_data);
            return -1;
        }

        if (rpc_start_response(conn, request_id) < 0)
        {
            return -1;
        }

        if (rpc_write(conn, host_data, count) < 0)
        {
            free(host_data);
            return -1;
        }

        // free temp memory after writing host data back
        free(host_data);
    }
    else
    {
        if (rpc_read(conn, &dst, sizeof(void *)) < 0)
            return -1;

        std::size_t count;
        if (rpc_read(conn, &count, sizeof(size_t)) < 0)
            return -1;

        void *src = malloc(count);
        if (src == NULL)
        {
            return -1;
        }

        if (rpc_read(conn, src, count) < 0)
        {
            free(src);
            return -1;
        }

        cudaStream_t stream;
        if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        {
            free(src);
            return -1;
        }

        int request_id = rpc_end_request(conn);
        if (request_id < 0)
        {
            free(src);
            return -1;
        }

        result = cudaMemcpyAsync(dst, src, count, kind, stream);

        free(src);

        if (rpc_start_response(conn, request_id) < 0)
        {
            return -1;
        }
    }

    if (rpc_end_response(conn, &result) < 0)
        return -1;

    std::cout << "end cudaMemcpyAsync" << std::endl;

    return 0;
}

int handle_cudaLaunchKernel(void *conn)
{
    int request_id;
    cudaError_t result;
    const void *func;
    void **args;
    dim3 gridDim, blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    int num_args;
    int arg_size;

    if (rpc_read(conn, &func, sizeof(const void *)) < 0 ||
        rpc_read(conn, &gridDim, sizeof(dim3)) < 0 ||
        rpc_read(conn, &blockDim, sizeof(dim3)) < 0 ||
        rpc_read(conn, &sharedMem, sizeof(size_t)) < 0 ||
        rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_read(conn, &num_args, sizeof(int)) < 0)
        goto ERROR_0;

    // Allocate memory for the arguments
    args = (void **)malloc(num_args * sizeof(void *));
    if (args == NULL)
        goto ERROR_0;

    for (int i = 0; i < num_args; ++i)
    {
        if (rpc_read(conn, &arg_size, sizeof(int)) < 0)
            goto ERROR_1;

        // Allocate memory for the argument
        args[i] = malloc(arg_size);
        if (args[i] == NULL)
            goto ERROR_1;

        // Read the actual argument data from the client
        if (rpc_read(conn, args[i], arg_size) < 0)
            goto ERROR_1;
    }

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_1;

    result = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        goto ERROR_1;

    return 0;
ERROR_1:
    for (int i = 0; i < num_args; ++i)
        if (args[i] != NULL)
            free(args[i]);
    free(args);
ERROR_0:
    return -1;
}

std::unordered_map<void **, __cudaFatCudaBinary2 *> fat_binary_map;

extern "C" void **__cudaRegisterFatBinary(void *fatCubin);

int handle___cudaRegisterFatBinary(void *conn)
{
    __cudaFatCudaBinary2 *fatCubin = (__cudaFatCudaBinary2 *)malloc(sizeof(__cudaFatCudaBinary2));
    unsigned long long size;

    if (rpc_read(conn, fatCubin, sizeof(__cudaFatCudaBinary2)) < 0 ||
        rpc_read(conn, &size, sizeof(unsigned long long)) < 0)
        return -1;

    void *cubin = malloc(size);
    if (rpc_read(conn, cubin, size) < 0)
        return -1;

    fatCubin->text = (uint64_t)cubin;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    void **p = __cudaRegisterFatBinary(fatCubin);

    int return_value = 0;

    fat_binary_map[p] = fatCubin;

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &p, sizeof(void **)) < 0 ||
        rpc_end_response(conn, &return_value) < 0)
        return -1;

    return 0;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubin);

int handle___cudaUnregisterFatBinary(void *conn)
{
    void **fatCubin;
    if (rpc_read(conn, &fatCubin, sizeof(void **)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    free((void *)fat_binary_map[fatCubin]->text);
    free(fat_binary_map[fatCubin]);
    fat_binary_map.erase(fatCubin);

    __cudaUnregisterFatBinary(fatCubin);

    int return_value = 0;

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &return_value) < 0)
        return -1;

    return 0;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun,
                                       char *deviceFun,
                                       const char *deviceName,
                                       int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

int handle___cudaRegisterFunction(void *conn)
{
    void *res;
    void **fatCubinHandle;
    char *hostFun;
    size_t deviceFunLen;
    size_t deviceNameLen;
    char *deviceFun;
    char *deviceName;
    int thread_limit;
    uint8_t mask;
    uint3 tid, bid;
    dim3 bDim, gDim;
    int wSize;

    int request_id;

    if (rpc_read(conn, &fatCubinHandle, sizeof(void **)) < 0 ||
        rpc_read(conn, &hostFun, sizeof(const char *)) < 0 ||
        rpc_read(conn, &deviceFunLen, sizeof(size_t)) < 0)
        goto ERROR_0;

    deviceFun = (char *)malloc(deviceFunLen);
    if (rpc_read(conn, deviceFun, deviceFunLen) < 0 ||
        rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0)
        goto ERROR_1;

    deviceName = (char *)malloc(deviceNameLen);
    if (rpc_read(conn, deviceName, deviceNameLen) < 0 ||
        rpc_read(conn, &thread_limit, sizeof(int)) < 0 ||
        rpc_read(conn, &mask, sizeof(uint8_t)) < 0 ||
        (mask & 1 << 0 && rpc_read(conn, &tid, sizeof(uint3)) < 0) ||
        (mask & 1 << 1 && rpc_read(conn, &bid, sizeof(uint3)) < 0) ||
        (mask & 1 << 2 && rpc_read(conn, &bDim, sizeof(dim3)) < 0) ||
        (mask & 1 << 3 && rpc_read(conn, &gDim, sizeof(dim3)) < 0) ||
        (mask & 1 << 4 && rpc_read(conn, &wSize, sizeof(int)) < 0))
        goto ERROR_2;

    request_id = rpc_end_request(conn);
    if (request_id < 0)
        goto ERROR_2;

    __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
                           mask & 1 << 0 ? &tid : nullptr, mask & 1 << 1 ? &bid : nullptr,
                           mask & 1 << 2 ? &bDim : nullptr, mask & 1 << 3 ? &gDim : nullptr,
                           mask & 1 << 4 ? &wSize : nullptr);

    return 0;
ERROR_2:
    free((void *)deviceName);
ERROR_1:
    free((void *)deviceFun);
ERROR_0:
    return -1;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);

int handle___cudaRegisterFatBinaryEnd(void *conn)
{
    void *res;
    void **fatCubinHandle;

    // Read the fatCubinHandle from the client
    if (rpc_read(conn, &fatCubinHandle, sizeof(void **)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    __cudaRegisterFatBinaryEnd(fatCubinHandle);

    return 0;
}

// Function pointer type for __cudaPushCallConfiguration
extern "C" cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                   size_t sharedMem, cudaStream_t stream);

int handle___cudaPushCallConfiguration(void *conn)
{
    dim3 gridDim, blockDim;
    size_t sharedMem;
    cudaStream_t stream;

    // Read the grid dimensions from the client
    if (rpc_read(conn, &gridDim, sizeof(dim3)) < 0 ||
        rpc_read(conn, &blockDim, sizeof(dim3)) < 0 ||
        rpc_read(conn, &sharedMem, sizeof(size_t)) < 0 ||
        rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_end_response(conn, &result) < 0)
        return -1;

    return 0;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream);

int handle___cudaPopCallConfiguration(void *conn)
{
    dim3 gridDim, blockDim;
    size_t sharedMem;
    cudaStream_t stream;

    std::cout << "received handle___cudaPopCallConfiguration" << std::endl;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cudaError_t result = __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &gridDim, sizeof(dim3)) < 0 ||
        rpc_write(conn, &blockDim, sizeof(dim3)) < 0 ||
        rpc_write(conn, &sharedMem, sizeof(size_t)) < 0 ||
        rpc_write(conn, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        return -1;

    return 0;
}

typedef void (*__cudaInitModule_type)(void **fatCubinHandle);

void __cudaInitModule(void **fatCubinHandle)
{
    std::cerr << "calling __cudaInitModule" << std::endl;
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle,
                                  char *hostVar, char *deviceAddress,
                                  const char *deviceName, int ext, size_t size,
                                  int constant, int global);

int handle___cudaRegisterVar(void *conn)
{
    void *res;
    void **fatCubinHandle;
    char *hostVar;
    char *deviceAddress;
    char *deviceName;
    int ext;
    size_t size;
    int constant;
    int global;

    // Read the fatCubinHandle
    if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        std::cerr << "Failed reading fatCubinHandle" << std::endl;
        return -1;
    }

    // Read hostVar
    size_t hostVarLen;
    if (rpc_read(conn, &hostVarLen, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read hostVar length" << std::endl;
        return -1;
    }
    hostVar = (char *)malloc(hostVarLen);
    if (rpc_read(conn, hostVar, hostVarLen) < 0)
    {
        std::cerr << "Failed to read hostVar" << std::endl;
        return -1;
    }

    // Read deviceAddress
    size_t deviceAddressLen;
    if (rpc_read(conn, &deviceAddressLen, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read deviceAddress length" << std::endl;
        return -1;
    }
    deviceAddress = (char *)malloc(deviceAddressLen);
    if (rpc_read(conn, deviceAddress, deviceAddressLen) < 0)
    {
        std::cerr << "Failed to read deviceAddress" << std::endl;
        return -1;
    }

    // Read deviceName
    size_t deviceNameLen;
    if (rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read deviceName length" << std::endl;
        return -1;
    }
    deviceName = (char *)malloc(deviceNameLen);
    if (rpc_read(conn, deviceName, deviceNameLen) < 0)
    {
        std::cerr << "Failed to read deviceName" << std::endl;
        return -1;
    }

    // Read ext, size, constant, global
    if (rpc_read(conn, &ext, sizeof(int)) < 0)
    {
        std::cerr << "Failed reading ext" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed reading size" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &constant, sizeof(int)) < 0)
    {
        std::cerr << "Failed reading constant" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &global, sizeof(int)) < 0)
    {
        std::cerr << "Failed reading global" << std::endl;
        return -1;
    }

    std::cout << "Received __cudaRegisterVar with deviceName: " << deviceName << std::endl;

    __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);

    // End request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    return 0;
}
