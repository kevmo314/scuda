#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const std::size_t size);
extern int rpc_read(void *data, const std::size_t size);
extern int rpc_wait_for_response(const unsigned int request_id);
extern int rpc_end_request(void *return_value, const unsigned int request_id);

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpy);
    if (request_id < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // we need to swap device directions in this case
    if (kind == cudaMemcpyDeviceToHost)
    {
        if (rpc_write(&src, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(&count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_read(dst, count) < 0)
        { // Read data into the destination buffer on the host
            return cudaErrorDevicesUnavailable;
        }
    }
    else
    {
        if (rpc_write(&dst, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(&count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(src, count) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_wait_for_response(request_id) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_end_request(&return_value, request_id) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyAsync);
    if (request_id < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // we need to swap device directions in this case
    if (kind == cudaMemcpyDeviceToHost)
    {
        if (rpc_write(&src, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(&count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(&stream, sizeof(cudaStream_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_read(dst, count) < 0)
        { // Read data into the destination buffer on the host
            return cudaErrorDevicesUnavailable;
        }
    }
    else
    {
        if (rpc_write(&dst, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(&count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(src, count) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(&stream, sizeof(cudaStream_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_wait_for_response(request_id) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_end_request(&return_value, request_id) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    cudaError_t return_value;

    // Start the RPC request
    int request_id = rpc_start_request(RPC_cudaLaunchKernel);
    if (request_id < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    std::cout << "device func... " << func << std::endl;

    // Write the function pointer (kernel) to be launched
    if (rpc_write(&func, sizeof(const void *)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // Write the grid dimensions (for launching the kernel)
    if (rpc_write(&gridDim, sizeof(dim3)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // Write the block dimensions
    if (rpc_write(&blockDim, sizeof(dim3)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // Write the shared memory size
    if (rpc_write(&sharedMem, sizeof(size_t)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // Write the stream to use for kernel execution
    if (rpc_write(&stream, sizeof(cudaStream_t)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // Write the kernel arguments one by one (assuming args is an array of void pointers)
    int num_args = 0; // Get the number of arguments
    while (args[num_args] != nullptr)
        num_args++;

    if (rpc_write(&num_args, sizeof(int)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    std::cout << "num arg... " << num_args << std::endl;

    // Send each argument to the remote server
    size_t total_size = num_args * sizeof(void *);

    void *arg_buffer = malloc(total_size);
    if (arg_buffer == NULL)
    {
        std::cerr << "Failed to allocate memory for arguments buffer." << std::endl;
        return cudaErrorDevicesUnavailable;
    }

    memcpy(arg_buffer, args, total_size);

    if (rpc_write(arg_buffer, total_size) < 0)
    {
        free(arg_buffer); // Free buffer if rpc_write fails
        return cudaErrorDevicesUnavailable;
    }

    free(arg_buffer);

    // Wait for a response after the kernel execution request is completed
    if (rpc_wait_for_response(request_id) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // End the request and get the return value
    if (rpc_end_request(&return_value, request_id) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

extern "C"
{
    void __cudaRegisterFunction(void **fatCubinHandle,
                                const char *hostFun,
                                char *deviceFun,
                                const char *deviceName,
                                int thread_limit,
                                uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
    {
        cudaError_t return_value;

        std::cout << "calling __cudaRegisterFunction" << std::endl;

        // Start the RPC request for __cudaRegisterFunction
        int request_id = rpc_start_request(RPC___cudaRegisterFunction);
        if (request_id < 0)
        {
            return;
        }

        if (rpc_write(fatCubinHandle, sizeof(void *)) < 0 ||
            rpc_write(&hostFun, sizeof(const char *)) < 0 ||
            rpc_write(&deviceFun, sizeof(char *)) < 0 ||
            rpc_write(&deviceName, sizeof(const char *)) < 0 ||
            rpc_write(&thread_limit, sizeof(int)) < 0 ||
            rpc_write(tid, sizeof(uint3)) < 0 ||
            rpc_write(bid, sizeof(uint3)) < 0 ||
            rpc_write(bDim, sizeof(dim3)) < 0 ||
            rpc_write(gDim, sizeof(dim3)) < 0 ||
            rpc_write(wSize, sizeof(int)) < 0)
        {
            return;
        }

        if (rpc_wait_for_response(request_id) < 0)
        {
            return;
        }

        if (rpc_end_request(&return_value, request_id) < 0)
        {
            return;
        }

        return;
    }
}