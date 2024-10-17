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
extern int rpc_wait_for_response();
extern int rpc_end_request(void *return_value);

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    if (rpc_start_request(RPC_cudaMemcpy) < 0)
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

        if (rpc_wait_for_response() < 0)
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

        if (rpc_wait_for_response() < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_end_request(&return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;

    if (rpc_start_request(RPC_cudaMemcpyAsync) < 0)
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

        if (rpc_wait_for_response() < 0)
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

        if (rpc_wait_for_response() < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_end_request(&return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}
