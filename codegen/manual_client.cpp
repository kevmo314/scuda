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

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(RPC_cudaMemcpyAsync);
    if (request_id < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&dst, sizeof(void*)) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&count, sizeof(size_t)) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(src, count) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&kind, sizeof(enum cudaMemcpyKind)) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&stream, sizeof(cudaStream_t)) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_wait_for_response(request_id) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_end_request(&return_value, request_id) < 0) {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}