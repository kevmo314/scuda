#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_response(const void *conn, void *return_value);

int handle_cudaMemcpy(void *conn)
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

        result = cudaMemcpy(host_data, dst, count, cudaMemcpyDeviceToHost);
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

        int request_id = rpc_end_request(conn);
        if (request_id < 0)
        {
            free(src);
            return -1;
        }

        result = cudaMemcpy(dst, src, count, kind);

        free(src);

        if (rpc_start_response(conn, request_id) < 0)
        {
            return -1;
        }
    }

    if (rpc_end_response(conn, &result) < 0)
        return -1;
    return 0;
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
    return 0;
}
