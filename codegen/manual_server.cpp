#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);

int handle_cudaMemcpyAsync(void *conn)
{
    void* dst;
    if (rpc_read(conn, &dst, sizeof(void*)) < 0)
        return -1;

    std::size_t count;
    if (rpc_read(conn, &count, sizeof(size_t)) < 0)
        return -1;

    void* src = malloc(count);
    if (src == NULL) {
        return -1;
    }

    if (rpc_read(conn, src, count) < 0) {
        free(src);
        return -1;
    }

    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0) {
        free(dst);
        free(src);
        return -1;
    }

    cudaStream_t stream;
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0) {
        free(dst);
        free(src);
        return -1;
    }

    int request_id = rpc_end_request(conn);
    if (request_id < 0) {
        return -1;
    }

    cudaError_t result = cudaMemcpyAsync(dst, src, count, kind, stream);

    // free memory after operation completes
    free(src);

    if (rpc_start_response(conn, request_id) < 0) {
        return -1;
    }

    return result;
}