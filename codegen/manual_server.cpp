#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <cublas_v2.h>
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

        if (rpc_write(conn, host_data, sizeof(count)) < 0)
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

        if (rpc_read(conn, src, sizeof(count)) < 0)
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
        if (result != cudaSuccess)
        {
            free(src);
            return -1;
        }

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

        if (rpc_write(conn, host_data, sizeof(count)) < 0)
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

        if (rpc_read(conn, src, sizeof(count)) < 0)
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

int handle_cublasCreate_v2(void *conn)
{
    cublasHandle_t* handle;

    if (rpc_read(conn, handle, sizeof(cublasHandle_t)) < 0)
        return -1;

    int request_id = rpc_end_request(conn);
    if (request_id < 0)
        return -1;

    cublasStatus_t result = cublasCreate(handle);

    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(cublasStatus_t)) < 0 ||
        rpc_end_response(conn, &result) < 0)
        return -1;

    return 0;
}

int handle_cublasSgemm_v2(void *conn)
{
    cublasHandle_t handle;
    cublasOperation_t transa, transb;
    int m, n, k, lda, ldb, ldc;
    float alpha, beta;
    const float *A, *B;
    float *C;

    if (rpc_read(conn, &handle, sizeof(cublasHandle_t)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &transa, sizeof(cublasOperation_t)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &transb, sizeof(cublasOperation_t)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &m, sizeof(int)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &n, sizeof(int)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &k, sizeof(int)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &alpha, sizeof(float)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &A, sizeof(const float *)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &lda, sizeof(int)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &B, sizeof(const float *)) < 0) {
        return -1;
    }
   
    if (rpc_read(conn, &ldb, sizeof(int)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &beta, sizeof(float)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &C, sizeof(float *)) < 0) {
        return -1;
    }

    if (rpc_read(conn, &ldc, sizeof(int)) < 0) {
        return -1;
    }

    int request_id = rpc_end_request(conn);
    if (request_id < 0) {
        return -1;
    }

    std::cout << "Calling cublasSgemm with handle: " << handle << std::endl;

    printf("Calling cublasSgemm with the following parameters:\n");
    printf("  Handle: %p\n", handle);
    printf("  transa: %d, transb: %d\n", transa, transb);
    printf("  m: %d, n: %d, k: %d\n", m, n, k);
    printf("  alpha: %f\n", alpha);
    printf("  A: %p, lda: %d\n", A, lda);
    printf("  B: %p, ldb: %d\n", B, ldb);
    printf("  beta: %f\n", beta);
    printf("  C: %p, ldc: %d\n", C, ldc);

    // Perform cublasSgemm
    cublasStatus_t result = cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);

    printf("finished calling cublasSgemm :\n");

    // Send the response
    if (rpc_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(cublasStatus_t)) < 0 ||
        rpc_end_response(conn, &result) < 0) {
        return -1;
    }

    return 0;
}
