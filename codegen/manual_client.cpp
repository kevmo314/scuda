#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

extern int rpc_size();
extern int rpc_start_request(const int index, const unsigned int request);
extern int rpc_write(const int index, const void *data, const std::size_t size);
extern int rpc_wait_for_response(const int index);
extern int rpc_read(const int index, void *data, const std::size_t size);
extern int rpc_end_request(const int index, void *return_value);
extern int rpc_close();

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(0, RPC_cudaMemcpy);
    if (request_id < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // we need to swap device directions in this case
    if (kind == cudaMemcpyDeviceToHost)
    {
        if (rpc_write(0, &src, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, &count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_wait_for_response(0) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_read(0, dst, sizeof(count)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }
    else
    {
        if (rpc_write(0, &dst, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, &count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, src, sizeof(count)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_wait_for_response(0) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_end_request(0, &return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t return_value;

    int request_id = rpc_start_request(0, RPC_cudaMemcpyAsync);
    if (request_id < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &kind, sizeof(enum cudaMemcpyKind)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // we need to swap device directions in this case
    if (kind == cudaMemcpyDeviceToHost)
    {
        if (rpc_write(0, &src, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, &count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, &stream, sizeof(cudaStream_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_wait_for_response(0) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_read(0, dst, sizeof(count)) < 0)
        { // Read data into the destination buffer on the host
            return cudaErrorDevicesUnavailable;
        }
    }
    else
    {
        if (rpc_write(0, &dst, sizeof(void *)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, &count, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, src, sizeof(count)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, &stream, sizeof(cudaStream_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_wait_for_response(0) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_end_request(0, &return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle)
{
    cublasStatus_t return_value;

    std::cout << "cublas handle: " << handle << std::endl;

    if (rpc_start_request(0, RPC_cublasCreate_v2) < 0 ||
        rpc_write(0, handle, sizeof(cublasHandle_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_request(0, &return_value) < 0)
    {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    return return_value;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B,  int ldb,
                           const float *beta,
                           float *C,  int ldc)
{
    cublasStatus_t return_value;

    std::cout << "calling cublasSgemm_v2: " << &handle << std::endl;

    if (rpc_start_request(0, RPC_cublasSgemm_v2) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &handle, sizeof(cublasHandle_t)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &transa, sizeof(cublasOperation_t)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &transb, sizeof(cublasOperation_t)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &m, sizeof(int)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &n, sizeof(int)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &k, sizeof(int)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, alpha, sizeof(float)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &A, sizeof(const float *)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &lda, sizeof(int)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &B, sizeof(const float *)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &ldb, sizeof(int)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, beta, sizeof(float)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &C, sizeof(float *)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_write(0, &ldc, sizeof(int)) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_wait_for_response(0) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    if (rpc_end_request(0, &return_value) < 0) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    std::cout << "cublasSgemm_v2 completed successfully" << std::endl;
    return return_value;
}
