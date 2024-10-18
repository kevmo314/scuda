#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

#include "gen_server.h"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);

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

    return result;
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

    return result;
}

int handle_cudaLaunchKernel(void *conn)
{
    cudaError_t result;
    const void *func;
    dim3 gridDim, blockDim;
    size_t sharedMem;
    cudaStream_t stream;

    // Read the function pointer (kernel) from the client
    if (rpc_read(conn, &func, sizeof(const void *)) < 0)
    {
        return -1;
    }

    // Read grid dimensions (gridDim)
    if (rpc_read(conn, &gridDim, sizeof(dim3)) < 0)
    {
        return -1;
    }

    // Read block dimensions (blockDim)
    if (rpc_read(conn, &blockDim, sizeof(dim3)) < 0)
    {
        return -1;
    }

    // Read shared memory size
    if (rpc_read(conn, &sharedMem, sizeof(size_t)) < 0)
    {
        return -1;
    }

    // Read the CUDA stream
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
    {
        return -1;
    }

    // Read the number of kernel arguments
    int num_args;
    if (rpc_read(conn, &num_args, sizeof(int)) < 0)
    {
        return -1;
    }

    // Allocate memory to hold kernel arguments
    void **args = (void **)malloc(num_args * sizeof(void *));
    if (args == NULL)
    {
        std::cerr << "Failed to allocate memory for kernel arguments." << std::endl;
        return -1;
    }

    std::cout << "allocated arg memory for args: " << num_args << std::endl;

   // Calculate the total size of all arguments (as pointers)
    size_t total_size = num_args * sizeof(void *);

    // Allocate a buffer to hold the serialized arguments from the client
    void *arg_buffer = malloc(total_size);
    if (arg_buffer == NULL)
    {
        std::cerr << "Failed to allocate memory for arguments buffer." << std::endl;
        free(args);
        return -1;
    }

    // Read all the arguments from the client into the buffer
    if (rpc_read(conn, arg_buffer, total_size) < 0)
    {
        std::cerr << "Failed to read arguments from client." << std::endl;
        free(args);
        free(arg_buffer);
        return -1;
    }

    // Deserialize the arguments from the buffer into the args array
    memcpy(args, arg_buffer, total_size);

    // Free the temporary buffer after deserialization
    free(arg_buffer);

    std::cout << "launching kernel... " << arg_buffer << " symbol: " << &func << std::endl;

    // Launch the kernel on the GPU using the provided parameters
    result = cudaLaunchKernel(&func, gridDim, blockDim, args, sharedMem, stream);
    if (result != cudaSuccess)
    {
        std::cerr << "cudaLaunchKernel failed: " << cudaGetErrorString(result) << std::endl;
        free(args);
        return -1;
    }

    std::cout << "launched kernel!" << std::endl;

    // Free the memory allocated for the kernel arguments
    free(args);

    // Finalize the request
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        return -1;
    }

    // Send response to the client with the result of the kernel launch
    if (rpc_start_response(conn, request_id) < 0)
    {
        return -1;
    }

    return result;
}

typedef void **(*__cudaRegisterFatBinary_type)(void **fatCubin);

int handle___cudaRegisterFatBinary(void *conn)
{
    void *fatCubin;
        
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        return -1;
    }

    __cudaRegisterFatBinary_type orig;
    orig =
        (__cudaRegisterFatBinary_type)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    auto ret = orig(&fatCubin);

    if (rpc_start_response(conn, request_id) < 0)
    {
        return -1;
    }

    std::cout << "registeredCubin after: " << ret << std::endl;

    if (rpc_write(conn, &ret, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to write fatCubin back to the client" << std::endl;
        return -1;
    }
    return 0;
}

typedef void (*__cudaRegisterFunction_type)(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);

int handle___cudaRegisterFunction(void *conn)
{
    void **fatCubinHandle;
    const char *hostFun;
    char *deviceFun;
    const char *deviceName;
    int thread_limit;
    uint3 *tid, *bid;
    dim3 *bDim, *gDim;
    int *wSize;

    // // Reading the input parameters from the client
    // if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &hostFun, sizeof(const char *)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &deviceFun, sizeof(char *)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &deviceName, sizeof(const char *)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &thread_limit, sizeof(int)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &tid, sizeof(uint3)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &bid, sizeof(uint3)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &bDim, sizeof(dim3)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &gDim, sizeof(dim3)) < 0)
    // {
    //     return -1;
    // }

    // if (rpc_read(conn, &wSize, sizeof(int)) < 0)
    // {
    //     return -1;
    // }

    // End request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Log the parameters before the call
    std::cout << "Before __cudaRegisterFunction call:" << std::endl;
    std::cout << "fatCubinHandle: " << fatCubinHandle << std::endl;

    __cudaRegisterFunction_type orig;
    orig =
        (__cudaRegisterFunction_type)dlsym(RTLD_NEXT, "__cudaRegisterFunction");

    orig(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);

    // Log the parameters after the call
    std::cout << "After __cudaRegisterFunction call:" << std::endl;
    std::cout << "fatCubinHandle: " << fatCubinHandle << std::endl;
    std::cout << "hostFun: " << hostFun << std::endl;
    std::cout << "deviceFun: " << deviceFun << std::endl;
    std::cout << "deviceName: " << deviceName << std::endl;
    std::cout << "thread_limit: " << thread_limit << std::endl;
  
    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    // Write the updated data back to the client
    if (rpc_write(conn, fatCubinHandle, sizeof(void *)) < 0)
    {
        std::cerr << "Failed writing fatCubinHandle" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &hostFun, sizeof(const char *)) < 0)
    {
        std::cerr << "Failed writing hostFun" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &deviceFun, sizeof(char *)) < 0)
    {
        std::cerr << "Failed writing deviceFun" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &deviceName, sizeof(const char *)) < 0)
    {
        std::cerr << "Failed writing deviceName" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &thread_limit, sizeof(int)) < 0)
    {
        std::cerr << "Failed writing thread_limit" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &tid, sizeof(uint3)) < 0)
    {
        std::cerr << "Failed writing tid" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &bid, sizeof(uint3)) < 0)
    {
        std::cerr << "Failed writing bid" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &bDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed writing bDim" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &gDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed writing gDim" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &wSize, sizeof(int)) < 0)
    {
        std::cerr << "Failed writing wSize" << std::endl;
        return -1;
    }

    return 0;
}

typedef void (*__cudaRegisterFatBinaryEnd_type)(void **fatCubinHandle);

int handle___cudaRegisterFatBinaryEnd(void *conn)
{
    void **fatCubinHandle;

    // Read the fatCubinHandle from the client
    if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        std::cerr << "Failed reading fatCubinHandle" << std::endl;
        return -1;
    }

    // End the request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    std::cout << "Calling __cudaRegisterFatBinaryEnd with fatCubinHandle: " << fatCubinHandle << std::endl;
    __cudaRegisterFatBinaryEnd_type orig;
    orig = (__cudaRegisterFatBinaryEnd_type)dlsym(RTLD_NEXT,
                                                    "__cudaRegisterFatBinaryEnd");

    orig(fatCubinHandle);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    // Write the fatCubinHandle back to the client
    if (rpc_write(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        std::cerr << "Failed writing fatCubinHandle back" << std::endl;
        return -1;
    }

    return 0;
}

