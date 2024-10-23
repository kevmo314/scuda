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

FILE *__cudart_trace_output_stream = stdout;

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
    void **args;
    dim3 gridDim, blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    int num_args;

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
    if (rpc_read(conn, &num_args, sizeof(int)) < 0)
    {
        return -1;
    }

    // Allocate memory for the arguments
    args = (void **)malloc(num_args * sizeof(void *));
    if (args == NULL)
    {
        std::cerr << "Failed to allocate memory for kernel arguments." << std::endl;
        return -1;
    }

    for (int i = 0; i < num_args; ++i)
    {
        size_t arg_size;
        
        if (rpc_read(conn, &arg_size, sizeof(size_t)) < 0)
        {
            std::cerr << "Failed to read size of argument " << i << " from client." << std::endl;
            free(args);
            return -1;
        }

        // Allocate memory for the argument
        args[i] = malloc(arg_size);
        if (args[i] == NULL)
        {
            std::cerr << "Failed to allocate memory for argument " << i << "." << std::endl;
            free(args);
            return -1;
        }

        // Read the actual argument data from the client
        if (rpc_read(conn, args[i], arg_size) < 0)
        {
            std::cerr << "Failed to read argument " << i << " from client." << std::endl;
            free(args[i]);
            free(args);
            return -1;
        }
    }

    std::cout << "Calling cudaLaunchKernel with func: " << func << std::endl;

    result = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    if (result != cudaSuccess)
    {
        std::cerr << "cudaLaunchKernel failed: " << cudaGetErrorString(result) << std::endl;
        for (int i = 0; i < num_args; ++i)
        {
            free(args[i]);
        }
        free(args);
        return -1;
    }

    std::cout << "Kernel launched successfully!" << std::endl;

    // Free argument memory after use
    for (int i = 0; i < num_args; ++i)
    {
        free(args[i]);
    }
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
    void **fatCubin;
    
    // Read the fatCubin data from the client
    if (rpc_read(conn, &fatCubin, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to read fatCubin from client" << std::endl;
        return -1;
    }

    std::cout << "Server received fatCubin: " << fatCubin << std::endl;

    // Call the original __cudaRegisterFatBinary function
    __cudaRegisterFatBinary_type orig;
    orig = (__cudaRegisterFatBinary_type)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaRegisterFatBinary" << std::endl;
        return -1;
    }

    auto ret = orig(fatCubin);

    // End the request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    fprintf(__cudart_trace_output_stream,
          "> __cudaRegisterFatBinary(fatCubin=%p) = %p\n", fatCubin, ret);

    if (rpc_write(conn, &ret, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to write fatCubin result back to the client" << std::endl;
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

    // Reading the input parameters from the client
    if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &hostFun, sizeof(const char *)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &deviceFun, sizeof(char *)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &deviceName, sizeof(const char *)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &thread_limit, sizeof(int)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &tid, sizeof(uint3)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &bid, sizeof(uint3)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &bDim, sizeof(dim3)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &gDim, sizeof(dim3)) < 0)
    {
        return -1;
    }

    if (rpc_read(conn, &wSize, sizeof(int)) < 0)
    {
        return -1;
    }

    // End request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    std::cout << "calling original __cudaRegisterFunction_type" << std::endl;

    __cudaRegisterFunction_type orig;
    orig = (__cudaRegisterFunction_type)dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaRegisterFunction" << std::endl;
        return -1;
    }

    // Call the original __cudaRegisterFunction
    orig(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);

    std::cout << "finished original __cudaRegisterFunction_type" << std::endl;

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
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

    __cudaRegisterFatBinaryEnd_type orig;
    orig = (__cudaRegisterFatBinaryEnd_type)dlsym(RTLD_NEXT, "__cudaRegisterFatBinaryEnd");

    orig(fatCubinHandle);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    return 0;
}

typedef unsigned (*__cudaPopCallConfiguration_type)(dim3 *gridDim, dim3 *blockDim,
                                                    size_t *sharedMem, void *stream);

int handle___cudaPopCallConfiguration(void *conn)
{
    dim3 *gridDim, *blockDim;
    size_t *sharedMem;
    void *stream;

    std::cout << "received handle___cudaPopCallConfiguration" << std::endl;

    // Read the pointers for gridDim, blockDim, sharedMem, and stream from the client
    if (rpc_read(conn, &gridDim, sizeof(dim3 *)) < 0)
    {
        std::cerr << "Failed to read gridDim pointer from client" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &blockDim, sizeof(dim3 *)) < 0)
    {
        std::cerr << "Failed to read blockDim pointer from client" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &sharedMem, sizeof(size_t *)) < 0)
    {
        std::cerr << "Failed to read sharedMem pointer from client" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &stream, sizeof(void *)) < 0)
    {
        std::cerr << "Failed to read stream pointer from client" << std::endl;
        return -1;
    }

    // End the request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Get the original function via dlsym
    __cudaPopCallConfiguration_type orig;
    orig = (__cudaPopCallConfiguration_type)dlsym(RTLD_NEXT, "__cudaPopCallConfiguration");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaPopCallConfiguration: " << dlerror() << std::endl;
        return -1;
    }

    // Call the original function to populate the data
    unsigned result = orig(gridDim, blockDim, sharedMem, stream);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    std::cout << "!!!" << std::endl;

    // Send the updated data back to the client
    if (rpc_write(conn, &gridDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to send gridDim to client" << std::endl;
        return -1;
    }

    std::cout << "???" << std::endl;

    if (rpc_write(conn, &blockDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to send blockDim to client" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &sharedMem, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to send sharedMem to client" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &stream, sizeof(void *)) < 0)
    {
        std::cerr << "Failed to send stream to client" << std::endl;
        return -1;
    }

     // Write the result back to the client
    // if (rpc_write(conn, &result, sizeof(unsigned)) < 0)
    // {
    //     std::cerr << "Failed to write result back to client" << std::endl;
    //     return -1;
    // }

    std::cout << "done invoking handle___cudaPopCallConfiguration" << std::endl;

    return 0;
}

// Function pointer type for __cudaPushCallConfiguration
typedef unsigned (*__cudaPushCallConfiguration_type)(dim3 gridDim, dim3 blockDim,
                                                     size_t sharedMem, cudaStream_t stream);

int handle___cudaPushCallConfiguration(void *conn)
{
    dim3 gridDim, blockDim;
    size_t sharedMem;
    cudaStream_t stream;

    std::cout << "received handle___cudaPushCallConfiguration" << std::endl;

    // Read the grid dimensions from the client
    if (rpc_read(conn, &gridDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to read grid dimensions" << std::endl;
        return -1;
    }

    // Read the block dimensions from the client
    if (rpc_read(conn, &blockDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to read block dimensions" << std::endl;
        return -1;
    }

    // Read the shared memory size from the client
    if (rpc_read(conn, &sharedMem, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read shared memory size" << std::endl;
        return -1;
    }

    // Read the CUDA stream from the client
    if (rpc_read(conn, &stream, sizeof(cudaStream_t)) < 0)
    {
        std::cerr << "Failed to read CUDA stream" << std::endl;
        return -1;
    }

    // End the request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Find the original __cudaPushCallConfiguration function using dlsym
    __cudaPushCallConfiguration_type orig;
    orig = (__cudaPushCallConfiguration_type)dlsym(RTLD_NEXT, "__cudaPushCallConfiguration");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaPushCallConfiguration: " << dlerror() << std::endl;
        return -1;
    }

    // Call the original function
    unsigned result = orig(gridDim, blockDim, sharedMem, stream);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    // Write the result back to the client
    // if (rpc_write(conn, &result, sizeof(unsigned)) < 0)
    // {
    //     std::cerr << "Failed to write result back to client" << std::endl;
    //     return -1;
    // }

    std::cout << "finalized handle___cudaPushCallConfiguration" << std::endl;

    return 0;
}


typedef void (*__cudaInitModule_type)(void **fatCubinHandle);

void __cudaInitModule(void **fatCubinHandle) {
    std::cerr << "calling __cudaInitModule" << std::endl;
}

