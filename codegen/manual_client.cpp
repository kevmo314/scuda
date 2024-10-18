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
const char* cudaGetErrorString(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess: No errors";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue: Invalid value";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation: Out of memory";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError: Initialization error";
        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure: Launch failure";
        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure: Launch failure of a previous kernel";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout: Launch timed out";
        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources: Launch exceeded resources";
        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction: Invalid device function";
        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration: Invalid configuration";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice: Invalid device";
        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection: Invalid memory copy direction";
        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver: CUDA driver is insufficient for the runtime version";
        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration: Missing configuration";
        case cudaErrorNoDevice:
            return "cudaErrorNoDevice: No CUDA-capable device is detected";
        case cudaErrorArrayIsMapped:
            return "cudaErrorArrayIsMapped: Array is already mapped";
        case cudaErrorAlreadyMapped:
            return "cudaErrorAlreadyMapped: Resource is already mapped";
        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice: No kernel image is available for the device";
        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable: Uncorrectable ECC error detected";
        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound: Shared object symbol not found";
        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed: Shared object initialization failed";
        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit: Unsupported limit";
        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName: Duplicate global variable name";
        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName: Duplicate texture name";
        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName: Duplicate surface name";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable: All devices are busy or unavailable";
        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage: The kernel image is invalid";
        case cudaErrorInvalidSource:
            return "cudaErrorInvalidSource: The device kernel source is invalid";
        case cudaErrorFileNotFound:
            return "cudaErrorFileNotFound: File not found";
        case cudaErrorInvalidPtx:
            return "cudaErrorInvalidPtx: The PTX is invalid";
        case cudaErrorInvalidGraphicsContext:
            return "cudaErrorInvalidGraphicsContext: Invalid OpenGL or DirectX context";
        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle: Invalid resource handle";
        case cudaErrorNotReady:
            return "cudaErrorNotReady: CUDA operations are not ready";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress: An illegal memory access occurred";
        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue: Invalid pitch value";
        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol: Invalid symbol";
        case cudaErrorUnknown:
            return "cudaErrorUnknown: Unknown error";
        // Add any other CUDA error codes that are missing
        default:
            return "Unknown CUDA error";
    }
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
    if (rpc_wait_for_response() < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    // End the request and get the return value
    if (rpc_end_request(&return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

extern "C" void **__cudaRegisterFatBinary(void **fatCubin)
{
    cudaError_t return_value;
    std::cout << "Intercepted __cudaRegisterFatBinary data:: " << std::endl;

    // Start the RPC request
    if (rpc_start_request(RPC___cudaRegisterFatBinary) < 0)
    {
        std::cerr << "Failed to start RPC request for __cudaRegisterFatBinary" << std::endl;
        return nullptr;
    }

    if (rpc_wait_for_response() < 0)
    {
        std::cerr << "Failed to get response from server for __cudaRegisterFatBinary" << std::endl;
        return nullptr;
    }

    if (rpc_read(&fatCubin, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to read the fatCubin from the server" << std::endl;
        return nullptr;
    }

    // End the request
    if (rpc_end_request(&return_value) < 0)
    {
        std::cerr << "Failed to end RPC request for __cudaRegisterFatBinary" << std::endl;
        return nullptr;
    }

    std::cout << "finalized __cudaRegisterFatBinary data:: " << fatCubin << std::endl;

    return fatCubin;
}

extern "C"
{
    void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
    {
        cudaError_t return_value;

        // Start the RPC request for __cudaRegisterFatBinaryEnd
        int request_id = rpc_start_request(RPC___cudaRegisterFatBinaryEnd);
        if (request_id < 0)
        {
            std::cerr << "Failed to start RPC request" << std::endl;
            return;
        }

        // Wait for the server's response
        if (rpc_wait_for_response() < 0)
        {
            std::cerr << "Failed waiting for response" << std::endl;
            return;
        }

        // Read the fatCubinHandle back from the server
        if (rpc_read(&fatCubinHandle, sizeof(void *)) < 0)
        {
            std::cerr << "Failed reading fatCubinHandle back" << std::endl;
            return;
        }

        // End the request and check for any errors
        if (rpc_end_request(&return_value) < 0)
        {
            std::cerr << "Failed to end request" << std::endl;
            return;
        }

        return;
    }
}

extern "C" void __cudaInitModule(void **fatCubinHandle) {
  std::cout << "__cudaInitModule writing data..." << std::endl;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  std::cout << "__cudaUnregisterFatBinary writing data..." << std::endl;
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

        std::cout << "Intercepted __cudaRegisterFunction data:: " << fatCubinHandle << std::endl;

        int request_id = rpc_start_request(RPC___cudaRegisterFunction);
        if (request_id < 0)
        {
            std::cerr << "Failed to start RPC request" << std::endl;
            return;
        }

        // if (rpc_write(&fatCubinHandle, sizeof(void *)) < 0)
        // {
        //     std::cerr << "Failed writing fatCubinHandle" << std::endl;
        //     return;
        // }

        // if (rpc_write(&hostFun, sizeof(const char *)) < 0)
        // {
        //     std::cerr << "Failed writing hostFun" << std::endl;
        //     return;
        // }

        // if (rpc_write(&deviceFun, sizeof(char *)) < 0)
        // {
        //     std::cerr << "Failed writing deviceFun" << std::endl;
        //     return;
        // }

        // if (rpc_write(&deviceName, sizeof(const char *)) < 0)
        // {
        //     std::cerr << "Failed writing deviceName" << std::endl;
        //     return;
        // }

        // if (rpc_write(&thread_limit, sizeof(int)) < 0)
        // {
        //     std::cerr << "Failed writing thread_limit" << std::endl;
        //     return;
        // }

        // if (rpc_write(&tid, sizeof(uint3)) < 0)
        // {
        //     std::cerr << "Failed writing tid" << std::endl;
        //     return;
        // }

        // if (rpc_write(&bid, sizeof(uint3)) < 0)
        // {
        //     std::cerr << "Failed writing bid" << std::endl;
        //     return;
        // }

        // if (rpc_write(&bDim, sizeof(dim3)) < 0)
        // {
        //     std::cerr << "Failed writing bDim" << std::endl;
        //     return;
        // }

        // if (rpc_write(&gDim, sizeof(dim3)) < 0)
        // {
        //     std::cerr << "Failed writing gDim" << std::endl;
        //     return;
        // }

        // if (rpc_write(&wSize, sizeof(int)) < 0)
        // {
        //     std::cerr << "Failed writing wSize" << std::endl;
        //     return;
        // }

        if (rpc_wait_for_response() < 0)
        {
            std::cerr << "Failed waiting for response" << std::endl;
            return;
        }

        if (rpc_read(&fatCubinHandle, sizeof(void *)) < 0)
        {
            std::cerr << "Failed reading fatCubinHandle" << std::endl;
            return;
        }

        if (rpc_read(&hostFun, sizeof(const char *)) < 0)
        {
            std::cerr << "Failed reading hostFun" << std::endl;
            return;
        }

        if (rpc_read(&deviceFun, sizeof(char *)) < 0)
        {
            std::cerr << "Failed reading deviceFun" << std::endl;
            return;
        }

        if (rpc_read(&deviceName, sizeof(const char *)) < 0)
        {
            std::cerr << "Failed reading deviceName" << std::endl;
            return;
        }

        if (rpc_read(&thread_limit, sizeof(int)) < 0)
        {
            std::cerr << "Failed reading thread_limit" << std::endl;
            return;
        }

        if (rpc_read(&tid, sizeof(uint3)) < 0)
        {
            std::cerr << "Failed reading tid" << std::endl;
            return;
        }

        if (rpc_read(&bid, sizeof(uint3)) < 0)
        {
            std::cerr << "Failed reading bid" << std::endl;
            return;
        }

        if (rpc_read(&bDim, sizeof(dim3)) < 0)
        {
            std::cerr << "Failed reading bDim" << std::endl;
            return;
        }

        if (rpc_read(&gDim, sizeof(dim3)) < 0)
        {
            std::cerr << "Failed reading gDim" << std::endl;
            return;
        }

        if (rpc_read(&wSize, sizeof(int)) < 0)
        {
            std::cerr << "Failed reading wSize" << std::endl;
            return;
        }

        if (rpc_end_request(&return_value) < 0)
        {
            std::cerr << "Failed to end request" << std::endl;
            return;
        }

        return;
    }
}
