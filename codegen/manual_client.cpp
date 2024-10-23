#include <nvml.h>
#include <cuda.h>
#include <iostream>
#include <dlfcn.h>
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

    std::cout << "Device function: " << func << std::endl;

    if (rpc_write(&func, sizeof(const void *)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&gridDim, sizeof(dim3)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&blockDim, sizeof(dim3)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&sharedMem, sizeof(size_t)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(&stream, sizeof(cudaStream_t)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    int num_args = 0;
    while (args[num_args] != nullptr)
        num_args++;

    if (rpc_write(&num_args, sizeof(int)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    for (int i = 0; i < num_args; ++i)
    {
        size_t arg_size = sizeof(args[i]);
        std::cout << "sending argument " << i << " of size " << arg_size << " bytes" << std::endl;

        // Send the argument size
        if (rpc_write(&arg_size, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(args[i], arg_size) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_wait_for_response() < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_end_request(&return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

extern "C" void **__cudaRegisterFatBinary(void **fatCubin)
{
    cudaError_t return_value;
    void **result;

    std::cout << "Intercepted __cudaRegisterFatBinary with fatCubin: " << fatCubin << std::endl;

    if (rpc_start_request(RPC___cudaRegisterFatBinary) < 0)
    {
        std::cerr << "Failed to start RPC request for __cudaRegisterFatBinary" << std::endl;
        return nullptr;
    }

    if (rpc_write(&fatCubin, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to write fatCubin to server" << std::endl;
        return nullptr;
    }

    if (rpc_wait_for_response() < 0)
    {
        std::cerr << "Failed to get response from server for __cudaRegisterFatBinary" << std::endl;
        return nullptr;
    }

    if (rpc_read(&result, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to read the result fatCubin from the server" << std::endl;
        return nullptr;
    }

    if (rpc_end_request(&return_value) < 0)
    {
        std::cerr << "Failed to end RPC request for __cudaRegisterFatBinary" << std::endl;
        return nullptr;
    }

    return result;
}

extern "C"
{
    void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
    {
        cudaError_t return_value;

        int request_id = rpc_start_request(RPC___cudaRegisterFatBinaryEnd);
        if (request_id < 0)
        {
            std::cerr << "Failed to start RPC request" << std::endl;
            return;
        }

        if (rpc_write(&fatCubinHandle, sizeof(const void *)) < 0)
        {
            return;
        }

        if (rpc_wait_for_response() < 0)
        {
            std::cerr << "Failed waiting for response" << std::endl;
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

extern "C" unsigned __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                               size_t *sharedMem, void *stream)
{
    cudaError_t res;

    std::cout << "received __cudaPopCallConfiguration." << std::endl;

    int request_id = rpc_start_request(RPC___cudaPopCallConfiguration);
    if (request_id < 0)
    {
        std::cerr << "Failed to start RPC request" << std::endl;
        return 0;
    }

    std::cout << "hereee" << std::endl;

    if (rpc_write(&gridDim, sizeof(dim3 *)) < 0)
    {
        std::cerr << "Failed to send gridDim pointer to server" << std::endl;
        return 0;
    }

    if (rpc_write(&blockDim, sizeof(dim3 *)) < 0)
    {
        std::cerr << "Failed to send blockDim pointer to server" << std::endl;
        return 0;
    }

    if (rpc_write(&sharedMem, sizeof(size_t *)) < 0)
    {
        std::cerr << "Failed to send sharedMem pointer to server" << std::endl;
        return 0;
    }

    if (rpc_write(&stream, sizeof(void *)) < 0)
    {
        std::cerr << "Failed to send stream pointer to server" << std::endl;
        return 0;
    }

    std::cout << "finished writing" << std::endl;

    if (rpc_wait_for_response() < 0)
    {
        std::cerr << "Failed to wait for response from server" << std::endl;
        return 0;
    }

    if (rpc_read(&gridDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to read gridDim from server" << std::endl;
        return 0;
    }

    if (rpc_read(&blockDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to read blockDim from server" << std::endl;
        return 0;
    }

    if (rpc_read(&sharedMem, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read sharedMem from server" << std::endl;
        return 0;
    }

    if (rpc_read(&stream, sizeof(void *)) < 0)
    {
        std::cerr << "Failed to read stream from server" << std::endl;
        return 0;
    }

    // if (rpc_read(&return_value, sizeof(unsigned)) < 0)
    // {
    //     std::cerr << "Failed to read stream from server" << std::endl;
    //     return 0;
    // }

    if (rpc_end_request(&res) < 0)
    {
        std::cerr << "Failed to retrieve return value from server" << std::endl;
        return 0;
    }

    std::cout << "done with __cudaPopCallConfiguration." << std::endl;

    return 0;
}

extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream) {
  cudaError_t res;
    std::cerr << "received __cudaPushCallConfiguration" << std::endl;

    // Start the RPC request
    int request_id = rpc_start_request(RPC___cudaPushCallConfiguration);
    if (request_id < 0)
    {
        std::cerr << "Failed to start RPC request" << std::endl;
        return 0;
    }

    // Write the grid dimensions
    if (rpc_write(&gridDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to write grid dimensions" << std::endl;
        return 0;
    }

    // Write the block dimensions
    if (rpc_write(&blockDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to write block dimensions" << std::endl;
        return 0;
    }

    // Write the shared memory size
    if (rpc_write(&sharedMem, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to write shared memory size" << std::endl;
        return 0;
    }

    // Write the stream
    if (rpc_write(&stream, sizeof(cudaStream_t)) < 0)
    {
        std::cerr << "Failed to write CUDA stream" << std::endl;
        return 0;
    }

    // Wait for a response from the server
    if (rpc_wait_for_response() < 0)
    {
        std::cerr << "Failed to wait for server response" << std::endl;
        return 0;
    }

    // if (rpc_read(&return_value, sizeof(unsigned)) < 0)
    // {
    //     std::cerr << "Failed to read stream from server" << std::endl;
    //     return 0;
    // }

    // Get the return value from the server
    if (rpc_end_request(&res) < 0)
    {
        std::cerr << "Failed to retrieve return value" << std::endl;
        return 0;
    }

    std::cerr << "done with __cudaPushCallConfiguration" << std::endl;

    return 0;
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

        std::cout << "Intercepted __cudaRegisterFunction" << deviceFun << std::endl;

        int request_id = rpc_start_request(RPC___cudaRegisterFunction);
        if (request_id < 0)
        {
            std::cerr << "Failed to start RPC request" << std::endl;
            return;
        }

        if (rpc_write(&fatCubinHandle, sizeof(void *)) < 0)
        {
            std::cerr << "Failed writing fatCubinHandle" << std::endl;
            return;
        }

        if (rpc_write(&hostFun, sizeof(const char *)) < 0)
        {
            std::cerr << "Failed writing hostFun" << std::endl;
            return;
        }

        if (rpc_write(&deviceFun, sizeof(char *)) < 0)
        {
            std::cerr << "Failed writing deviceFun" << std::endl;
            return;
        }

        if (rpc_write(&deviceName, sizeof(const char *)) < 0)
        {
            std::cerr << "Failed writing deviceName" << std::endl;
            return;
        }

        if (rpc_write(&thread_limit, sizeof(int)) < 0)
        {
            std::cerr << "Failed writing thread_limit" << std::endl;
            return;
        }

        if (rpc_write(&tid, sizeof(uint3)) < 0)
        {
            std::cerr << "Failed writing tid" << std::endl;
            return;
        }

        if (rpc_write(&bid, sizeof(uint3)) < 0)
        {
            std::cerr << "Failed writing bid" << std::endl;
            return;
        }

        if (rpc_write(&bDim, sizeof(dim3)) < 0)
        {
            std::cerr << "Failed writing bDim" << std::endl;
            return;
        }

        if (rpc_write(&gDim, sizeof(dim3)) < 0)
        {
            std::cerr << "Failed writing gDim" << std::endl;
            return;
        }

        if (rpc_write(&wSize, sizeof(int)) < 0)
        {
            std::cerr << "Failed writing wSize" << std::endl;
            return;
        }

        if (rpc_wait_for_response() < 0)
        {
            std::cerr << "Failed waiting for response" << std::endl;
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
