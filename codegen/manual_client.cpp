#include <nvml.h>
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <dlfcn.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"
#include "ptx_fatbin.hpp"

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

        if (rpc_read(0, dst, count) < 0)
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

        if (rpc_write(0, src, count) < 0)
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

        if (rpc_read(0, dst, count) < 0)
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

        if (rpc_write(0, src, count) < 0)
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
const char *cudaGetErrorString(cudaError_t error)
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

    std::cout << "starting function: " << &func << std::endl;

    // Start the RPC request
    int request_id = rpc_start_request(0, RPC_cudaLaunchKernel);
    if (request_id < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &func, sizeof(const void *)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &gridDim, sizeof(dim3)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &blockDim, sizeof(dim3)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &sharedMem, sizeof(size_t)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_write(0, &stream, sizeof(cudaStream_t)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    int num_args = 0;
    while (args[num_args] != nullptr)
        num_args++;

    if (rpc_write(0, &num_args, sizeof(int)) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    for (int i = 0; i < num_args; ++i)
    {
        size_t arg_size = sizeof(args[i]);
        std::cout << "sending argument " << i << " of size " << arg_size << " bytes" << std::endl;

        // Send the argument size
        if (rpc_write(0, &arg_size, sizeof(size_t)) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }

        if (rpc_write(0, args[i], arg_size) < 0)
        {
            return cudaErrorDevicesUnavailable;
        }
    }

    if (rpc_wait_for_response(0) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    if (rpc_end_request(0, &return_value) < 0)
    {
        return cudaErrorDevicesUnavailable;
    }

    return return_value;
}

// void ptx_from_fatbin(const void *cubin_ptr)
// {
//     assert(cubin_ptr != 0);
//     if(*(int*)cubin_ptr == __cudaFatMAGIC)
//     {
//       __cudaFatCudaBinary *binary = (__cudaFatCudaBinary *)cubin_ptr;
//       assert(binary->ident != 0);
//       std::cout << binary->ident << std::endl;
//       assert(binary->ptx != 0);
//       assert(binary->ptx->ptx != 0);
//       int i=0;
//       while(binary->ptx[i].ptx != 0){
//         assert(binary->ptx[i].gpuProfileName != 0);
//         std::cout << binary->ptx[i].gpuProfileName << std::endl;
//         assert(binary->ptx[i].ptx != 0);
//         std::cout << binary->ptx[i].ptx << std::endl;
//         i++;
//       }
//     }
//     else if(*(unsigned*)cubin_ptr == __cudaFatMAGIC2)
//     {
// 	__cudaFatCudaBinary2* binary = (__cudaFatCudaBinary2*) cubin_ptr;

//         printf("%x ", binary->magic);
//         printf("%x ", binary->version);
//         printf("%p ", binary->fatbinData);
//         printf("%p ", binary->f);
//         printf("\n");

// 	__cudaFatCudaBinary2Header* header = (__cudaFatCudaBinary2Header*) binary->fatbinData;
// 	char* base = (char*)(header + 1);
// 	long long unsigned int offset = 0;
// 	__cudaFatCudaBinary2EntryRec* entry = (__cudaFatCudaBinary2EntryRec*)(base);

//         printf("%x ", header->magic);
//         printf("%x ", header->version);
//         printf("%llx ", header->length);
//         printf("\n");

// 	while (!(entry->type & FATBIN_2_PTX) && offset < header->length)
//         {
// 	  entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
// 	  printf("%x ", entry->type);
// 	  printf("%x ", entry->binary);
// 	  printf("%llx ", entry->binarySize);
// 	  printf("%x ", entry->unknown2);
// 	  printf("%x ", entry->kindOffset);
// 	  printf("%x ", entry->unknown3);
// 	  printf("%x ", entry->unknown4);
// 	  printf("%x ", entry->name);
// 	  printf("%x ", entry->nameSize);
// 	  printf("%llx ", entry->flags);
// 	  printf("%llx ", entry->unknown7);
// 	  printf("%llx ", entry->uncompressedBinarySize);
// 	  printf("\n");

// 	  printf("%s\n", (char *)((char*)entry + entry->name));
// 	  printf("%lld %lld bytes\n", entry->binary, entry->binarySize);

// 	  offset += entry->binary + entry->binarySize;
// 	}
//         printf("Found PTX\n");
//         assert(entry->type & FATBIN_2_PTX);
// 	printf("%x ", entry->type);
// 	printf("%x ", entry->binary);
// 	printf("%llx ", entry->binarySize);
// 	printf("%x ", entry->unknown2);
// 	printf("%x ", entry->kindOffset);
// 	printf("%x ", entry->unknown3);
// 	printf("%x ", entry->unknown4);
// 	printf("%x ", entry->name);
// 	printf("%x ", entry->nameSize);
// 	printf("%llx ", entry->flags);
// 	printf("%llx ", entry->unknown7);
// 	printf("%llx ", entry->uncompressedBinarySize);
// 	printf("\n");

// 	printf("%s\n", (char *)((char*)entry + entry->name));
// 	printf("%ld (C)%lld B (U)%lld B\n", entry->binary, entry->binarySize, entry->uncompressedBinarySize);
//     }
//     else
//     {
//         printf("Unrecognized CUDA FAT MAGIC 0x%x\n", *(int*)cubin_ptr);
//     }
// }

extern "C" void **__cudaRegisterFatBinary(void *fatCubin)
{
    void **p;
    int return_value;

    if (rpc_start_request(0, RPC___cudaRegisterFatBinary) < 0)
        return nullptr;

    if (*(unsigned *)fatCubin == __cudaFatMAGIC2)
    {
        __cudaFatCudaBinary2 *binary = (__cudaFatCudaBinary2 *)fatCubin;

        if (rpc_write(0, &binary->magic, sizeof(int)) < 0 ||
            rpc_write(0, &binary->version, sizeof(int)) < 0)
            return nullptr;

        __cudaFatCudaBinary2Header *header = (__cudaFatCudaBinary2Header *)binary->fatbinData;

        unsigned long long size = sizeof(__cudaFatCudaBinary2Header) + header->length;

        if (rpc_write(0, &size, sizeof(unsigned long long)) < 0 ||
            rpc_write(0, header, size) < 0)
            return nullptr;
    }

    if (rpc_wait_for_response(0) < 0 ||
        rpc_read(0, &p, sizeof(void **)) < 0 ||
        rpc_end_request(0, &return_value) < 0)
        return nullptr;

    return p;
}

extern "C"
{
    void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
    {
        void *return_value;

        std::cout << "!!!! " << fatCubinHandle << std::endl;

        int request_id = rpc_start_request(0, RPC___cudaRegisterFatBinaryEnd);
        if (request_id < 0)
        {
            std::cerr << "Failed to start RPC request" << std::endl;
            return;
        }

        if (rpc_write(0, &fatCubinHandle, sizeof(const void *)) < 0)
        {
            return;
        }

        if (rpc_wait_for_response(0) < 0)
        {
            std::cerr << "Failed waiting for response" << std::endl;
            return;
        }

        // End the request and check for any errors
        if (rpc_end_request(0, &return_value) < 0)
        {
            std::cerr << "Failed to end request" << std::endl;
            return;
        }

        return;
    }
}

extern "C" void __cudaInitModule(void **fatCubinHandle)
{
    std::cout << "__cudaInitModule writing data..." << std::endl;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    //   std::cout << "__cudaUnregisterFatBinary writing data..." << std::endl;
}

extern "C" cudaError_t __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                   size_t sharedMem, cudaStream_t stream)
{
    cudaError_t res;

    if (rpc_start_request(0, RPC___cudaPushCallConfiguration) < 0 ||
        rpc_write(0, &gridDim, sizeof(dim3)) < 0 ||
        rpc_write(0, &blockDim, sizeof(dim3)) < 0 ||
        rpc_write(0, &sharedMem, sizeof(size_t)) < 0 ||
        rpc_write(0, &stream, sizeof(cudaStream_t)) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_request(0, &res) < 0)
        return cudaErrorDevicesUnavailable;

    return res;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                  size_t *sharedMem, cudaStream_t *stream)
{
    cudaError_t res;

    if (rpc_start_request(0, RPC___cudaPopCallConfiguration) < 0 ||
        rpc_wait_for_response(0) < 0 ||
        rpc_read(0, gridDim, sizeof(dim3)) < 0 ||
        rpc_read(0, blockDim, sizeof(dim3)) < 0 ||
        rpc_read(0, sharedMem, sizeof(size_t)) < 0 ||
        rpc_read(0, stream, sizeof(cudaStream_t)) < 0 ||
        rpc_end_request(0, &res) < 0)
        return cudaErrorDevicesUnavailable;

    return res;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun,
                                       char *deviceFun,
                                       const char *deviceName,
                                       int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    std::cout << "Intercepted __cudaRegisterFunction for deviceName: " << deviceName << std::endl;

    void *return_value;

    size_t deviceFunLen = strlen(deviceFun) + 1;
    size_t deviceNameLen = strlen(deviceName) + 1;

    uint8_t mask = 0;
    if (tid != nullptr)
        mask |= 1 << 0;
    if (bid != nullptr)
        mask |= 1 << 1;
    if (bDim != nullptr)
        mask |= 1 << 2;
    if (gDim != nullptr)
        mask |= 1 << 3;
    if (wSize != nullptr)
        mask |= 1 << 4;

    printf("fatCubeHandle: %p\n", fatCubinHandle);

    if (rpc_start_request(0, RPC___cudaRegisterFunction) < 0 ||
        rpc_write(0, &fatCubinHandle, sizeof(void **)) < 0 ||
        rpc_write(0, &hostFun, sizeof(const char *)) < 0 ||
        rpc_write(0, &deviceFunLen, sizeof(size_t)) < 0 ||
        rpc_write(0, deviceFun, deviceFunLen) < 0 ||
        rpc_write(0, &deviceNameLen, sizeof(size_t)) < 0 ||
        rpc_write(0, deviceName, deviceNameLen) < 0 ||
        rpc_write(0, &thread_limit, sizeof(int)) < 0 ||
        rpc_write(0, &mask, sizeof(uint8_t)) < 0 ||
        (tid != nullptr && rpc_write(0, tid, sizeof(uint3)) < 0) ||
        (bid != nullptr && rpc_write(0, bid, sizeof(uint3)) < 0) ||
        (bDim != nullptr && rpc_write(0, bDim, sizeof(dim3)) < 0) ||
        (gDim != nullptr && rpc_write(0, gDim, sizeof(dim3)) < 0) ||
        (wSize != nullptr && rpc_write(0, wSize, sizeof(int)) < 0) ||
        rpc_wait_for_response(0) < 0 ||
        rpc_end_request(0, &return_value) < 0)
        return;
}

extern "C"
{
    void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size, int constant, int global)
    {
        void *return_value;

        std::cout << "Intercepted __cudaRegisterVar for deviceName: " << deviceName << std::endl;

        // Start the RPC request
        int request_id = rpc_start_request(0, RPC___cudaRegisterVar);
        if (request_id < 0)
        {
            std::cerr << "Failed to start RPC request" << std::endl;
            return;
        }

        // Write fatCubinHandle
        if (rpc_write(0, &fatCubinHandle, sizeof(void *)) < 0)
        {
            std::cerr << "Failed writing fatCubinHandle" << std::endl;
            return;
        }

        // Send hostVar length and data
        size_t hostVarLen = strlen(hostVar) + 1;
        if (rpc_write(0, &hostVarLen, sizeof(size_t)) < 0)
        {
            std::cerr << "Failed to send hostVar length" << std::endl;
            return;
        }
        if (rpc_write(0, hostVar, hostVarLen) < 0)
        {
            std::cerr << "Failed writing hostVar" << std::endl;
            return;
        }

        // Send deviceAddress length and data
        size_t deviceAddressLen = strlen(deviceAddress) + 1;
        if (rpc_write(0, &deviceAddressLen, sizeof(size_t)) < 0)
        {
            std::cerr << "Failed to send deviceAddress length" << std::endl;
            return;
        }
        if (rpc_write(0, deviceAddress, deviceAddressLen) < 0)
        {
            std::cerr << "Failed writing deviceAddress" << std::endl;
            return;
        }

        // Send deviceName length and data
        size_t deviceNameLen = strlen(deviceName) + 1;
        if (rpc_write(0, &deviceNameLen, sizeof(size_t)) < 0)
        {
            std::cerr << "Failed to send deviceName length" << std::endl;
            return;
        }
        if (rpc_write(0, deviceName, deviceNameLen) < 0)
        {
            std::cerr << "Failed writing deviceName" << std::endl;
            return;
        }

        // Write the rest of the arguments
        if (rpc_write(0, &ext, sizeof(int)) < 0)
        {
            std::cerr << "Failed writing ext" << std::endl;
            return;
        }

        if (rpc_write(0, &size, sizeof(size_t)) < 0)
        {
            std::cerr << "Failed writing size" << std::endl;
            return;
        }

        if (rpc_write(0, &constant, sizeof(int)) < 0)
        {
            std::cerr << "Failed writing constant" << std::endl;
            return;
        }

        if (rpc_write(0, &global, sizeof(int)) < 0)
        {
            std::cerr << "Failed writing global" << std::endl;
            return;
        }

        // Wait for a response from the server
        if (rpc_wait_for_response(0) < 0)
        {
            std::cerr << "Failed waiting for response" << std::endl;
            return;
        }

        if (rpc_end_request(0, &return_value) < 0)
        {
            std::cerr << "Failed to end request" << std::endl;
            return;
        }

        std::cout << "Done with __cudaRegisterVar for deviceName: " << deviceName << std::endl;
    }
}
