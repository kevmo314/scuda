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
#include "ptx_fatbin.hpp"

extern int rpc_read(const void *conn, void *data, const std::size_t size);
extern int rpc_end_request(const void *conn);
extern int rpc_start_response(const void *conn, const int request_id);
extern int rpc_write(const void *conn, const void *data, const std::size_t size);
extern int rpc_end_response(const void *conn, void *return_value);

FILE *__cudart_trace_output_stream = stdout;

int handle_cudaMemcpy(void *conn)
{
    cudaError_t result;
    void *dst;

    std::cout << "calling cudaMemcpy" << std::endl;

    enum cudaMemcpyKind kind;
    if (rpc_read(conn, &kind, sizeof(enum cudaMemcpyKind)) < 0)
    {
        return -1;
    }

    std::cout << "hmmm " << kind << std::endl;

    if (kind == cudaMemcpyDeviceToHost)
    {
        std::cout << "IN HERE cudaMemcpy" << std::endl;
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

        std::cout << "call... cudaMemcpy" << std::endl;

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

        std::cout << "about to end... cudaMemcpy" << std::endl;
    }
    else
    {
        std::cout << "host to device ... cudaMemcpy" << std::endl;
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

        std::cout << "finishing" << std::endl;
    }

    if (rpc_end_response(conn, &result) < 0)
        return -1;

    std::cout << "end cudaMemcpy" << std::endl;
    return 0;
}

int handle_cudaMemcpyAsync(void *conn)
{
    cudaError_t result;
    void *dst;

    std::cout << "calling cudaMemcpyAsync" << std::endl;

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

    std::cout << "end cudaMemcpyAsync" << std::endl;    

    return 0;
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

    std::cout << "cudaLaunchKernel request incoming!" << std::endl;

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
    int request_id;
    void *res;
    void ** fatBinary;
    std::cout << "REQUEST!!!"  << std::endl;
    if (rpc_read(conn, &fatBinary, sizeof(void **)) < 0) {
        std::cerr << "Failed to read fat binary header" << std::endl;
        return -1;
    }

    std::cout << "fat binary..." << fatBinary << std::endl;

    // unsigned long long hlen;
    // unsigned long long int blen;
    // int magic;
    // int version;

    // if (rpc_read(conn, &magic, sizeof(int)) < 0) 
    // {
    //     std::cerr << "Failed to read magic" << std::endl;
    //     return -1;
    // }
    
    // if (rpc_read(conn, &version, sizeof(int)) < 0) 
    // {
    //     std::cerr << "Failed to read magic" << std::endl;
    //     return -1;
    // }

    // printf("MAGIC: %x ", magic);
    // printf("BINARY VERSION: %x ", version);
    // printf("\n");

    // __cudaFatCudaBinary2Header* header;

    // if (rpc_read(conn, &hlen, sizeof(unsigned long long int)) < 0) 
    // {
    //     std::cerr << "Failed to read version for __cudaFatMAGIC2" << std::endl;
    //     return -1;
    // }

    // std::cout << "done reading hlen: " << hlen << std::endl;

    // header = (__cudaFatCudaBinary2Header*)malloc(hlen);
    // if (rpc_read(conn, header, hlen) < 0) {
    //     std::cerr << "Failed to read fat binary header" << std::endl;
    //     return -1;
    // }

    // unsigned long long int length = header->length;

    // char* fatbinData = (char*)malloc(length);
    // if (!fatbinData) {
    //     std::cerr << "Failed to allocate memory for fatbin data" << std::endl;
    //     return -1;
    // }

    // char* base = fatbinData;
    // uint64_t offset = 0;
    // __cudaFatCudaBinary2EntryRec* entry = (__cudaFatCudaBinary2EntryRec*)(base);

    // std::cout << "starting read entries... " << std::endl;

    // while (offset < length) {
    //     std::cout << "reading entry... " << std::endl;

    //     unsigned long long esize;
    //     if (rpc_read(conn, &esize, sizeof((unsigned long long)__cudaFatCudaBinary2EntryRec::binarySize)) < 0) {
    //         std::cerr << "Failed to read entry size from client" << std::endl;
    //         free(fatbinData);
    //         return -1;
    //     }

    //     printf("ENTRY BINARY SIZE: %llx ", esize);

    //     char* readEntry = (char*)malloc(esize);
    //     if (!readEntry) {
    //         std::cerr << "Failed to allocate memory for binary data" << std::endl;
    //         return -1;
    //     }

    //     if (rpc_read(conn, readEntry, esize) < 0) {
    //         std::cerr << "Failed to read binary data" << std::endl;
    //         return -1;
    //     }

    //     entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
    //     memcpy(entry, readEntry, esize);

    //     printf("ENTRY BINARY AND SIZE: %lld %lld bytes\n", entry->binary, entry->binarySize);

    //     // move to the next entry
    //     offset += entry->binary + entry->binarySize;

    //     free(readEntry);
    // }

    // __cudaFatCudaBinary2* rebuiltBinary = (__cudaFatCudaBinary2*)malloc(sizeof(__cudaFatCudaBinary2));
    // if (!rebuiltBinary) {
    //     std::cerr << "Failed to allocate memory for rebuiltBinary" << std::endl;
    //     free(fatbinData);
    //     free(header);
    //     return -1;
    // }

    // rebuiltBinary->magic = magic;
    // rebuiltBinary->version = version;
    // rebuiltBinary->fatbinData = (const unsigned long long*)header;
    // rebuiltBinary->f = nullptr;  // set to null unless needed

    // __cudaFatCudaBinary2Header* vheader = (__cudaFatCudaBinary2Header*)rebuiltBinary->fatbinData;

    // std::cout << "Validating rebuilt binary..." << std::endl;
    // std::cout << "Header Magic: " << std::hex << vheader->magic << std::endl;
    // std::cout << "Header Version: " << vheader->version << std::endl;
    // std::cout << "Header Length: " << vheader->length << std::endl;

    // char* vbase = (char*)(vheader + 1); 
    // uint64_t voffset = 0;

    // while (voffset < vheader->length) {
    //     __cudaFatCudaBinary2EntryRec* entry = (__cudaFatCudaBinary2EntryRec*)(vbase + voffset);

    //     // Print out details about the entry
    //     std::cout << "Entry Type: " << entry->type << std::endl;
    //     std::cout << "Entry Binary: " << entry->binary << std::endl;
    //     std::cout << "Entry Binary Size: " << entry->binarySize << " bytes" << std::endl;

    //     if (entry->nameSize > 0) {
    //         char* name = (char*)entry + entry->name;  // Pointer to the entry name
    //         std::cout << "Entry Name: " << std::string(name, entry->nameSize) << std::endl;
    //     }

    //     // Move to the next entry in the fatbin
    //     voffset += entry->binary + entry->binarySize;
    // }

    std::cout << "Validation complete." << std::endl;

    void** (*fptr)(void *);
    fptr = (void** (*)(void *))dlsym(dlopen("libcudart.so", RTLD_NOW), "__cudaRegisterFatBinary");

    request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Call the original __cudaRegisterFatBinary with the rebuilt binary
    void** ret = fptr((void**)&fatBinary);

    std::cout << "called OG __cudaRegisterFatBinary... validating returned: " << ret << std::endl;

    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &ret, sizeof(void **)) < 0)
    {
        std::cerr << "Failed to write fatCubin result back to the client" << std::endl;
        return -1;
    }

    if (rpc_end_response(conn, &res) < 0)
        return -1;

    return 0;
}

typedef void (*__cudaRegisterFunction_type)(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);

int handle___cudaRegisterFunction(void *conn)
{
    void *res;
    void **fatCubinHandle;
    char *hostFun;
    char *deviceFun;
    char *deviceName;
    int thread_limit;
    uint3 *tid, *bid;
    dim3 *bDim, *gDim;
    int *wSize;

    // Reading the input parameters from the client
    if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        return -1;
    }

    size_t hostFunLen;
    if (rpc_read(conn, &hostFunLen, sizeof(size_t)) < 0)
    {
        return -1;
    }

    // Allocate memory for the hostFun string
    hostFun = (char *)malloc(hostFunLen);
    if (!hostFun)
    {
        std::cerr << "Failed to allocate memory for hostFun" << std::endl;
        return -1;
    }

    // Read the actual string data into the allocated memory
    if (rpc_read(conn, hostFun, hostFunLen) < 0)
    {
        free(hostFun);
        return -1;
    }

    size_t deviceFunLen;
    if (rpc_read(conn, &deviceFunLen, sizeof(size_t)) < 0)
    {
        return -1;
    }

    // Allocate memory for the hostFun string
    deviceFun = (char *)malloc(deviceFunLen);
    if (!deviceFun)
    {
        std::cerr << "Failed to allocate memory for hostFun" << std::endl;
        return -1;
    }

    // Read the actual string data into the allocated memory
    if (rpc_read(conn, deviceFun, deviceFunLen) < 0)
    {
        free(deviceFun);
        return -1;
    }

    size_t deviceNameLen;
    if (rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0)
    {
        return -1;
    }

    deviceName = (char *)malloc(deviceNameLen);
    if (!deviceName)
    {
        std::cerr << "Failed to allocate memory for hostFun" << std::endl;
        return -1;
    }

    // Read the actual string data into the allocated memory
    if (rpc_read(conn, deviceName, deviceNameLen) < 0)
    {
        free(deviceName);
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

    __cudaRegisterFunction_type orig;
    orig = (__cudaRegisterFunction_type)dlsym(RTLD_NEXT, "__cudaRegisterFunction");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaRegisterFunction" << std::endl;
        return -1;
    }

    std::cout << "Calling original __cudaRegisterFunction with parameters:" << std::endl;
    std::cout << "  fatCubinHandle: " << fatCubinHandle << std::endl;
    std::cout << "  hostFun: " << (hostFun ? hostFun : "null") << std::endl;
    std::cout << "  deviceFun: " << (deviceFun ? deviceFun : "null") << std::endl;
    std::cout << "  deviceName: " << (deviceName ? deviceName : "null") << std::endl;
    std::cout << "  thread_limit: " << thread_limit << std::endl;
    std::cout << "  tid: " << tid << std::endl;
    std::cout << "  bid: " << bid << std::endl;
    std::cout << "  bDim: " << bDim << std::endl;
    std::cout << "  gDim: " << gDim << std::endl;
    std::cout << "  wSize: " << wSize << std::endl;

    // Call the original __cudaRegisterFunction
    orig(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    if (rpc_end_response(conn, &res) < 0)
        return -1;

    return 0;
}


typedef void (*__cudaRegisterFatBinaryEnd_type)(void **fatCubinHandle);

int handle___cudaRegisterFatBinaryEnd(void *conn)
{
    void *res;
    void **fatCubinHandle;

    // Read the fatCubinHandle from the client
    if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        std::cerr << "Failed reading fatCubinHandle" << std::endl;
        return -1;
    }

    std::cout << "received cudaRegisterFatBinaryEnd: " << fatCubinHandle << std::endl;

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

    if (rpc_end_response(conn, &res) < 0)
        return -1;

    return 0;
}

typedef unsigned (*__cudaPopCallConfiguration_type)(dim3 *gridDim, dim3 *blockDim,
                                                    size_t *sharedMem, void *stream);

int handle___cudaPopCallConfiguration(void *conn)
{
    void *res;
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

    std::cout << "Calling original __cudaPopCallConfiguration with parameters:" << std::endl;
    std::cout << "  gridDim: " << gridDim << std::endl;
    std::cout << "  blockDim: " << blockDim << std::endl;
    std::cout << "  sharedMem: " << sharedMem << std::endl;
    std::cout << "  stream: " << stream << std::endl;

    // Call the original function to populate the data
    unsigned result = orig(gridDim, blockDim, sharedMem, stream);

    printf("The __cudaPopCallConfiguration value of num is: %u\n", result);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    // Send the updated data back to the client
    if (rpc_write(conn, &result, sizeof(unsigned)) < 0)
    {
        std::cerr << "Failed to send result to client" << std::endl;
        return -1;
    }

    if (rpc_end_response(conn, &res) < 0)
        return -1;

    return 0;
}

// Function pointer type for __cudaPushCallConfiguration
typedef unsigned (*__cudaPushCallConfiguration_type)(dim3 gridDim, dim3 blockDim,
                                                     size_t sharedMem, cudaStream_t stream);

int handle___cudaPushCallConfiguration(void *conn)
{
    void *res;
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

    std::cout << "Calling original __cudaPushCallConfiguration with parameters:" << std::endl;
    std::cout << "  gridDim: " << &gridDim << std::endl;
    std::cout << "  blockDim: " << &blockDim << std::endl;
    std::cout << "  sharedMem: " << &sharedMem << std::endl;
    std::cout << "  stream: " << &stream << std::endl;

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

    printf("The __cudaPushCallConfiguration value of num is: %u\n", result);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    if (rpc_write(conn, &result, sizeof(unsigned)) < 0)
    {
        std::cerr << "Failed to send result to client" << std::endl;
        return -1;
    }

    if (rpc_end_response(conn, &res) < 0)
        return -1;

    return 0;
}


typedef void (*__cudaInitModule_type)(void **fatCubinHandle);

void __cudaInitModule(void **fatCubinHandle) {
    std::cerr << "calling __cudaInitModule" << std::endl;
}

typedef void (*__cudaRegisterVar_type)(
    void **fatCubinHandle,
    char *hostVar,
    char *deviceAddress,
    const char *deviceName,
    int ext,
    size_t size,
    int constant,
    int global
);

int handle___cudaRegisterVar(void *conn)
{
    void *res;
    void **fatCubinHandle;
    char *hostVar;
    char *deviceAddress;
    char *deviceName;
    int ext;
    size_t size;
    int constant;
    int global;

    // Read the fatCubinHandle
    if (rpc_read(conn, &fatCubinHandle, sizeof(void *)) < 0)
    {
        std::cerr << "Failed reading fatCubinHandle" << std::endl;
        return -1;
    }

    // Read hostVar
    size_t hostVarLen;
    if (rpc_read(conn, &hostVarLen, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read hostVar length" << std::endl;
        return -1;
    }
    hostVar = (char *)malloc(hostVarLen);
    if (rpc_read(conn, hostVar, hostVarLen) < 0)
    {
        std::cerr << "Failed to read hostVar" << std::endl;
        return -1;
    }

    // Read deviceAddress
    size_t deviceAddressLen;
    if (rpc_read(conn, &deviceAddressLen, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read deviceAddress length" << std::endl;
        return -1;
    }
    deviceAddress = (char *)malloc(deviceAddressLen);
    if (rpc_read(conn, deviceAddress, deviceAddressLen) < 0)
    {
        std::cerr << "Failed to read deviceAddress" << std::endl;
        return -1;
    }

    // Read deviceName
    size_t deviceNameLen;
    if (rpc_read(conn, &deviceNameLen, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed to read deviceName length" << std::endl;
        return -1;
    }
    deviceName = (char *)malloc(deviceNameLen);
    if (rpc_read(conn, deviceName, deviceNameLen) < 0)
    {
        std::cerr << "Failed to read deviceName" << std::endl;
        return -1;
    }

    // Read ext, size, constant, global
    if (rpc_read(conn, &ext, sizeof(int)) < 0)
    {
        std::cerr << "Failed reading ext" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &size, sizeof(size_t)) < 0)
    {
        std::cerr << "Failed reading size" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &constant, sizeof(int)) < 0)
    {
        std::cerr << "Failed reading constant" << std::endl;
        return -1;
    }

    if (rpc_read(conn, &global, sizeof(int)) < 0)
    {
        std::cerr << "Failed reading global" << std::endl;
        return -1;
    }

    std::cout << "Received __cudaRegisterVar with deviceName: " << deviceName << std::endl;

    // Call the original __cudaRegisterVar function
    __cudaRegisterVar_type orig;
    orig = (__cudaRegisterVar_type)dlsym(RTLD_NEXT, "__cudaRegisterVar");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaRegisterVar" << std::endl;
        return -1;
    }

    orig(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);

    // End request phase
    int request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Start response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    // No need to send anything back here; just end the response
    if (rpc_end_response(conn, &res) < 0)
        return -1;

    return 0;
}

