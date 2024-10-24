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

        std::cout << "calling to device ... cudaMemcpy: " << dst << " " << src << " " << count << " " << kind << std::endl;
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
    std::cout << "incoming..." << std::endl;
    // Read magic number to distinguish between formats
    int magic;
    int version;
    char* fatbinData;
    unsigned long long int blen;
    unsigned int hmagic;
    unsigned int hversion;
    __cudaFatCudaBinary2EntryRec* entry;

    if (rpc_read(conn, &magic, sizeof(int)) < 0)
    {
        std::cerr << "Failed to read magic from client" << std::endl;
        return -1;
    }

    // if (magic == __cudaFatMAGIC)
    // {
    //     // Handle the __cudaFatMAGIC format
    //     std::cout << "Received __cudaFatMAGIC format" << std::endl;

    //     unsigned long version;
    //     if (rpc_read(conn, &version, sizeof(unsigned long)) < 0) 
    //     {
    //         std::cerr << "Failed to read version for __cudaFatMAGIC" << std::endl;
    //         return -1;
    //     }

    //     // Read the PTX information
    //     while (true)
    //     {
    //         int gpuProfileNameLength;
    //         if (rpc_read(conn, &gpuProfileNameLength, sizeof(int)) < 0) return -1;

    //         if (gpuProfileNameLength == 0) break; // End of entries

    //         char* gpuProfileName = (char*)malloc(gpuProfileNameLength + 1);
    //         if (rpc_read(conn, gpuProfileName, gpuProfileNameLength) < 0) return -1;
    //         gpuProfileName[gpuProfileNameLength] = '\0';  // Null-terminate

    //         int ptxLength;
    //         if (rpc_read(conn, &ptxLength, sizeof(int)) < 0) return -1;

    //         char* ptx = (char*)malloc(ptxLength + 1);
    //         if (rpc_read(conn, ptx, ptxLength) < 0) return -1;
    //         ptx[ptxLength] = '\0';  // Null-terminate

    //         // Process the GPU profile name and PTX code
    //         std::cout << "GPU Profile: " << gpuProfileName << ", PTX: " << ptx << std::endl;

    //         free(gpuProfileName);
    //         free(ptx);
    //     }
    // }
    // else
    
     if (magic == __cudaFatMAGIC2)
    {
        // Handle the __cudaFatMAGIC2 format
        std::cout << "Received __cudaFatMAGIC2 format" << std::endl;

        // Read version and length of the fat binary
        if (rpc_read(conn, &version, sizeof(int)) < 0) 
        {
            std::cerr << "Failed to read version for __cudaFatMAGIC2" << std::endl;
            return -1;
        }

        if (rpc_read(conn, &hmagic, sizeof(unsigned int)) < 0) 
        {
            std::cerr << "unsigned magic read" << std::endl;
            return -1;
        }

        std::cout << "hmagic: " << hmagic << std::endl;

        if (rpc_read(conn, &hversion, sizeof(unsigned int)) < 0) 
        {
            std::cerr << "Failed to read length for __cudaFatMAGIC2" << std::endl;
            return -1;
        }

        std::cout << "hversion: " << hversion << std::endl;

        if (rpc_read(conn, &blen, sizeof(unsigned long long int)) < 0) 
        {
            std::cerr << "Failed to read length for __cudaFatMAGIC2" << std::endl;
            return -1;
        }

        std::cout << "blen: " << blen << std::endl;

        // Allocate memory to store the fat binary data
        fatbinData = (char*)malloc(blen);
        if (!fatbinData)
        {
            std::cerr << "Failed to allocate memory for fatbin data" << std::endl;
            return -1;
        }

        // Now process the fat binary entries
        char* base = fatbinData + sizeof(__cudaFatCudaBinary2Header);
        uint64_t offset = 0;
        entry = (__cudaFatCudaBinary2EntryRec*)(base);

        while (offset < blen)
        {
            std::cout << "Processing entry" << std::endl;

            if (rpc_read(conn, &entry->type, sizeof(unsigned int)) < 0) return -1;
            std::cout << "read type " << entry->type << std::endl;
            if (rpc_read(conn, &entry->binary, sizeof(unsigned int)) < 0) return -1;
            std::cout << "read binary " << entry->binary << std::endl;
            if (rpc_read(conn, &entry->binarySize, sizeof(unsigned long long int)) < 0) return -1;

            std::cout << "read binary size " << entry->binarySize << std::endl;

            // if (rpc_read(conn, &entry->nameSize, sizeof(unsigned int)) < 0) return -1;

            if (entry->nameSize > 0) 
            {
                std::cout << "Processing name size" << std::endl;
                char* name = (char*)malloc(entry->nameSize + 1);
                if (rpc_read(conn, name, entry->nameSize) < 0) 
                {
                    std::cerr << "Failed to read entry name" << std::endl;
                    free(name);
                    free(fatbinData);
                    return -1;
                }
                name[entry->nameSize] = '\0';
                std::cout << "Entry name: " << name << std::endl;
                free(name);
            }

            // // Process the actual binary data here...
            // if (entry->binarySize > 0) 
            // {
            //     char* binaryData = (char*)malloc(entry->binarySize);
            //     if (!binaryData) 
            //     {
            //         std::cerr << "Failed to allocate memory for binary data" << std::endl;
            //         free(fatbinData);
            //         return -1;
            //     }

            //     // Read the binary data
            //     if (rpc_read(conn, binaryData, entry->binarySize) < 0) 
            //     {
            //         std::cerr << "Failed to read binary data" << std::endl;
            //         free(binaryData);
            //         free(fatbinData);
            //         return -1;
            //     }

            //     std::cout << "Binary data processed, size: " << entry->binarySize << " bytes" << std::endl;

            //     free(binaryData);
            // }

            // Move to the next entry
            offset += entry->binary + entry->binarySize;
            entry = (__cudaFatCudaBinary2EntryRec*)(base + offset);
        }

        free(fatbinData);
    }
    else
    {
        std::cerr << "Unknown magic number: " << magic << std::endl;
        return -1;
    }

     // Rebuild the fat binary structure to pass it to __cudaRegisterFatBinary
    __cudaFatCudaBinary2* rebuiltBinary = (__cudaFatCudaBinary2*)malloc(sizeof(__cudaFatCudaBinary2));
    rebuiltBinary->magic = hmagic;
    rebuiltBinary->version = hversion;
    rebuiltBinary->fatbinData = (unsigned long long*)entry;
    rebuiltBinary->f = nullptr;

    // Find the original __cudaRegisterFatBinary function using dlsym
    __cudaRegisterFatBinary_type orig;
    orig = (__cudaRegisterFatBinary_type)dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
    if (!orig)
    {
        std::cerr << "Failed to find original __cudaRegisterFatBinary" << std::endl;
        free(fatbinData);
        return -1;
    }

    request_id = rpc_end_request(conn);
    if (request_id < 0)
    {
        std::cerr << "rpc_end_request failed" << std::endl;
        return -1;
    }

    // Call the original __cudaRegisterFatBinary with the rebuilt binary
    void** ret = orig((void**)rebuiltBinary);

    std::cout << "Original __cudaRegisterFatBinary returned: " << ret << std::endl;

    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    if (rpc_write(conn, ret, sizeof(void **)) < 0)
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

    std::cout << "hostFunhostFunhostFunhostFunhostFun thread_limit" << thread_limit << std::endl;

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

    std::cout << "received cudaRegisterFatBinaryEnd" << std::endl;

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

    // Call the original function to populate the data
    unsigned result = orig(gridDim, blockDim, sharedMem, stream);

    // Start the response phase
    if (rpc_start_response(conn, request_id) < 0)
    {
        std::cerr << "rpc_start_response failed" << std::endl;
        return -1;
    }

    // Send the updated data back to the client
    if (rpc_write(conn, &gridDim, sizeof(dim3)) < 0)
    {
        std::cerr << "Failed to send gridDim to client" << std::endl;
        return -1;
    }

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

