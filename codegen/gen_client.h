#ifndef GEN_CLIENT_H
#define GEN_CLIENT_H

#include <nvml.h>
#include <cuda.h>

void *get_function_pointer(const char *function_name);

CUresult cuGetProcAddressHandler(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
CUresult cuGetProcAddress_v2_handler(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);

#endif