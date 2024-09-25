#ifndef GEN_CLIENT_H
#define GEN_CLIENT_H

#include <nvml.h>
#include <cuda.h>

void *get_function_pointer(const char *function_name);
#endif