#ifndef __MANUAL_SERVER_H__
#define __MANUAL_SERVER_H__

#include "rpc.h"

int handle_cudaFree(conn_t *conn);
int handle_cudaMemcpy(conn_t *conn);
int handle_cudaMemcpyAsync(conn_t *conn);
int handle_cudaLaunchKernel(conn_t *conn);
int handle_cudaMallocManaged(conn_t *conn);
int handle___cudaRegisterVar(conn_t *conn);
int handle___cudaRegisterFunction(conn_t *conn);
int handle___cudaRegisterFatBinary(conn_t *conn);
int handle___cudaRegisterFatBinaryEnd(conn_t *conn);
int handle___cudaPushCallConfiguration(conn_t *conn);
int handle___cudaPopCallConfiguration(conn_t *conn);

#endif