#ifndef MANUAL_SERVER_H
#define MANUAL_SERVER_H

int handle_cudaMemcpy(void *conn);
int handle_cudaMemcpyAsync(void *conn);
int handle_cudaLaunchKernel(void *conn);
int handle___cudaRegisterVar(void *conn);
int handle___cudaRegisterFunction(void *conn);
int handle___cudaRegisterFatBinary(void *conn);
int handle___cudaRegisterFatBinaryEnd(void *conn);
int handle___cudaPushCallConfiguration(void *conn);
int handle___cudaPopCallConfiguration(void *conn);
int handle_cudaHostRegister(void *conn);
int handle_cudaHostUnregister(void *conn);

#endif