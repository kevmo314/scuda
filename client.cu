#define _GNU_SOURCE

#include <arpa/inet.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <nvml.h>
#include <unistd.h>
#include <pthread.h>

int sockfd;

int open_rpc_client()
{
    int connfd;
    struct sockaddr_in servaddr, cli;

    if (sockfd != 0)
    {
        return sockfd;
    }

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("socket creation failed...\n");
        exit(0);
    }

    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    servaddr.sin_port = htons(14833);
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0)
    {
        printf("connection with the server failed...\n");
        exit(0);
    }
    return sockfd;
}

int nextRequestId = 0, responseId = -1;

int send_rpc_message(const char *op, const void *args, const int argslen)
{
    char oplen = (char)strlen(op);
    if (write(sockfd, (void *)&nextRequestId, sizeof(int)) < 0)
        return -1;
    if (write(sockfd, (void *)&oplen, sizeof(char)) < 0)
        return -1;
    if (write(sockfd, op, oplen) < 0)
        return -1;
    if (write(sockfd, (void *)&argslen, sizeof(int)) < 0)
        return -1;
    if (write(sockfd, args, argslen) < 0)
        return -1;
}

void close_rpc_client()
{
    close(sockfd);
    sockfd = 0;
}

nvmlReturn_t nvmlInitWithFlags(unsigned int flags)
{
    open_rpc_client();
    char buffer[1024];
    sprintf(buffer, "nvmlInitWithFlags %d\n", flags);
    write(sockfd, buffer, strlen(buffer));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    // write "HI MUGIT" to the name buffer
    strcpy(name, "HI MUGIT");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    // write "HI MUGIT" to the name buffer
    strcpy(name, "HI MUGIT");
    return NVML_SUCCESS;
}

void *dlsym(void *handle, const char *name)
{
    printf("Resolving symbol: %s\n", name);

    if (!strcmp(name, "nvmlDeviceGetName"))
        return (void *)nvmlDeviceGetName;

    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL)
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    /* my target binary is even asking for dlsym() via dlsym()... */
    if (!strcmp(name, "dlsym"))
        return (void *)dlsym;
    return real_dlsym(handle, name);
}
