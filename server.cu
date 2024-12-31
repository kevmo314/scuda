#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <nvml.h>
#include <future>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <sys/uio.h>

#include "codegen/gen_server.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

typedef struct
{
    int connfd;
    int read_request_id;
    int write_request_id;
    struct iovec write_iov[128];
    int write_iov_count = 0;
} conn_t;

int request_handler(const conn_t *conn)
{
    unsigned int op;

    // Attempt to read the operation code from the client
    if (read(conn->connfd, &op, sizeof(unsigned int)) < 0)
        return -1;

    auto opHandler = get_handler(op);

    if (opHandler == NULL)
    {
        std::cerr << "Unknown or unsupported operation: " << op << std::endl;
        return -1;
    }

    return opHandler((void *)conn);
}

void client_handler(int connfd)
{
    conn_t conn = {connfd};

#ifdef VERBOSE
    printf("Client connected.\n");
#endif

    while (1)
    {
        int n = read(connfd, &conn.read_request_id, sizeof(int));
        if (n == 0)
        {
            printf("client disconnected, loop continuing. \n");
            break;
        }
        else if (n < 0)
        {
            printf("error reading from client.\n");
            break;
        }

        if (request_handler(&conn) < 0)
            std::cerr << "Error handling request." << std::endl;
    }

    close(connfd);
}

int rpc_read(const void *conn, void *data, size_t size)
{
    return recv(((conn_t *)conn)->connfd, data, size, MSG_WAITALL);
}

int rpc_write(const void *conn, const void *data, const size_t size)
{
    ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] = (struct iovec){(void *)data, size};
    return 0;
}

// signal from the handler that the request read is complete.
int rpc_end_request(const void *conn)
{
    int request_id = ((conn_t *)conn)->read_request_id;
    return request_id;
}

int rpc_start_response(const void *conn, const int request_id)
{
    ((conn_t *)conn)->write_request_id = request_id;
    ((conn_t *)conn)->write_iov_count = 1;
    return 0;
}

int rpc_end_response(const void *conn, void *result)
{
    ((conn_t *)conn)->write_iov[0] = (struct iovec){&((conn_t *)conn)->write_request_id, sizeof(int)};
    ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] = (struct iovec){result, sizeof(int)};
    if (writev(((conn_t *)conn)->connfd, ((conn_t *)conn)->write_iov, ((conn_t *)conn)->write_iov_count) < 0)
        return -1;
    return 0;
}

int main()
{
    int port = DEFAULT_PORT;
    struct sockaddr_in servaddr, cli;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("Socket creation failed.\n");
        exit(EXIT_FAILURE);
    }

    char *p = getenv("SCUDA_PORT");

    if (p == NULL)
    {
        port = DEFAULT_PORT;
    }
    else
    {
        port = atoi(p);
    }

    // Bind the socket
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(port);

    const int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
    {
        printf("Socket bind failed.\n");
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0)
    {
        printf("Socket bind failed.\n");
        exit(EXIT_FAILURE);
    }

    if (listen(sockfd, MAX_CLIENTS) != 0)
    {
        printf("Listen failed.\n");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d...\n", port);

    // Server loop
    while (1)
    {
        socklen_t len = sizeof(cli);
        int connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

        if (connfd < 0)
        {
            std::cerr << "Server accept failed." << std::endl;
            continue;
        }

        pid_t pid = fork();
        if (pid < 0)
        {
            std::cerr << "Fork failed." << std::endl;
            close(connfd);
            continue;
        }
        else if (pid == 0)
        {
            close(sockfd);
            client_handler(connfd);
            exit(0);
        }
        else
        {
            close(connfd);
        }
    }

    close(sockfd);
    return 0;
}
