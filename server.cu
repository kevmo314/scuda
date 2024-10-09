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
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <pthread.h>

#include "codegen/gen_server.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

typedef struct
{
    int connfd;
    int read_request_id;
    pthread_mutex_t read_mutex, write_mutex;
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
    std::vector<std::future<void>> futures;
    conn_t conn = {connfd};
    if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
        pthread_mutex_init(&conn.write_mutex, NULL) < 0)
    {
        std::cerr << "Error initializing mutex." << std::endl;
        return;
    }
    while (1)
    {
        if (pthread_mutex_lock(&conn.read_mutex) < 0)
        {
            std::cerr << "Error locking mutex." << std::endl;
            break;
        }
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

        // run our request handler in a separate thread
        auto future = std::async(
            std::launch::async,
            [&conn]()
            {
                int result = request_handler(&conn);
                if (write(conn.connfd, &result, sizeof(int)) < 0)
                    std::cerr << "error writing to client." << std::endl;

                if (pthread_mutex_unlock(&conn.write_mutex) < 0)
                    std::cerr << "Error unlocking mutex." << std::endl;
            });

        futures.push_back(std::move(future));
    }

    for (auto &future : futures)
        future.wait();

    if (pthread_mutex_destroy(&conn.read_mutex) < 0 ||
        pthread_mutex_destroy(&conn.write_mutex) < 0)
        std::cerr << "Error destroying mutex." << std::endl;
    close(connfd);
}

int rpc_read(const void *conn, void *data, const size_t size)
{
    return read(((conn_t *)conn)->connfd, data, size);
}

int rpc_write(const void *conn, const void *data, const size_t size)
{
    return write(((conn_t *)conn)->connfd, data, size);
}

// signal from the handler that the request read is complete.
int rpc_end_request(const void *conn)
{
    int request_id = ((conn_t *)conn)->read_request_id;
    if (pthread_mutex_unlock(&((conn_t *)conn)->read_mutex) < 0)
        return -1;
    return request_id;
}

int rpc_start_response(const void *conn, const int request_id)
{
    return pthread_mutex_lock(&((conn_t *)conn)->write_mutex) || write(((conn_t *)conn)->connfd, &request_id, sizeof(unsigned int));
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

        std::thread client_thread(client_handler, connfd);

        // detach the thread so it runs independently
        client_thread.detach();
    }

    close(sockfd);
    return 0;
}
