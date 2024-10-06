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

#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <fcntl.h>

#include <sys/epoll.h>  // Add this line for epoll support
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstring>

#include "codegen/gen_server.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

typedef struct
{
    int connfd;
    int read_request_id;
    pthread_mutex_t read_mutex, write_mutex;
} conn_t;

int request_handler(conn_t *conn) {
    unsigned int op = 0;
    int request_id = -1;
    ssize_t bytesRead = 0;

    std::cout << "Before ID " << request_id << std::endl;

    // Read the request ID from the client
    bytesRead = 0;
    while (bytesRead < sizeof(request_id)) {
        ssize_t n = read(conn->connfd, ((char*)&request_id) + bytesRead, sizeof(request_id) - bytesRead);
        if (n < 0) {
            std::cerr << "Error reading request ID from client: " << strerror(errno) << std::endl;
            return -1;
        }
        if (n == 0) {
            std::cerr << "Client disconnected while reading request ID." << std::endl;
            return -1;
        }
        bytesRead += n;
    }

    // Ensure the request ID is valid
    if (request_id <= 0) {
        std::cerr << "Invalid request ID received: " << request_id << std::endl;
        return -1;
    }

    // Set the request ID in the conn_t struct so it can be used later in rpc_end_request
    conn->read_request_id = request_id;

    // Read the operation code from the client
    bytesRead = 0;
    while (bytesRead < sizeof(op)) {
        ssize_t n = read(conn->connfd, ((char*)&op) + bytesRead, sizeof(op) - bytesRead);
        if (n < 0) {
            std::cerr << "Error reading operation from client: " << strerror(errno) << std::endl;
            return -1;
        }
        if (n == 0) {
            std::cerr << "Client disconnected while reading operation." << std::endl;
            return -1;
        }
        bytesRead += n;
    }

    std::cout << "Handling request ID: " << request_id << ", operation: " << op << std::endl;

    // Validate the operation code before proceeding
    auto opHandler = get_handler(op);
    if (opHandler == NULL) {
        std::cerr << "Unknown or unsupported operation: " << op << std::endl;
        return -1;
    }

    // Process the operation
    int result = opHandler((void *)conn);

    // Write the result back to the client
    if (write(conn->connfd, &result, sizeof(result)) <= 0) {
        std::cerr << "Failed to write result to client: " << strerror(errno) << std::endl;
        return -1;
    }

    if (pthread_mutex_unlock(&conn->write_mutex) < 0) {
        std::cerr << "Error unlocking mutex." << std::endl;
    }

    std::cout << "Finished writing response for request ID: " << request_id << std::endl;

    return 0;
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

    // Log the request_id to ensure it's properly initialized
    std::cout << "[Server] Ending request for request ID: " << request_id << std::endl;

    // Ensure mutex is unlocked correctly
    if (pthread_mutex_unlock(&((conn_t *)conn)->read_mutex) < 0) {
        std::cerr << "[Server] Failed to unlock read mutex." << std::endl;
        return -1;
    }

    return request_id;
}

int rpc_start_response(const void *conn, const int request_id)
{
    std::cout << "[Server] Sending response for request ID: " << request_id << std::endl;
    pthread_mutex_lock(&((conn_t *)conn)->write_mutex);

    if (write(((conn_t *)conn)->connfd, &request_id, sizeof(int)) < 0)
    {
        std::cerr << "[Server] Failed to write request ID back to client." << std::endl;
        pthread_mutex_unlock(&((conn_t *)conn)->write_mutex);
        return -1;
    }

    pthread_mutex_unlock(&((conn_t *)conn)->write_mutex);
    return 0;
}

#define MAX_EVENTS 100

int set_non_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
}

int main() {
    int port = DEFAULT_PORT;
    struct sockaddr_in servaddr, cli;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        std::cerr << "Socket creation failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    char *p = getenv("SCUDA_PORT");
    if (p != NULL) {
        port = atoi(p);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(port);

    const int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
        std::cerr << "Socket bind failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
        std::cerr << "Socket bind failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (listen(sockfd, SOMAXCONN) != 0) {
        std::cerr << "Listen failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Server listening on port " << port << "..." << std::endl;

    // Set the server socket to non-blocking
    set_non_blocking(sockfd);

    // Initialize epoll
    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1) {
        std::cerr << "Epoll creation failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Add the server socket to the epoll instance
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1) {
        std::cerr << "Failed to add socket to epoll." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<conn_t> clients;

    // Event loop
    while (true) {
        struct epoll_event events[MAX_EVENTS];
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (nfds == -1) {
            std::cerr << "Epoll wait error." << std::endl;
            break;
        }

        for (int i = 0; i < nfds; ++i) {
            if (events[i].data.fd == sockfd) {
                // Accept new connections
                socklen_t len = sizeof(cli);
                int connfd = accept(sockfd, (struct sockaddr *)&cli, &len);
                if (connfd == -1) {
                    std::cerr << "Server accept failed." << std::endl;
                    continue;
                }

                set_non_blocking(connfd);

                // Add the new connection to epoll
                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = connfd;
                if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, connfd, &ev) == -1) {
                    std::cerr << "Failed to add client to epoll." << std::endl;
                    close(connfd);
                    continue;
                }

                // Track the client
                conn_t client;
                client.connfd = connfd;
                pthread_mutex_init(&client.read_mutex, NULL);
                pthread_mutex_init(&client.write_mutex, NULL);
                clients.push_back(client);

                std::cout << "New client connected on socket " << connfd << std::endl;
            } else {
                // Handle client I/O
                int client_fd = events[i].data.fd;
                auto it = std::find_if(clients.begin(), clients.end(), [client_fd](const conn_t &c) { return c.connfd == client_fd; });
                if (it != clients.end()) {
                    conn_t *conn = &(*it);
                    if (request_handler(conn) < 0) {
                        std::cerr << "Client disconnected from socket " << client_fd << std::endl;

                        // Close the client socket
                        close(client_fd);

                        // Remove from epoll
                        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client_fd, NULL);

                        // Destroy mutexes
                        pthread_mutex_destroy(&conn->read_mutex);
                        pthread_mutex_destroy(&conn->write_mutex);

                        // Remove client from vector
                        clients.erase(it);
                    }
                }
            }
        }
    }

    close(sockfd);
    return 0;
}
