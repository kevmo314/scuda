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
    unsigned int op;
    int request_id;

    // Read the request ID from the client
    if (read(conn->connfd, &request_id, sizeof(int)) <= 0) {
        std::cerr << "Failed to read request ID from client." << std::endl;
        return -1;
    }

    // Read the operation code from the client
    if (read(conn->connfd, &op, sizeof(unsigned int)) <= 0) {
        std::cerr << "Failed to read operation from client." << std::endl;
        return -1;
    }

    std::cout << "Handling request ID: " << request_id << ", operation: " << op << std::endl;

    // Simulate some operation handler function
    auto opHandler = get_handler(op);

    if (opHandler == NULL) {
        std::cerr << "Unknown or unsupported operation: " << op << std::endl;
        return -1;
    }

    // Handle the operation
    int result = opHandler((void *)conn);

    // Write the request ID back to the client (important for client to verify the response)
    if (write(conn->connfd, &request_id, sizeof(request_id)) <= 0) {
        std::cerr << "Failed to write request ID back to client." << std::endl;
        return -1;
    }

    // Write the result of the operation back to the client
    if (write(conn->connfd, &result, sizeof(result)) <= 0) {
        std::cerr << "Failed to write result to client." << std::endl;
        return -1;
    }

    std::cout << "Finished writing response for request ID " << request_id << std::endl;

    return 0;
}

// void client_handler(int connfd)
// {
//     std::vector<std::future<void>> futures;
//     conn_t conn = {connfd};
//     if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
//         pthread_mutex_init(&conn.write_mutex, NULL) < 0)
//     {
//         std::cerr << "Error initializing mutex." << std::endl;
//         return;
//     }
//     while (1)
//     {
//         if (pthread_mutex_lock(&conn.read_mutex) < 0)
//         {
//             std::cerr << "Error locking mutex." << std::endl;
//             break;
//         }
//         int n = read(connfd, &conn.read_request_id, sizeof(int));
//         if (n == 0)
//         {
//             printf("client disconnected, loop continuing. \n");
//             break;
//         }
//         else if (n < 0)
//         {
//             printf("error reading from client.\n");
//             break;
//         }

//         // run our request handler in a separate thread
//         auto future = std::async(
//             std::launch::async,
//             [&conn]()
//             {
//                 int result = request_handler(&conn);
//                 if (write(conn.connfd, &result, sizeof(int)) < 0)
//                     std::cerr << "error writing to client." << std::endl;

//                 if (pthread_mutex_unlock(&conn.write_mutex) < 0)
//                     std::cerr << "Error unlocking mutex." << std::endl;
//             });

//         futures.push_back(std::move(future));
//     }

//     for (auto &future : futures)
//         future.wait();

//     if (pthread_mutex_destroy(&conn.read_mutex) < 0 ||
//         pthread_mutex_destroy(&conn.write_mutex) < 0)
//         std::cerr << "Error destroying mutex." << std::endl;
//     close(connfd);
// }

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

void set_non_blocking(int sockfd) {
    int flags = fcntl(sockfd, F_GETFL, 0);
    if (flags < 0) {
        std::cerr << "Failed to get socket flags" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        std::cerr << "Failed to set non-blocking" << std::endl;
        exit(EXIT_FAILURE);
    }
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

    // Bind the socket
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

    if (listen(sockfd, MAX_CLIENTS) != 0) {
        std::cerr << "Listen failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Server listening on port " << port << "..." << std::endl;

    // Set the server socket to non-blocking
    set_non_blocking(sockfd);

    // Initialize fd_set for select()
    fd_set readfds;
    int client_sockets[MAX_CLIENTS] = {0};  // To track active client sockets

    // Server loop
    while (true) {
        FD_ZERO(&readfds);
        FD_SET(sockfd, &readfds);
        int max_sd = sockfd;

        // Add client sockets to fd_set
        for (int i = 0; i < MAX_CLIENTS; i++) {
            int sd = client_sockets[i];
            if (sd > 0) FD_SET(sd, &readfds);
            if (sd > max_sd) max_sd = sd;
        }

        // Use select() to wait for activity on one of the sockets
        int activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);
        if (activity < 0 && errno != EINTR) {
            std::cerr << "Select error" << std::endl;
        }

        // Check if there's a new connection on the server socket
        if (FD_ISSET(sockfd, &readfds)) {
            socklen_t len = sizeof(cli);
            int connfd = accept(sockfd, (struct sockaddr *)&cli, &len);
            if (connfd < 0) {
                std::cerr << "Server accept failed." << std::endl;
                continue;
            }

            // Add new connection to client_sockets
            for (int i = 0; i < MAX_CLIENTS; i++) {
                if (client_sockets[i] == 0) {
                    client_sockets[i] = connfd;
                    std::cout << "New client connected on socket " << connfd << std::endl;
                    break;
                }
            }
        }

        // Handle IO for each client
        for (int i = 0; i < MAX_CLIENTS; i++) {
            int sd = client_sockets[i];
            if (FD_ISSET(sd, &readfds)) {
                conn_t conn = {sd};
                if (request_handler(&conn) < 0) {
                    // Connection is closed
                    close(sd);
                    client_sockets[i] = 0;
                }
            }
        }
    }

    close(sockfd);
    return 0;
}
