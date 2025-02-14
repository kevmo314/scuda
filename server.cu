#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <nvml.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "codegen/gen_server.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

typedef struct {
  int connfd;

  int read_id;
  int request_id;
  int write_id;
  int write_op;

  pthread_mutex_t read_mutex, write_mutex;
  pthread_cond_t read_cond;
  struct iovec write_iov[128];
  int write_iov_count = 0;
} conn_t;

int request_handler(const conn_t *conn) {
  unsigned int op;

  // Attempt to read the operation code from the client
  if (read(conn->connfd, &op, sizeof(unsigned int)) < 0)
    return -1;

  auto opHandler = get_handler(op);

  if (opHandler == NULL) {
    std::cerr << "Unknown or unsupported operation: " << op << std::endl;
    return -1;
  }

  return opHandler((void *)conn);
}

void client_handler(int connfd) {
  conn_t conn = {connfd};
  if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.write_mutex, NULL) < 0) {
    std::cerr << "Error initializing mutex." << std::endl;
    return;
  }

#ifdef VERBOSE
  printf("Client connected.\n");
#endif

  if (pthread_mutex_lock(&conn.read_mutex) < 0) {
    std::cerr << "Error locking mutex." << std::endl;
  }
  while (1) {
    while (conn.read_id != 0)
      pthread_cond_wait(&conn.read_cond, &conn.read_mutex);

    int n = read(connfd, &conn.read_id, sizeof(int));
    if (n == 0) {
      printf("client disconnected, loop continuing. \n");
      break;
    } else if (n < 0) {
      printf("error reading from client.\n");
      break;
    }

    if (conn.read_id < 0) {
      // this is a response to an existing request, notify everyone else
      // and spin again.
      if (pthread_cond_broadcast(&conn.read_cond) < 0) {
        std::cerr << "Error broadcasting condition or unlocking mutex."
                  << std::endl;
        break;
      }
      continue;
    } else {
      // TODO: this can't be multithreaded as some of the __cuda* functions
      // assume that they are running in the same thread as the one that
      // calls cudaLaunchKernel. we'll need to find a better way to map
      // function calls to threads. maybe each rpc maps to an optional
      // thread id that is passed to the handler?
      if (request_handler(&conn) < 0)
        std::cerr << "Error handling request." << std::endl;
    }
  }

  if (pthread_mutex_destroy(&conn.read_mutex) < 0 ||
      pthread_mutex_destroy(&conn.write_mutex) < 0)
    std::cerr << "Error destroying mutex." << std::endl;
  close(connfd);
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on entry.
int rpc_read_start(const void *conn, int write_id) {
  if (pthread_mutex_lock(&((conn_t *)conn)->read_mutex) < 0)
    return -1;

  // wait for the active read id to be the request id we are waiting for
  while (((conn_t *)conn)->read_id != write_id)
    if (pthread_cond_wait(&((conn_t *)conn)->read_cond,
                          &((conn_t *)conn)->read_mutex) < 0)
      return -1;

  return 0;
}

int rpc_read(const void *conn, void *data, size_t size) {
  return recv(((conn_t *)conn)->connfd, data, size, MSG_WAITALL);
}

// rpc_read_end releases the response lock on the given connection.
int rpc_read_end(const void *conn) {
  int read_id = ((conn_t *)conn)->read_id;
  ((conn_t *)conn)->read_id = 0;
  if (pthread_cond_broadcast(&((conn_t *)conn)->read_cond) < 0 ||
      pthread_mutex_unlock(&((conn_t *)conn)->read_mutex) < 0)
    return -1;
  return read_id;
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(const void *conn, const unsigned int op) {
  if (pthread_mutex_lock(&((conn_t *)conn)->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  ((conn_t *)conn)->write_iov_count = 2; // skip 2 for the header
  ((conn_t *)conn)->write_id = ++(((conn_t *)conn)->request_id);
  ((conn_t *)conn)->write_op = op;
  return 0;
}
// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_response(const void *conn, const int read_id) {
  if (pthread_mutex_lock(&((conn_t *)conn)->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  ((conn_t *)conn)->write_iov_count = 1; // skip 1 for the header
  ((conn_t *)conn)->write_id = read_id;
  ((conn_t *)conn)->write_op = 0;
  return 0;
}

int rpc_write(const void *conn, const void *data, const size_t size) {
  ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] =
      (struct iovec){(void *)data, size};
  return 0;
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(const void *conn) {
  ((conn_t *)conn)->write_iov[0] = {&((conn_t *)conn)->write_id, sizeof(int)};
  if (((conn_t *)conn)->write_op != -1) {
    ((conn_t *)conn)->write_iov[1] = {&((conn_t *)conn)->write_op,
                                      sizeof(unsigned int)};
  }

  // write the request to the server
  if (writev(((conn_t *)conn)->connfd, ((conn_t *)conn)->write_iov,
             ((conn_t *)conn)->write_iov_count) < 0 ||
      pthread_mutex_unlock(&((conn_t *)conn)->write_mutex) < 0)
    return -1;
  return ((conn_t *)conn)->write_id;
}

int main() {
  int port = DEFAULT_PORT;
  struct sockaddr_in servaddr, cli;
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    printf("Socket creation failed.\n");
    exit(EXIT_FAILURE);
  }

  char *p = getenv("SCUDA_PORT");

  if (p == NULL) {
    port = DEFAULT_PORT;
  } else {
    port = atoi(p);
  }

  // Bind the socket
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(port);

  const int enable = 1;
  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
    printf("Socket bind failed.\n");
    exit(EXIT_FAILURE);
  }

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
    printf("Socket bind failed.\n");
    exit(EXIT_FAILURE);
  }

  if (listen(sockfd, MAX_CLIENTS) != 0) {
    printf("Listen failed.\n");
    exit(EXIT_FAILURE);
  }

  printf("Server listening on port %d...\n", port);

  // Server loop
  while (1) {
    socklen_t len = sizeof(cli);
    int connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

    if (connfd < 0) {
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
