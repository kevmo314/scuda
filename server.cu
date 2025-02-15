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
#include "rpc.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

int request_handler(conn_t *conn) {
  unsigned int op;

  // Attempt to read the operation code from the client
  if (read(conn->connfd, &op, sizeof(unsigned int)) < 0)
    return -1;

  auto opHandler = get_handler(op);

  if (opHandler == NULL) {
    std::cerr << "Unknown or unsupported operation: " << op << std::endl;
    return -1;
  }

  return opHandler(conn);
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
