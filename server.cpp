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
#include <cuda_runtime_api.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include <dlfcn.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <vector>

#include <map>

#include <csignal>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>

#include "codegen/gen_server.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

typedef struct {
  int connfd;
  int read_request_id;
  int write_request_id;
  pthread_mutex_t read_mutex, write_mutex;
  struct iovec write_iov[128];
  int write_iov_count = 0;
} conn_t;

std::map<conn_t*, std::map<void*, size_t>> managed_ptrs;
std::map<conn_t*, void*> host_funcs;

static jmp_buf catch_segfault;
static void *faulting_address = nullptr;

int rpc_write(const void *conn, const void *data, const size_t size) {
  ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] =
      (struct iovec){(void *)data, size};
  return 0;
}

static void segfault(int sig, siginfo_t *info, void *unused) {
  faulting_address = info->si_addr;

  int found = -1;
  void* ptr;
  size_t size;

  std::cout << "segfault!!" << faulting_address << std::endl;

  for (const auto& conn_entry : host_funcs) {
    if (conn_entry.second == faulting_address) {
      std::cout << "Faulting address matches stored pointer in host_funcs!" << std::endl;
      uintptr_t aligned = (uintptr_t)faulting_address;

      uintptr_t page_size = sysconf(_SC_PAGESIZE);
      uintptr_t aligned_address = (uintptr_t)faulting_address & ~(page_size - 1);

      // Ensure that mmap covers the full page
      void *allocated = mmap((void *)aligned_address, page_size,
                              PROT_READ | PROT_WRITE | PROT_EXEC,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);

      if (allocated == MAP_FAILED) {
          perror("Failed to allocate memory at faulting address");
          _exit(1);
      }

      printf("The address of x is: %p\n", (void*)allocated);

      found = 1;

      break;
    }
  }
  
  if (found == 1) {
    return;
  };

  for (const auto& conn_entry : managed_ptrs) {
    for (const auto& mem_entry : conn_entry.second) {
      size_t allocated_size = mem_entry.second;

      // Check if faulting address is inside this allocated region
      if ((uintptr_t)mem_entry.first <= (uintptr_t)faulting_address &&
      (uintptr_t)faulting_address < ((uintptr_t)mem_entry.first + allocated_size)) {
          found = 1;
          size = allocated_size;

          // Align memory allocation to the closest possible address
          uintptr_t aligned = (uintptr_t)faulting_address & ~(allocated_size - 1);

          // Allocate memory at the faulting address
          void *allocated =
            mmap((void *)aligned, allocated_size + (uintptr_t)faulting_address - aligned, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

          if (allocated == MAP_FAILED) {
              perror("Failed to allocate memory at faulting address");
              _exit(1);
          }

          printf("The address of x is: %p\n", (void*)allocated);

          // if (rpc_write(conn_entry.first, (void*)&allocated, sizeof(void*)) < 0) {
          //   std::cout << "failed to write memory: " << &faulting_address << std::endl;
          // }

          // printf("wrote data...\n");

          break;
      }
    }
  }

  if (found == 1) {
    printf("FOUND!!\n");
    return;
  };

  // raise our original segfault handler
  struct sigaction sa;
  sa.sa_handler = SIG_DFL;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction(SIGSEGV, &sa, nullptr) == -1) {
    perror("Failed to reset SIGSEGV handler");
    _exit(EXIT_FAILURE);
  }

  raise(SIGSEGV);
}

void append_host_func_ptr(const void *conn, cudaHostNodeParams params) {
  conn_t *connfd = (conn_t*)conn;

  host_funcs[connfd] = (void*)params.fn;
}

void append_managed_ptr(const void *conn, cudaPitchedPtr ptr) {
  conn_t *connfd = (conn_t*)conn;

  // Ensure the inner map exists before inserting the cudaPitchedPtr
  managed_ptrs[connfd][ptr.ptr] = ptr.pitch;
}

static void set_segfault_handlers() {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = segfault;

  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    perror("sigaction");
    exit(EXIT_FAILURE);
  }

  std::cout << "Segfault handler installed." << std::endl;
}

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

  while (1) {
    if (pthread_mutex_lock(&conn.read_mutex) < 0) {
      std::cerr << "Error locking mutex." << std::endl;
      break;
    }
    int n = read(connfd, &conn.read_request_id, sizeof(int));
    if (n == 0) {
      printf("client disconnected, loop continuing. \n");
      break;
    } else if (n < 0) {
      printf("error reading from client.\n");
      break;
    }

    // TODO: this can't be multithreaded as some of the __cuda* functions
    // assume that they are running in the same thread as the one that
    // calls cudaLaunchKernel. we'll need to find a better way to map
    // function calls to threads. maybe each rpc maps to an optional
    // thread id that is passed to the handler?
    if (request_handler(&conn) < 0)
      std::cerr << "Error handling request." << std::endl;
  }

  if (pthread_mutex_destroy(&conn.read_mutex) < 0 ||
      pthread_mutex_destroy(&conn.write_mutex) < 0)
    std::cerr << "Error destroying mutex." << std::endl;
  close(connfd);
}

int rpc_read(const void *conn, void *data, size_t size) {
  return recv(((conn_t *)conn)->connfd, data, size, MSG_WAITALL);
}

// signal from the handler that the request read is complete.
int rpc_end_request(const void *conn) {
  int request_id = ((conn_t *)conn)->read_request_id;
  if (pthread_mutex_unlock(&((conn_t *)conn)->read_mutex) < 0)
    return -1;
  return request_id;
}

int rpc_start_response(const void *conn, const int request_id) {
  if (pthread_mutex_lock(&((conn_t *)conn)->write_mutex) < 0)
    return -1;
  ((conn_t *)conn)->write_request_id = request_id;
  ((conn_t *)conn)->write_iov_count = 1;
  return 0;
}

int rpc_end_response(const void *conn, void *result) {
  ((conn_t *)conn)->write_iov[0] =
      (struct iovec){&((conn_t *)conn)->write_request_id, sizeof(int)};
  ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] =
      (struct iovec){result, sizeof(int)};
  if (writev(((conn_t *)conn)->connfd, ((conn_t *)conn)->write_iov,
             ((conn_t *)conn)->write_iov_count) < 0 ||
      pthread_mutex_unlock(&((conn_t *)conn)->write_mutex) < 0)
    return -1;
  return 0;
}

int main() {
  int port = DEFAULT_PORT;
  struct sockaddr_in servaddr, cli;
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    printf("Socket creation failed.\n");
    exit(EXIT_FAILURE);
  }

  set_segfault_handlers();

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
