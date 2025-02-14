#include <arpa/inet.h>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <netdb.h>
#include <netinet/tcp.h>
#include <nvml.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <unordered_map>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>

#include "codegen/gen_client.h"

typedef struct {
  int connfd;

  int read_id;
  int request_id;
  int write_id;
  int write_op;

  pthread_t read_thread;
  pthread_mutex_t read_mutex, write_mutex;
  pthread_cond_t read_cond;
  struct iovec write_iov[128];
  int write_iov_count = 0;

  std::unordered_map<void *, size_t> unified_devices;
} conn_t;

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;

const char *DEFAULT_PORT = "14833";

static int init = 0;
static jmp_buf catch_segfault;
static void *faulting_address = nullptr;

static void segfault(int sig, siginfo_t *info, void *unused) {
  faulting_address = info->si_addr;

  for (const auto &[ptr, sz] : conns[0].unified_devices) {
    if ((uintptr_t)ptr <= (uintptr_t)faulting_address &&
        (uintptr_t)faulting_address < ((uintptr_t)ptr + sz)) {
      // ensure we assign memory as close to the faulting address as possible...
      // by masking via the allocated unified memory size.
      uintptr_t aligned = (uintptr_t)faulting_address & ~(sz - 1);

      // Allocate memory at the faulting address
      void *allocated =
          mmap((void *)aligned, sz + (uintptr_t)faulting_address - aligned,
               PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
      if (allocated == MAP_FAILED) {
        perror("Failed to allocate memory at faulting address");
        _exit(1);
      }

      return;
    }
  }

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

int is_unified_pointer(const int index, void *arg) {
  auto &unified_devices = conns[index].unified_devices;
  auto found = unified_devices.find(arg);
  if (found != unified_devices.end())
    return 1;

  return 0;
}

int maybe_copy_unified_arg(const int index, void *arg,
                           enum cudaMemcpyKind kind) {
  auto &unified_devices = conns[index].unified_devices;
  auto found = unified_devices.find(arg);
  if (found != unified_devices.end()) {
    std::cout << "found unified arg pointer; copying..." << std::endl;

    void *ptr = found->first;
    size_t size = found->second;

    cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);

    if (res != cudaSuccess) {
      std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(res)
                << std::endl;

      return -1;
    } else {
      std::cout << "Successfully copied " << size << " bytes" << std::endl;
    }
  }

  return 0;
}

static void set_segfault_handlers() {
  if (init > 0) {
    return;
  }

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = segfault;

  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    perror("sigaction");
    exit(EXIT_FAILURE);
  }

  std::cout << "Segfault handler installed." << std::endl;

  init = 1;
}

int rpc_open() {
  set_segfault_handlers();

  sigsetjmp(catch_segfault, 1);

  if (pthread_mutex_lock(&conn_mutex) < 0)
    return -1;

  if (nconns > 0) {
    if (pthread_mutex_unlock(&conn_mutex) < 0)
      return -1;
    return 0;
  }

  char *server_ips = getenv("SCUDA_SERVER");
  if (server_ips == NULL) {
    printf("SCUDA_SERVER environment variable not set\n");
    std::exit(1);
  }

  char *server_ip = strdup(server_ips);
  char *token;
  while ((token = strsep(&server_ip, ","))) {
    char *host;
    char *port;

    // Split the string into IP address and port
    char *colon = strchr(token, ':');
    if (colon == NULL) {
      host = token;
      port = const_cast<char *>(DEFAULT_PORT);
    } else {
      *colon = '\0';
      host = token;
      port = colon + 1;
    }

    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
      std::cout << "getaddrinfo of " << host << " port " << port << " failed"
                << std::endl;
      continue;
    }

    int flag = 1;
    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1) {
      printf("socket creation failed...\n");
      exit(1);
    }

    int opts = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag,
                          sizeof(int));
    if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0) {
      std::cerr << "Connecting to " << host << " port " << port
                << " failed: " << strerror(errno) << std::endl;
      exit(1);
    }

    conns[nconns++] = {sockfd,
                       0,
                       0,
                       0,
                       0,
                       0,
                       PTHREAD_MUTEX_INITIALIZER,
                       PTHREAD_MUTEX_INITIALIZER,
                       PTHREAD_COND_INITIALIZER};

    pthread_create(&conns[nconns].read_thread, NULL, rpc_read_thread,
                   (void *)&conns[nconns]);
  }

  if (pthread_mutex_unlock(&conn_mutex) < 0)
    return -1;
  if (nconns == 0)
    return -1;
  return 0;
}

int rpc_size() { return nconns; }

void *rpc_read_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  // this thread's job is to read from the connection and set the read id.

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    std::cerr << "rpc_read_thread failed to lock read mutex" << std::endl;
    return;
  }

  while (true) {
    while (conn->read_id != 0)
      pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

    // the read id is zero so it's our turn to read the next int
    if (read(conn->connfd, &conn->read_id, sizeof(int)) < 0) {
      std::cerr << "read failed" << std::endl;
      pthread_mutex_unlock(&conn->read_mutex);
      return;
    }
    if (conn->read_id != 0 && pthread_cond_broadcast(&conn->read_cond) < 0) {
      std::cerr << "rpc_read_thread failed to broadcast read_cond" << std::endl;
      pthread_mutex_unlock(&conn->read_mutex);
      return;
    }
  }
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(const int index, const unsigned int op) {
  if (rpc_open() < 0 || pthread_mutex_lock(&conns[index].write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  conns[index].write_iov_count = 2; // skip 2 for the header
  conns[index].write_id = ++(conns[index].request_id);
  conns[index].write_op = op;
  return 0;
}
// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_response(const int index, const int read_id) {
  if (rpc_open() < 0 || pthread_mutex_lock(&conns[index].write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  conns[index].write_iov_count = 1; // skip 2 for the header
  conns[index].write_id = read_id;
  conns[index].write_op = -1;
  return 0;
}

// rpc_write writes data to the current request builder on the given connection
int rpc_write(const int index, const void *data, const size_t size) {
  conns[index].write_iov[conns[index].write_iov_count++] = {
      const_cast<void *>(data), size};
  return 0;
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(const int index) {
  conn_t conn = conns[index];
  conn.write_iov[0] = {&conn.write_id, sizeof(int)};
  if (conn.write_op != -1) {
    conn.write_iov[1] = {&conn.write_op, sizeof(unsigned int)};
  }

  // write the request to the server
  if (writev(conn.connfd, conn.write_iov, conn.write_iov_count) < 0 ||
      pthread_mutex_unlock(&conn.write_mutex) < 0)
    return -1;
  return conn.write_id;
}

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(const int index) {
  int wait_for_request_id = rpc_write_end(index);
  if (wait_for_request_id < 0)
    return -1;
  if (rpc_read_start(index, wait_for_request_id) < 0)
    return -1;
  return 0;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
int rpc_read_start(const int index, const int write_id) {
  if (pthread_mutex_lock(&conns[index].read_mutex) < 0)
    return -1;

  // wait for the active read id to be the request id we are waiting for
  while (conns[index].read_id != write_id)
    if (pthread_cond_wait(&conns[index].read_cond, &conns[index].read_mutex) <
        0)
      return -1;

  return 0;
}

// rpc_read reads data from the current response on the given connection.
int rpc_read(const int index, void *data, size_t size) {
  if (data == nullptr) {
    // temp buffer to discard data
    char tempBuffer[256];
    while (size > 0) {
      ssize_t bytesRead = read(conns[index].connfd, tempBuffer,
                               std::min(size, sizeof(tempBuffer)));
      if (bytesRead < 0) {
        pthread_mutex_unlock(&conns[index].read_mutex);
        return -1; // error if reading fails
      }
      size -= bytesRead;
    }
    return size;
  }

  ssize_t n = recv(conns[index].connfd, data, size, MSG_WAITALL);
  if (n < 0)
    pthread_mutex_unlock(&conns[index].read_mutex);
  return n;
}

// rpc_read_end releases the response lock on the given connection.
int rpc_read_end(const int index) {
  int read_id = conns[index].read_id;
  conns[index].read_id = 0;
  if (pthread_cond_broadcast(&conns[index].read_cond) < 0 ||
      pthread_mutex_unlock(&conns[index].read_mutex) < 0)
    return -1;
  return read_id;
}

void rpc_close() {
  if (pthread_mutex_lock(&conn_mutex) < 0)
    return;
  while (--nconns >= 0)
    close(conns[nconns].connfd);
  pthread_mutex_unlock(&conn_mutex);
}

void allocate_unified_mem_pointer(const int index, void *dev_ptr, size_t size) {
  // allocate new space for pointer mapping
  conns[index].unified_devices.insert({dev_ptr, size});
}

cudaError_t cuda_memcpy_unified_ptrs(const int index, cudaMemcpyKind kind) {
  for (const auto &[ptr, sz] : conns[index].unified_devices) {
    size_t size = reinterpret_cast<size_t>(sz);

    // ptr is the same on both host/device
    cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);
    if (res != cudaSuccess)
      return res;
  }
  return cudaSuccess;
}

void maybe_free_unified_mem(const int index, void *ptr) {
  for (const auto &[dev_ptr, sz] : conns[index].unified_devices) {
    size_t size = reinterpret_cast<size_t>(sz);

    if (dev_ptr == ptr) {
      munmap(dev_ptr, size);
      return;
    }
  }
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  std::cout << "cuGetProcAddress getting symbol: " << symbol << std::endl;

  auto it = get_function_pointer(symbol);
  if (it != nullptr) {
    *pfn = (void *)(&it);
    std::cout << "cuGetProcAddress: Mapped symbol '" << symbol
              << "' to function: " << *pfn << std::endl;
    return CUDA_SUCCESS;
  }

  if (strcmp(symbol, "cuGetProcAddress_v2") == 0 ||
      strcmp(symbol, "cuGetProcAddress") == 0) {
    *pfn = (void *)&cuGetProcAddress_v2;
    return CUDA_SUCCESS;
  }

  std::cout << "cuGetProcAddress: Symbol '" << symbol
            << "' not found in cudaFunctionMap." << std::endl;

  // fall back to dlsym
  static void *(*real_dlsym)(void *, const char *) = NULL;
  if (real_dlsym == NULL) {
    real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                         "GLIBC_2.2.5");
  }

  void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!libCudaHandle) {
    std::cerr << "Error: Failed to open libcuda.so" << std::endl;
    return CUDA_ERROR_UNKNOWN;
  }

  *pfn = real_dlsym(libCudaHandle, symbol);
  if (!(*pfn)) {
    std::cerr << "Error: Could not resolve symbol '" << symbol
              << "' using dlsym." << std::endl;
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}

void *dlsym(void *handle, const char *name) __THROW {
  void *func = get_function_pointer(name);

  /** proc address function calls are basically dlsym; we should handle this
   * differently at the top level. */
  if (strcmp(name, "cuGetProcAddress_v2") == 0 ||
      strcmp(name, "cuGetProcAddress") == 0) {
    return (void *)&cuGetProcAddress_v2;
  }

  if (func != nullptr) {
    // std::cout << "[dlsym] Function address from cudaFunctionMap: " << func <<
    // " " << name << std::endl;
    return func;
  }

  // Real dlsym lookup
  static void *(*real_dlsym)(void *, const char *) = NULL;
  if (real_dlsym == NULL) {
    real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                         "GLIBC_2.2.5");
  }

  // std::cout << "[dlsym] Falling back to real_dlsym for name: " << name <<
  // std::endl;
  return real_dlsym(handle, name);
}
