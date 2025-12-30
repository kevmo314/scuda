#include <arpa/inet.h>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
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
#include <vector>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>

#include "codegen/gen_client.h"
#include "rpc.h"

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;

const char *DEFAULT_PORT = "14833";

static int init = 0;
static jmp_buf catch_segfault;
static void *faulting_address = nullptr;

std::map<conn_t *, std::map<void *, size_t>> unified_devices;
// Track pinned host memory separately - needs segfault handling but NOT kernel launch sync
std::map<conn_t *, std::map<void *, size_t>> pinned_host_devices;
// Track which page ranges have been mapped (start address, end address)
// Using uintptr_t for atomic-safe access in signal handler
struct MappedRange {
  uintptr_t start;
  uintptr_t end;
};
volatile MappedRange mapped_ranges[1024];
volatile int mapped_range_count = 0;
std::map<void *, void *> host_funcs;

static bool is_ptr_mapped(void *ptr) {
  uintptr_t addr = (uintptr_t)ptr;
  for (int i = 0; i < mapped_range_count; i++) {
    if (addr >= mapped_ranges[i].start && addr < mapped_ranges[i].end) {
      return true;
    }
  }
  return false;
}

// Check if all pages from aligned_ptr to aligned_ptr+size are already mapped
static bool is_range_fully_mapped(uintptr_t start, uintptr_t end) {
  // Check if this range is fully covered by existing mappings
  for (int i = 0; i < mapped_range_count; i++) {
    if (mapped_ranges[i].start <= start && mapped_ranges[i].end >= end) {
      return true;
    }
  }
  return false;
}

static void mark_range_mapped(uintptr_t start, uintptr_t end) {
  if (mapped_range_count < 1024) {
    // Try to extend an existing range if possible
    for (int i = 0; i < mapped_range_count; i++) {
      // Check for overlap or adjacency
      if (start <= mapped_ranges[i].end && end >= mapped_ranges[i].start) {
        // Extend the range
        if (start < mapped_ranges[i].start) mapped_ranges[i].start = start;
        if (end > mapped_ranges[i].end) mapped_ranges[i].end = end;
        return;
      }
    }
    // No overlap, add new range
    mapped_ranges[mapped_range_count].start = start;
    mapped_ranges[mapped_range_count].end = end;
    mapped_range_count++;
  }
}

static void mark_ptr_mapped(void *ptr) {
  // Legacy function - not used anymore but kept for compatibility
  (void)ptr;
}

// Helper to map a range, handling partial overlaps with existing mappings
static bool map_range_safe(uintptr_t start, uintptr_t end) {
  // Check if already fully mapped
  if (is_range_fully_mapped(start, end)) {
    return true;
  }

  // For simplicity, if any part is already mapped, just return true
  // (the data should still be accessible for the overlapping parts)
  // We only need to map pages that aren't already mapped

  size_t page_size = sysconf(_SC_PAGESIZE);
  uintptr_t current = start;

  while (current < end) {
    // Check if this page is already in a mapped range
    bool page_mapped = false;
    for (int i = 0; i < mapped_range_count; i++) {
      if (current >= mapped_ranges[i].start && current < mapped_ranges[i].end) {
        page_mapped = true;
        // Skip to end of this mapped range
        current = mapped_ranges[i].end;
        break;
      }
    }

    if (!page_mapped) {
      // Find the extent of unmapped pages
      uintptr_t unmapped_start = current;
      uintptr_t unmapped_end = end;

      // Check if there's a mapped range that starts before our end
      for (int i = 0; i < mapped_range_count; i++) {
        if (mapped_ranges[i].start > current && mapped_ranges[i].start < unmapped_end) {
          unmapped_end = mapped_ranges[i].start;
        }
      }

      size_t map_size = unmapped_end - unmapped_start;
      void *allocated = mmap((void*)unmapped_start, map_size, PROT_READ | PROT_WRITE,
                             MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (allocated == MAP_FAILED) {
        return false;
      }

      mark_range_mapped(unmapped_start, unmapped_end);
      current = unmapped_end;
    }
  }

  return true;
}

// Ensure a unified or pinned host pointer has local memory mapped (for receiving DeviceToHost data)
bool ensure_unified_ptr_mapped(conn_t *conn, void *ptr) {
  size_t page_size = sysconf(_SC_PAGESIZE);

  // Check unified devices first
  auto conn_it = unified_devices.find(conn);
  if (conn_it != unified_devices.end()) {
    auto &devices = conn_it->second;
    auto device_it = devices.find(ptr);
    if (device_it != devices.end()) {
      size_t size = device_it->second;
      // Page-align the pointer for mmap
      uintptr_t aligned_start = (uintptr_t)ptr & ~(page_size - 1);
      size_t offset = (uintptr_t)ptr - aligned_start;
      size_t aligned_size = ((size + offset + page_size - 1) & ~(page_size - 1));
      uintptr_t aligned_end = aligned_start + aligned_size;

      return map_range_safe(aligned_start, aligned_end);
    }
  }

  // Check pinned host devices
  auto pinned_it = pinned_host_devices.find(conn);
  if (pinned_it != pinned_host_devices.end()) {
    auto &devices = pinned_it->second;
    auto device_it = devices.find(ptr);
    if (device_it != devices.end()) {
      size_t size = device_it->second;
      // Page-align the pointer for mmap
      uintptr_t aligned_start = (uintptr_t)ptr & ~(page_size - 1);
      size_t offset = (uintptr_t)ptr - aligned_start;
      size_t aligned_size = ((size + offset + page_size - 1) & ~(page_size - 1));
      uintptr_t aligned_end = aligned_start + aligned_size;

      return map_range_safe(aligned_start, aligned_end);
    }
  }

  return false;
}

// Signal-safe version of mapping - maps only the faulting page if not already mapped
static void segfault_map_page(uintptr_t fault_addr, uintptr_t alloc_start, size_t alloc_size) {
  size_t page_size = sysconf(_SC_PAGESIZE);

  // Calculate the page-aligned start of the entire allocation
  uintptr_t alloc_aligned_start = alloc_start & ~(page_size - 1);
  size_t offset = alloc_start - alloc_aligned_start;
  size_t alloc_aligned_size = ((alloc_size + offset + page_size - 1) & ~(page_size - 1));
  uintptr_t alloc_aligned_end = alloc_aligned_start + alloc_aligned_size;

  // Check if this entire range is already mapped
  if (is_range_fully_mapped(alloc_aligned_start, alloc_aligned_end)) {
    return; // Already mapped, nothing to do
  }

  // Find the page containing the fault
  uintptr_t fault_page_start = fault_addr & ~(page_size - 1);

  // Check if this specific page is already in a mapped range
  for (int i = 0; i < mapped_range_count; i++) {
    if (fault_page_start >= mapped_ranges[i].start &&
        fault_page_start < mapped_ranges[i].end) {
      // Page already mapped, but we got a segfault?
      // This shouldn't happen, but just return
      return;
    }
  }

  // Map only the unmapped portion of this allocation
  // For safety, map from alloc_aligned_start to alloc_aligned_end
  // but only the pages that aren't already mapped

  uintptr_t current = alloc_aligned_start;
  while (current < alloc_aligned_end) {
    // Check if current is already in a mapped range
    bool page_mapped = false;
    uintptr_t next_unmapped = current;

    for (int i = 0; i < mapped_range_count; i++) {
      if (current >= mapped_ranges[i].start && current < mapped_ranges[i].end) {
        page_mapped = true;
        next_unmapped = mapped_ranges[i].end;
        break;
      }
    }

    if (page_mapped) {
      current = next_unmapped;
      continue;
    }

    // Find end of unmapped region
    uintptr_t unmapped_end = alloc_aligned_end;
    for (int i = 0; i < mapped_range_count; i++) {
      if (mapped_ranges[i].start > current && mapped_ranges[i].start < unmapped_end) {
        unmapped_end = mapped_ranges[i].start;
      }
    }

    size_t map_size = unmapped_end - current;
    void *allocated = mmap((void*)current, map_size, PROT_READ | PROT_WRITE,
                           MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocated == MAP_FAILED) {
      // Can't call perror in signal handler safely, just exit
      _exit(1);
    }

    mark_range_mapped(current, unmapped_end);
    current = unmapped_end;
  }
}

static void segfault(int sig, siginfo_t *info, void *unused) {
  faulting_address = info->si_addr;
  uintptr_t fault_addr = (uintptr_t)faulting_address;

  // Check unified memory pointers
  for (const auto &conn_entry : unified_devices) {
    for (const auto &[ptr, sz] : conn_entry.second) {
      uintptr_t alloc_start = (uintptr_t)ptr;
      if (alloc_start <= fault_addr && fault_addr < (alloc_start + sz)) {
        segfault_map_page(fault_addr, alloc_start, sz);
        return;
      }
    }
  }

  // Check pinned host memory pointers
  for (const auto &conn_entry : pinned_host_devices) {
    for (const auto &[ptr, sz] : conn_entry.second) {
      uintptr_t alloc_start = (uintptr_t)ptr;
      if (alloc_start <= fault_addr && fault_addr < (alloc_start + sz)) {
        segfault_map_page(fault_addr, alloc_start, sz);
        return;
      }
    }
  }

  // Not a tracked pointer - restore default handler and re-raise
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

int is_unified_pointer(conn_t *conn, void *arg) {
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return 0;
  }

  // now check if the argument exists in the connection's mapped devices
  auto &devices = conn_it->second;
  if (devices.find(arg) != devices.end()) {
    return 1;
  }

  return 0;
}

int maybe_copy_unified_arg(conn_t *conn, void *arg, enum cudaMemcpyKind kind) {
  // find the connection in the map first
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return 0;
  }

  // now find the argument in the sub-map for this connection
  auto &devices = conn_it->second;
  auto device_it = devices.find(arg);
  if (device_it != devices.end()) {
    std::cout << "Found unified arg pointer; copying..." << std::endl;

    void *ptr = device_it->first;
    size_t size = device_it->second;

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

  init = 1;
}

void rpc_close(conn_t *conn) {
  if (pthread_mutex_lock(&conn_mutex) < 0)
    return;
  while (--nconns >= 0)
    close(conn->connfd);
  pthread_mutex_unlock(&conn_mutex);
}

typedef void (*func_t)(void *);

void add_host_node(void *fn, void *udata) { host_funcs[fn] = udata; }

void invoke_host_func(void *fn) {
  for (const auto &pair : host_funcs) {
    if (pair.first == fn) {
      func_t func = reinterpret_cast<func_t>(pair.first);
      std::cout << "Invoking function at: " << pair.first << std::endl;
      func(pair.second);
      return;
    }
  }
}

void *rpc_client_dispatch_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  int op;

  while (true) {
    op = rpc_dispatch(conn, 1);

    if (op == 1) {
      std::cout << "Transferring memory..." << std::endl;

      int found = 0;

      rpc_read(conn, &found, sizeof(int));

      if (found > 0) {
        void *host_data = nullptr;
        void *dst = nullptr;
        const void *src = nullptr;
        size_t count = 0;
        cudaError_t result;
        int request_id;

        if (rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0) {
          std::cerr << "Failed to read transfer parameters." << std::endl;
          break;
        }

        host_data = malloc(count);
        if (!host_data) {
          std::cerr << "Memory allocation failed." << std::endl;
          break;
        }

        // Read the actual data from the server (sent from `src` in device
        // memory)
        if (rpc_read(conn, host_data, count) < 0) {
          std::cerr << "Failed to read device data from server." << std::endl;
          free(host_data);
          break;
        }

        // Copy received data to the destination (dst) on the host
        memcpy(dst, host_data, count);
      }

      void *temp_mem;
      if (rpc_read(conn, &temp_mem, sizeof(void *)) <= 0) {
        std::cerr << "rpc_read failed for mem. Closing connection."
                  << std::endl;
        break;
      }

      int request_id = rpc_read_end(conn);
      void *mem = temp_mem;

      if (mem == nullptr) {
        std::cerr << "Invalid function pointer!" << std::endl;
        continue;
      }

      invoke_host_func(mem);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        std::cerr << "rpc_write failed. Closing connection." << std::endl;
        break;
      }
    }
  }

  std::cerr << "Exiting dispatch thread due to an error." << std::endl;
  return nullptr;
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

  std::cout << "Opening connection to server" << std::endl;

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

    memset(&conns[nconns], 0, sizeof(conn_t));
    conns[nconns].connfd = sockfd;
    conns[nconns].request_id = 0;
    if (pthread_mutex_init(&conns[nconns].read_mutex, NULL) < 0 ||
        pthread_mutex_init(&conns[nconns].write_mutex, NULL) < 0 ||
        pthread_cond_init(&conns[nconns].read_cond, NULL) < 0) {
      return -1;
    }

    pthread_create(&conns[nconns].read_thread, NULL, rpc_client_dispatch_thread,
                   (void *)&conns[nconns]);

    nconns++;
  }

  if (pthread_mutex_unlock(&conn_mutex) < 0)
    return -1;
  if (nconns == 0)
    return -1;
  return 0;
}

conn_t *rpc_client_get_connection(unsigned int index) {
  if (rpc_open() < 0)
    return nullptr;
  return &conns[index];
}

int rpc_size() { return nconns; }

void allocate_unified_mem_pointer(conn_t *conn, void *dev_ptr, size_t size) {
  // allocate new space for pointer mapping
  unified_devices[conn][dev_ptr] = size;
}

void allocate_pinned_host_pointer(conn_t *conn, void *host_ptr, size_t size) {
  // Track pinned host memory for segfault handling (but NOT for kernel launch sync)
  pinned_host_devices[conn][host_ptr] = size;
}

cudaError_t cuda_memcpy_unified_ptrs(conn_t *conn, cudaMemcpyKind kind) {
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return cudaSuccess;
  }

  for (const auto &[ptr, sz] : conn_it->second) {
    // Only sync pointers that have been locally mapped
    if (!is_ptr_mapped(ptr)) {
      continue;
    }

    size_t size = reinterpret_cast<size_t>(sz);

    // ptr is the same on both host/device
    cudaError_t res = cudaMemcpy(ptr, ptr, size, kind);
    if (res != cudaSuccess) {
      return res;
    }
  }
  return cudaSuccess;
}

void maybe_free_unified_mem(conn_t *conn, void *ptr) {
  auto conn_it = unified_devices.find(conn);
  if (conn_it == unified_devices.end()) {
    return;
  }

  for (const auto &[dev_ptr, sz] : conn_it->second) {
    size_t size = reinterpret_cast<size_t>(sz);

    if (dev_ptr == ptr) {
      munmap(dev_ptr, size);
      return;
    }
  }
}

void maybe_free_pinned_host(conn_t *conn, void *ptr) {
  auto conn_it = pinned_host_devices.find(conn);
  if (conn_it == pinned_host_devices.end()) {
    return;
  }

  for (const auto &[host_ptr, sz] : conn_it->second) {
    size_t size = reinterpret_cast<size_t>(sz);

    if (host_ptr == ptr) {
      munmap(host_ptr, size);
      pinned_host_devices[conn].erase(host_ptr);
      return;
    }
  }
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  auto it = get_function_pointer(symbol);
  if (it != nullptr) {
    *pfn = (void *)(&it);
    return CUDA_SUCCESS;
  }

  if (strcmp(symbol, "cuGetProcAddress_v2") == 0 ||
      strcmp(symbol, "cuGetProcAddress") == 0) {
    *pfn = (void *)&cuGetProcAddress_v2;
    return CUDA_SUCCESS;
  }

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
    // std::cout << "[dlsym] Function address from cudaFunctionMap: " << func
    // << " " << name << std::endl;
    return func;
  }

  // Real dlsym lookup
  static void *(*real_dlsym)(void *, const char *) = NULL;
  if (real_dlsym == NULL) {
    real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym",
                                                         "GLIBC_2.2.5");
  }

  return real_dlsym(handle, name);
}
