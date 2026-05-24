#include <dlfcn.h>
#if defined(__linux__)
#include <features.h>
#endif

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: client_loader_test <libcuda.so.1>" << std::endl;
    return 2;
  }

  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    std::cerr << "dlopen failed: " << dlerror() << std::endl;
    return 1;
  }

  using DlsymFn = void *(*)(void *, const char *);
  auto client_dlsym = reinterpret_cast<DlsymFn>(dlsym(handle, "dlsym"));
#if defined(__GLIBC__)
  if (client_dlsym == nullptr) {
    std::cerr << "client library does not export dlsym" << std::endl;
    return 1;
  }
#endif

  if (dlsym(handle, "cuGetProcAddress") == nullptr) {
    std::cerr << "client library does not export cuGetProcAddress" << std::endl;
    return 1;
  }

  if (dlsym(handle, "cuInit") == nullptr) {
    std::cerr << "client library does not export cuInit" << std::endl;
    return 1;
  }

  void *printf_symbol = dlsym(RTLD_DEFAULT, "printf");
  if (printf_symbol == nullptr) {
    std::cerr << "default dlsym lookup failed after loading client library"
              << std::endl;
    return 1;
  }

  if (client_dlsym != nullptr) {
    void *printf_via_client = client_dlsym(RTLD_DEFAULT, "printf");
    if (printf_via_client == nullptr) {
      std::cerr << "client dlsym interposer did not delegate printf"
                << std::endl;
      return 1;
    }

    void *missing_via_client =
        client_dlsym(RTLD_DEFAULT, "cuDefinitelyMissingForDlsymTest");
    if (missing_via_client == nullptr) {
      std::cerr << "client dlsym interposer did not create a missing CUDA stub"
                << std::endl;
      return 1;
    }
  }

  using CuGetProcAddressV2Fn = int (*)(const char *, void **, int,
                                      unsigned long long, int *);
  auto cu_get_proc_address_v2 = reinterpret_cast<CuGetProcAddressV2Fn>(
      dlsym(handle, "cuGetProcAddress_v2"));
  if (cu_get_proc_address_v2 == nullptr) {
    std::cerr << "client library does not export cuGetProcAddress_v2"
              << std::endl;
    return 1;
  }

  void *missing_via_proc_address = nullptr;
  int symbol_status = -1;
  int result = cu_get_proc_address_v2("cuDefinitelyMissingForProcAddressTest",
                                      &missing_via_proc_address, 13000, 0,
                                      &symbol_status);
  if (result != 0 || missing_via_proc_address == nullptr || symbol_status != 0) {
    std::cerr << "cuGetProcAddress_v2 missing CUDA stub lookup failed: result="
              << result << " symbol_status=" << symbol_status << std::endl;
    return 1;
  }

  dlclose(handle);
  return 0;
}
