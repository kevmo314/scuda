#include <cublas_v2.h>
#include <cuda.h>

#include <cstdio>
#include <vector>

static void check_cu(CUresult result, const char *call) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s failed: %d %s\n", call, static_cast<int>(result),
                 name == nullptr ? "" : name);
    std::exit(1);
  }
}

static void check_cublas(cublasStatus_t status, const char *call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(status));
    std::exit(1);
  }
}

static void run_device(int ordinal) {
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  check_cu(cuDeviceGet(&dev, ordinal), "cuDeviceGet");
  check_cu(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check_cu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  constexpr int n = 64;
  const size_t bytes = n * n * sizeof(float);
  std::vector<float> h_a(n * n, 1.0f);
  std::vector<float> h_b(n * n, 2.0f);
  std::vector<float> h_c(n * n, 0.0f);

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  check_cu(cuMemAlloc(&d_a, bytes), "cuMemAlloc A");
  check_cu(cuMemAlloc(&d_b, bytes), "cuMemAlloc B");
  check_cu(cuMemAlloc(&d_c, bytes), "cuMemAlloc C");
  check_cu(cuMemcpyHtoD(d_a, h_a.data(), bytes), "cuMemcpyHtoD A");
  check_cu(cuMemcpyHtoD(d_b, h_b.data(), bytes), "cuMemcpyHtoD B");

  cublasHandle_t handle = nullptr;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                           reinterpret_cast<const float *>(d_a), n,
                           reinterpret_cast<const float *>(d_b), n, &beta,
                           reinterpret_cast<float *>(d_c), n),
               "cublasSgemm");
  check_cu(cuCtxSynchronize(), "cuCtxSynchronize");
  check_cu(cuMemcpyDtoH(h_c.data(), d_c, bytes), "cuMemcpyDtoH C");
  check_cublas(cublasDestroy(handle), "cublasDestroy");
  check_cu(cuMemFree(d_a), "cuMemFree A");
  check_cu(cuMemFree(d_b), "cuMemFree B");
  check_cu(cuMemFree(d_c), "cuMemFree C");

  std::printf("device %d c0=%f expected=%f\n", ordinal, h_c[0],
              static_cast<float>(n * 2));
  if (h_c[0] != static_cast<float>(n * 2)) {
    std::exit(2);
  }
}

int main() {
  check_cu(cuInit(0), "cuInit");
  int count = 0;
  check_cu(cuDeviceGetCount(&count), "cuDeviceGetCount");
  std::printf("count=%d\n", count);
  if (count < 2) {
    return 3;
  }
  run_device(0);
  run_device(1);
  return 0;
}
