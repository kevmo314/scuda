#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[])
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;
    const int batch_count = 2;

    const std::vector<std::vector<data_type>> A_array = {{1.0, 3.0, 2.0, 4.0},
                                                         {5.0, 7.0, 6.0, 8.0}};
    const std::vector<std::vector<data_type>> B_array = {{5.0, 7.0, 6.0, 8.0},
                                                         {9.0, 11.0, 10.0, 12.0}};
    std::vector<std::vector<data_type>> C_array(batch_count, std::vector<data_type>(m * n));

    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type **d_A_array = nullptr;
    data_type **d_B_array = nullptr;
    data_type **d_C_array = nullptr;

    std::vector<data_type *> d_A(batch_count, nullptr);
    std::vector<data_type *> d_B(batch_count, nullptr);
    std::vector<data_type *> d_C(batch_count, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("A[0]\n");
    print_matrix(m, k, A_array[0].data(), lda);
    printf("=====\n");

    printf("A[1]\n");
    print_matrix(m, k, A_array[1].data(), lda);
    printf("=====\n");

    printf("B[0]\n");
    print_matrix(k, n, B_array[0].data(), ldb);
    printf("=====\n");

    printf("B[1]\n");
    print_matrix(k, n, B_array[1].data(), ldb);
    printf("=====\n");

    /* Step 1: Create cuBLAS handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* Step 2: Allocate unified memory */
    CUDA_CHECK(cudaMallocManaged(&d_A_array, sizeof(data_type *) * batch_count));
    CUDA_CHECK(cudaMallocManaged(&d_B_array, sizeof(data_type *) * batch_count));
    CUDA_CHECK(cudaMallocManaged(&d_C_array, sizeof(data_type *) * batch_count));

    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaMallocManaged(&d_A[i], sizeof(data_type) * A_array[i].size()));
        CUDA_CHECK(cudaMallocManaged(&d_B[i], sizeof(data_type) * B_array[i].size()));
        CUDA_CHECK(cudaMallocManaged(&d_C[i], sizeof(data_type) * C_array[i].size()));

        // Copy data to unified memory (host-side initialization is sufficient)
        std::copy(A_array[i].begin(), A_array[i].end(), d_A[i]);
        std::copy(B_array[i].begin(), B_array[i].end(), d_B[i]);

        d_A_array[i] = d_A[i];
        d_B_array[i] = d_B[i];
        d_C_array[i] = d_C[i];
    }

    /* Step 3: Compute */
    CUBLAS_CHECK(cublasDgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda,
                                    d_B_array, ldb, &beta, d_C_array, ldc, batch_count));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Step 4: Verify results */
    printf("C[0]\n");
    print_matrix(m, n, d_C[0], ldc);
    printf("=====\n");

    printf("C[1]\n");
    print_matrix(m, n, d_C[1], ldc);
    printf("=====\n");

    /* Free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batch_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
