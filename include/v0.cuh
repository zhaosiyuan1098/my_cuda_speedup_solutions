#ifndef V0_CUH
#define V0_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Function to perform matrix multiplication using cuBLAS
void matrixMultiply(const float* A, const float* B, float* C, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform the matrix multiplication: C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);

    cublasDestroy(handle);
}

#endif // V0_CUH