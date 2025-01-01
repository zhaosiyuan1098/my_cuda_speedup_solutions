#ifndef GLOBAL_GEMM_CUH
#define GLOBAL_GEMM_CUH

#include <cuda_runtime.h>
#include <iostream>

__global__ void global_gemm_thread(int M, int N, int K, int *A, int *B, int *C);

int* global_gemm(int M, int N, int K, int *A, int *B, int *C);

#endif
