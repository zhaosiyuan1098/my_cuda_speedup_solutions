#include "global_gemm.cuh"

__global__ void global_gemm_thread(int M, int N, int K, int *A, int *B, int *C)
{
    printf("global_gemm_thread\n");
}

int* global_gemm(int M, int N, int K, int *A, int *B, int *C)
{
    printf("global_gemm\n");
}