#ifndef ORIGIN_GEMM_CUH
#define ORIGIN_GEMM_CUH

#include "cuda_runtime.h"
#include "args.h"

__global__ void origin_gemm_thread(int M, int N, int K, int *A, int *B, int *C);

void origin_gemm(args arg ,int *A, int *B, int *C);


#endif