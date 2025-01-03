#ifndef V1_CUH
#define V1_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"


//最原始的gemm实现


__global__ void v1_kernel(args arg, float *A, float *B, float *C)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = bx * blockDim.x + tx;
    int col = by * blockDim.y + ty;
    float temp = 0.0;
    if (row < arg.M && col < arg.N) // 添加边界检查
    {
        for (int k_count = 0; k_count < arg.K; k_count++)
        {
            temp += A[row * arg.K + k_count] * B[k_count * arg.N + col];
        }
        C[row * arg.N + col] = temp;
    }
}

float *v1(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    v1_kernel<<<numBlocks, threadsPerBlock>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif