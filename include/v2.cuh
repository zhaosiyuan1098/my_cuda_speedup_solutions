#ifndef V2_CUH
#define V2_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

//使用共享内存block_A和block_B存放矩阵块


__global__ void v2_kernel(args arg, float *A, float *B, float *C)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = bx * blockDim.x + tx;
    int col = by * blockDim.y + ty;
    __shared__ float block_A[arg.block_size][arg.block_size];
    __shared__ float block_B[arg.block_size][arg.block_size];
    float temp = 0.0;
    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++) {
        if (row < arg.M && (i * arg.block_size + tx) < arg.K)
            block_A[ty][tx] = A[row * arg.K + i * arg.block_size + tx];
        else
            block_A[ty][tx] = 0.0;

        if (col < arg.N && (i * arg.block_size + ty) < arg.K)
            block_B[ty][tx] = B[(i * arg.block_size + ty) * arg.N + col];
        else
            block_B[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < arg.block_size; k++) {
            temp += block_A[ty][k] * block_B[k][tx];
        }
        __syncthreads();
    }

    if (row < arg.M && col < arg.N) {
        C[row * arg.N + col] = temp;
    }
}

float *v2(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    v2_kernel<<<numBlocks, threadsPerBlock>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif