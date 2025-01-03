#ifndef V4_CUH
#define V4_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

// block_B 矩阵以转置方式存储，于循环计算中顺序读取

__global__ void v4_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + arg.block_size * arg.block_size;

    int tx = threadIdx.x, ty = threadIdx.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++) {
        int tiledRow = i * arg.block_size + tx;
        int tiledCol = i * arg.block_size + ty;

        // 加载 A 的小块
        if (row < arg.M && tiledRow < arg.K) {
            block_A[ty * arg.block_size + tx] = A[row * arg.K + tiledRow];
        } else {
            block_A[ty * arg.block_size + tx] = 0.0f;
        }

        // 加载 B 的转置小块
        if (col < arg.N && tiledCol < arg.K) {
            block_B[tx * arg.block_size + ty] = B[tiledCol * arg.N + col];
        } else {
            block_B[tx * arg.block_size + ty] = 0.0f;
        }

        __syncthreads();

        // 在共享内存中累加相乘结果
        for (int k = 0; k < arg.block_size; k++) {
            temp += block_A[ty * arg.block_size + k] * block_B[tx * arg.block_size + k];
        }
        __syncthreads();
    }

    if (row < arg.M && col < arg.N) {
        C[row * arg.N + col] = temp;
    }
}

// 入口函数
float* v4(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks(
        (arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // 分配共享内存大小：两块 block_size*block_size
    size_t shared_mem_size = 2ULL * arg.block_size * arg.block_size * sizeof(float);
    v4_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif