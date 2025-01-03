#ifndef V5_CUH
#define V5_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

__global__ void v5_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + arg.block_size * (arg.block_size + 1); // Padding

    // 每个线程用来计算 C 的一个元素
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float temp = 0.0;

    // 依次把 block_size*block_size 的 A 和 B 矩阵块加载到共享内存中
    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++)
    {
        // 加载 A 的块
        if (row < arg.M && (i * arg.block_size + tx) < arg.K)
            block_A[ty * arg.block_size + tx] = A[row * arg.K + i * arg.block_size + tx];
        else
            block_A[ty * arg.block_size + tx] = 0.0;

        // 加载 B 的块，使用 Padding 避免 Bank Conflict
        if (col < arg.N && (i * arg.block_size + ty) < arg.K)
            block_B[ty * (arg.block_size + 1) + tx] = B[(i * arg.block_size + ty) * arg.N + col];
        else
            block_B[ty * (arg.block_size + 1) + tx] = 0.0;

        __syncthreads();

        // 计算 C 的一个元素的一部分
        for (int k = 0; k < arg.block_size && (i * arg.block_size + k) < arg.K; k++)
        {
            temp += block_A[ty * arg.block_size + k] * block_B[k * (arg.block_size + 1) + tx];
        }
        __syncthreads();
    }

    // 写回结果
    if (row < arg.M && col < arg.N)
    {
        C[row * arg.N + col] = temp;
    }
}

// 入口函数
float* v5(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 修正共享内存大小计算，考虑padding
    size_t shared_mem_size = (arg.block_size * arg.block_size + 
                             arg.block_size * (arg.block_size + 1)) * sizeof(float);
    v5_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif