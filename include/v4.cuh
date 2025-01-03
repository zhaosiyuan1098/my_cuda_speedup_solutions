#ifndef V4_CUH
#define V4_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

// 顺序读取内存+避免bankconflict

__global__ void v4_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + arg.block_size * arg.block_size;

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

        // 加载 B 的块，转置存储
        if (col < arg.N && (i * arg.block_size + ty) < arg.K)
            block_B[tx * arg.block_size + ty] = B[(i * arg.block_size + ty) * arg.N + col];
        else
            block_B[tx * arg.block_size + ty] = 0.0;

        __syncthreads();

        // 计算 C 的一个元素的一部分
        for (int k = 0; k < arg.block_size && (i * arg.block_size + k) < arg.K; k++)
        {
            temp += block_A[ty * arg.block_size + k] * block_B[tx * arg.block_size + k];
        }
        __syncthreads();
    }

    // 写回结果
    if (row < arg.M && col < arg.N)
    {
        C[row * arg.N + col] = temp;
    }
}

float *v4(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 修改共享内存大小计算，不再需要padding
    size_t shared_mem_size = 2 * arg.block_size * arg.block_size * sizeof(float);
    v4_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif