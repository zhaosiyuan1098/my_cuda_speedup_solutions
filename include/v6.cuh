#ifndef V6_CUH
#define V6_CUH



#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

__global__ void v6_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A[2] = {shared_mem, 
                        shared_mem + arg.block_size * arg.block_size};
    float *block_B[2] = {shared_mem + 2 * arg.block_size * arg.block_size,
                        shared_mem + 2 * arg.block_size * arg.block_size + 
                        arg.block_size * (arg.block_size + 1)};

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float temp = 0.0f;

    // 预加载第一块数据
    int current = 0;
    if (row < arg.M && tx < arg.K)
        block_A[current][ty * arg.block_size + tx] = A[row * arg.K + tx];
    else
        block_A[current][ty * arg.block_size + tx] = 0.0f;

    if (col < arg.N && ty < arg.K)
        block_B[current][ty * (arg.block_size + 1) + tx] = B[ty * arg.N + col];
    else
        block_B[current][ty * (arg.block_size + 1) + tx] = 0.0f;

    __syncthreads();

    // 主计算循环
    #pragma unroll 
    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++)
    {
        // 预加载下一块数据到另一个缓冲区
        if (i < (arg.K + arg.block_size - 1) / arg.block_size - 1)
        {
            int next = 1 - current;
            int next_idx = (i + 1) * arg.block_size;
            
            if (row < arg.M && (next_idx + tx) < arg.K)
                block_A[next][ty * arg.block_size + tx] = 
                    A[row * arg.K + next_idx + tx];
            else
                block_A[next][ty * arg.block_size + tx] = 0.0f;

            if (col < arg.N && (next_idx + ty) < arg.K)
                block_B[next][ty * (arg.block_size + 1) + tx] = 
                    B[(next_idx + ty) * arg.N + col];
            else
                block_B[next][ty * (arg.block_size + 1) + tx] = 0.0f;
        }

        // 计算当前块
        #pragma unroll 
        for (int k = 0; k < arg.block_size && (i * arg.block_size + k) < arg.K; k++)
        {
            temp += block_A[current][ty * arg.block_size + k] * 
                   block_B[current][k * (arg.block_size + 1) + tx];
        }

        current = 1 - current;
        __syncthreads();
    }

    // 写回结果
    if (row < arg.M && col < arg.N)
    {
        C[row * arg.N + col] = temp;
    }
}

float* v6(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 双缓冲的共享内存大小计算
    size_t shared_mem_size = (2 * arg.block_size * arg.block_size + 
                             2 * arg.block_size * (arg.block_size + 1)) * sizeof(float);

    v6_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif
