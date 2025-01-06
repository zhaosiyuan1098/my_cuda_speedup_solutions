#ifndef V8_CUH
#define V8_CUH

#include "cuda_runtime.h"
#include "args.h"  // 包含参数定义
#include "utils.h" // 包含辅助工具函数

#define A(x, y) A[(x) + (y) * N]
#define B(x, y) B[(x) + (y) * N]
#define C(x, y) C[(x) + (y) * N]
#define block_A(x, y) block_A[(x)* block_size + (y) ]
#define block_B(x, y) block_B[(x)* block_size + (y) ]

__global__ void v8_kernel(int N, int block_size, float *A, float *B, float *C)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + block_size * block_size;

    A = &A(bx * block_size, 0);
    B = &B(0, by * block_size);
    C = &C(bx * block_size, by * block_size);

    float sum = 0.0f;

    for (int w = 0; w < N; w += block_size)
    {
        if (tx + bx * block_size < N && ty + w < N)
            block_A(tx, ty) = A(tx, ty);
        else
            block_A(tx, ty) = 0.0f;

        if (tx + w < N && ty + by * block_size < N)
            block_B(tx, ty) = B(tx, ty);
        else
            block_B(tx, ty) = 0.0f;

        __syncthreads();

        for (int q = 0; q < block_size; ++q)
        {
            sum += block_A(tx, q) * block_B(q, ty);
        }

        __syncthreads();
        A += N * block_size;
        B += block_size;
    }

    if (tx + bx * block_size < N && ty + by * block_size < N)
        C(tx, ty) = sum;
}

float *v8(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x, (arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t shared_mem_size = 2 * arg.block_size * arg.block_size * sizeof(float);
    v8_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg.N, arg.block_size, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif