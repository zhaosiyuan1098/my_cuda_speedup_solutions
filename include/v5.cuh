#ifndef V5_CUH
#define V5_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

__global__ void v5_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + arg.block_size * (arg.block_size + 1);

    int tx = threadIdx.x, ty = threadIdx.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++) {
        int tiledRow = i * arg.block_size + tx;
        int tiledCol = i * arg.block_size + ty;

        block_A[ty * (arg.block_size+1) + tx] = (row < arg.M && tiledRow < arg.K) ? A[row * arg.K + tiledRow] : 0.0f;
        block_B[tx *( arg.block_size+1) + ty] = (col < arg.N && tiledCol < arg.K) ? B[tiledCol * arg.N + col] : 0.0f;

        __syncthreads();

        // 在共享内存中累加相乘结果
        for (int k = 0; k < arg.block_size; k++) {
            temp += block_A[ty * (arg.block_size+1) + k] * block_B[tx * (arg.block_size+1) + k];
        }
        __syncthreads();
    }

    if (row < arg.M && col < arg.N) {
        C[row * arg.N + col] = temp;
    }
}

// 入口函数
float* v5(args arg, float *A, float *B, float *C)
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