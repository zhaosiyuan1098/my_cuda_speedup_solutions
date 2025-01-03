#ifndef V2_CUH
#define V2_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

//使用共享内存block_A和block_B存放矩阵块

__global__ void v2_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + arg.block_size * arg.block_size;

    //每个线程用来计算C的一个元素
    int tx = threadIdx.x, ty = threadIdx.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0;
    //依次把block_size*block_size的A和B矩阵块加载到共享内存中
    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++) {
        if (row < arg.M && (i * arg.block_size + tx) < arg.K)
            block_A[ty * arg.block_size + tx] = A[row * arg.K + i * arg.block_size + tx];
        else
            block_A[ty * arg.block_size + tx] = 0.0;

        if (col < arg.N && (i * arg.block_size + ty) < arg.K)
            block_B[ty * arg.block_size + tx] = B[(i * arg.block_size + ty) * arg.N + col];
        else
            block_B[ty * arg.block_size + tx] = 0.0;

        __syncthreads();
        //计算C的一个元素的一部分
        for (int k = 0; k < arg.block_size; k++) {
            temp += block_A[ty * arg.block_size + k] * block_B[k * arg.block_size + tx];
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

    size_t shared_mem_size = 2 * arg.block_size * arg.block_size * sizeof(float);
    v2_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif