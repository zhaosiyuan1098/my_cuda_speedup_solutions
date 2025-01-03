#ifndef V3_CUH
#define V3_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

//避免条件分化+省寄存器

__global__ void v3_kernel(args arg, float *A, float *B, float *C)
{
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + arg.block_size * arg.block_size;

    int tx = threadIdx.x, ty = threadIdx.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0;

    for (int i = 0; i < (arg.K + arg.block_size - 1) / arg.block_size; i++) {
        int tiledRow = i * arg.block_size + tx;
        int tiledCol = i * arg.block_size + ty;

        block_A[ty * arg.block_size + tx] = (row < arg.M && tiledRow < arg.K) ? A[row * arg.K + tiledRow] : 0.0;
        block_B[ty * arg.block_size + tx] = (col < arg.N && tiledCol < arg.K) ? B[tiledCol * arg.N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < arg.block_size; k++) {
            temp += block_A[ty * arg.block_size + k] * block_B[k * arg.block_size + tx];
        }
        __syncthreads();
    }

    if (row < arg.M && col < arg.N) {
        C[row * arg.N + col] = temp;
    }
}

float *v3(args arg, float *A, float *B, float *C)
{
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t shared_mem_size = 2 * arg.block_size * arg.block_size * sizeof(float);
    v3_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif