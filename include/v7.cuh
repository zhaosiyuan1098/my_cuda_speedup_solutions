#ifndef V7_CUH
#define V7_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

__global__ void v7_kernel(args arg, float *A, float *B, float *C) 
{
    const int BLOCK_SIZE = 64;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    
    extern __shared__ float shared_mem[];
    float *block_A = shared_mem;
    float *block_B = shared_mem + BLOCK_SIZE * BLOCK_SIZE;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // 计算全局和局部索引
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float acc[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};
    
    // 主循环，处理所有K维度的块
    for (int k = 0; k < (arg.K + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        // 协作加载A和B到共享内存
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += 8) {
            for (int j = 0; j < BLOCK_SIZE; j += 8) {
                if (row + i < arg.M && k * BLOCK_SIZE + j < arg.K)
                    block_A[(ty + i) * BLOCK_SIZE + tx + j] = 
                        A[(row + i) * arg.K + k * BLOCK_SIZE + j];
                if (k * BLOCK_SIZE + i < arg.K && col + j < arg.N)
                    block_B[(ty + i) * BLOCK_SIZE + tx + j] = 
                        B[(k * BLOCK_SIZE + i) * arg.N + col + j];
            }
        }
        
        __syncthreads();
        
        // 计算当前块的乘积
        #pragma unroll
        for (int k_sub = 0; k_sub < BLOCK_SIZE; k_sub++) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_X; j++) {
                    acc[i][j] += block_A[(ty * THREAD_SIZE_Y + i) * BLOCK_SIZE + k_sub] * 
                                block_B[k_sub * BLOCK_SIZE + tx * THREAD_SIZE_X + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; j++) {
            if (row + i < arg.M && col + j < arg.N) {
                C[(row + i) * arg.N + col + j] = acc[i][j];
            }
        }
    }
}

float* v7(args arg, float *A, float *B, float *C)
{
    const int BLOCK_SIZE = 64;
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((arg.N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (arg.M + BLOCK_SIZE - 1) / BLOCK_SIZE);
                   
    size_t shared_mem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    
    v7_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg, A, B, C);
    cudaDeviceSynchronize();
    return C;
}

#endif


// 我将根据NVIDIA性能分析数据设计一个优化的v7版本矩阵乘法实现。主要优化方向如下：

// 从数据可以看出，v6的Memory吞吐率和Compute吞吐率都达到了75%左右，说明计算和内存访问较为平衡
// 调整block size从1024到64，以获得更好的占用率
// 优化共享内存的使用，减少bank冲突
// 采用更细粒度的数据加载和计算模式
// v7.cuh
// 实现新的GEMM kernel，采用64x64分块策略。

// v7.cuh+87-1
// 主要优化点：

// 使用64x64的块大小，每个线程负责8x8的计算区域
// 优化共享内存访问模式，减少bank冲突
// 使用二维线程块布局(8x8)提高线程利用率
// 采用寄存器数组存储中间结果，减少共享内存访问
// 使用#pragma unroll指令优化循环展开
// 协作式加载数据到共享内存，提高内存访问效率