#include "origin_gemm.cuh"
#include <iostream>

__global__ void origin_gemm_thread(int M, int N, int K, int *A, int *B, int *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        int sum = 0;
        for (int k = 0; k < K; k++)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int* origin_gemm(args arg, int *A, int *B, int *C)
{
    dim3 threadsPerBlock(arg.M/arg.bk, arg.M/arg.bk);
    dim3 numBlocks((arg.N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (arg.M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    std::cout << "numBlocks: " << numBlocks.x << " " << numBlocks.y << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock.x << " " << threadsPerBlock.y << std::endl;

    origin_gemm_thread<<<numBlocks, threadsPerBlock>>>(arg.M, arg.N, arg.K, A, B, C);

    cudaDeviceSynchronize();
    return C;
}
