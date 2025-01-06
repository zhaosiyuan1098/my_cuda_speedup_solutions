#ifndef V8_OPTIMIZED_CUH
#define V8_OPTIMIZED_CUH

#include "cuda_runtime.h"
#include "args.h"
#include "utils.h"

// Simplified macros for indexing
#define A(i, j) A[(i) + (j) * lda]
#define B(i, j) B[(i) + (j) * ldb]
#define C(i, j) C[(i) + (j) * ldc]
#define sa(i, j) sa[(i) * block_size + (j)]
#define sb(i, j) sb[(i) * block_size + (j)]

__global__ void v8_kernel(int N, int block_size, float *A, float *B, float *C, int lda, int ldb, int ldc)
{
    // Thread and block indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Coordinates in C
    int row = by * block_size + ty;
    int col = bx * block_size + tx;

    // Shared memory for submatrices
    extern __shared__ float shared_mem[];
    float *sa = shared_mem;                                // Shared A block
    float *sb = shared_mem + block_size * block_size;      // Shared B block

    // Temporary accumulator for the result
    float temp = 0.0f;

    // Offset pointers for A and B
    A += row * lda;
    B += col;

    // Iterate over submatrices of A and B
    for (int k = 0; k < N; k += block_size)
    {
        // Load submatrices into shared memory
        if (row < N && (k + tx) < N)
            sa(ty, tx) = A[tx];
        else
            sa(ty, tx) = 0.0f;

        if (col < N && (k + ty) < N)
            sb(ty, tx) = B[ty * ldb];
        else
            sb(ty, tx) = 0.0f;

        __syncthreads();

        // Perform the multiplication for this block
        #pragma unroll
        for (int p = 0; p < block_size; ++p)
        {
            temp += sa(ty, p) * sb(p, tx);
        }

        __syncthreads();

        // Update A and B pointers
        A += block_size;
        B += block_size * ldb;
    }

    // Write the result to C
    if (row < N && col < N)
        C[row * ldc + col] = temp;
}

float *v8(args arg, float *A, float *B, float *C)
{
    int lda = arg.N, ldb = arg.N, ldc = arg.N;
    dim3 threadsPerBlock(arg.block_size, arg.block_size);
    dim3 numBlocks((arg.N + arg.block_size - 1) / arg.block_size, 
                   (arg.N + arg.block_size - 1) / arg.block_size);

    // Shared memory size calculation
    size_t shared_mem_size = 2 * arg.block_size * arg.block_size * sizeof(float);

    // Kernel launch
    v8_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(arg.N, arg.block_size, A, B, C, lda, ldb, ldc);
    cudaDeviceSynchronize();
    return C;
}

#endif