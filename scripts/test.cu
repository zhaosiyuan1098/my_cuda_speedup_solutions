#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define N 512
#define block_size 32

#define sa(i,j) sa[((i)<<5) + (j)]
#define sb(i,j) sb[((i)<<5) + (j)]

// 宏定义，用于将二维索引映射到线性数组
#define A(x, y) A[(x) + (y) * N]
#define B(x, y) B[(x) + (y) * N]
#define C(x, y) C[(x) + (y) * N]


// 初始化随机矩阵
void init_random(float *A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CUDA 核函数：矩阵乘法
__global__ void matmul(float *A, float *B, float *C) {
    int tx = threadIdx.x;  // 当前线程在块内的列索引
    int ty = threadIdx.y;  // 当前线程在块内的行索引
    int bx = blockIdx.x;   // 当前块的列索引
    int by = blockIdx.y;   // 当前块的行索引

    __shared__ float sa[block_size*block_size];
    __shared__ float sb[block_size*block_size];
    // 偏移 A、B 和 C 到当前块
    A = &A(0, by << 5);   
    B = &B(bx << 5, 0);   
    C = &C(bx << 5, by << 5); 

    float tmp=0.;
    for (int k_count = 0; k_count<N; k_count+=block_size){
        sa(tx,ty)=A(tx,ty);
        sb(ty,tx)=B(tx,ty);
        A+=(N<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<N;inner_k_count++){
            tmp += sa(tx,inner_k_count) * sb(ty,inner_k_count);
        }
        __syncthreads();
    }
    // 将结果存储到 C
    C(tx,ty) = tmp;
}

// 主机函数：调用核函数执行矩阵乘法
void matmul_v1(float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // 分配设备内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将主机数据拷贝到设备
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // 配置网格和块大小
    dim3 threadsPerBlock(block_size, block_size);  // 每个块 32x32 个线程
    dim3 numBlocks(N / block_size, N / block_size); // 网格大小

    // 调用核函数
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // 检查内核执行是否出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // 将结果拷贝回主机
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

bool compare_matrices(float *A, float *B, int size, float tol) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tol) {
            return false;
        }
    }
    return true;
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_v1;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_C_v1 = (float*)malloc(size);

    // Initialize host arrays with random values
    srand(time(0));
    init_random(h_A, N * N);
    init_random(h_B, N * N);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host arrays to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cuBLAS: C = A * B
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication using matmul_v1
    matmul_v1(h_A, h_B, h_C_v1);

    // Compare the results
    float tol = 1e-2;
    if (compare_matrices(h_C, h_C_v1, N * N, tol)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_v1);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    return 0;
}