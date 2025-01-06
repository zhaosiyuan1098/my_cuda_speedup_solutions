#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "args.h"
#include "origin_gemm.cuh"
#include "v1.cuh"
#include "v2.cuh"
#include "v3.cuh"
#include "v4.cuh"
#include "v5.cuh"
#include "v6.cuh"
#include "v7.cuh"
#include "v8.cuh"

void init_matrix(args arg, float **A, float **B, float **C)
{
    int M = arg.M;
    int K = arg.K;
    int N = arg.N;
    cudaError_t err;
    err = cudaMallocManaged((void **)&(*A), M * K * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMallocManaged failed for A: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMallocManaged((void **)&(*B), K * N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMallocManaged failed for B: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMallocManaged((void **)&(*C), M * N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMallocManaged failed for C: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    srand(time(NULL));
    for (int i = 0; i < M * K; i++)
    {
        (*A)[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++)
    {
        (*B)[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++)
    {
        (*C)[i] = 0.0f;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <MNK> <method>" << std::endl;
        return -1;
    }

    args arg;
    arg.M = std::atoi(argv[1]);
    arg.N = arg.M;
    arg.K = arg.M;
    int method = std::atoi(argv[2]);

    float *A, *B, *C, *C_cublas;
    init_matrix(arg, &A, &B, &C);
    cudaMallocManaged(&C_cublas, arg.M * arg.N * sizeof(float));
    std::cout << "matrix size: " << arg.M << std::endl;
#ifdef USE_CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto blas_start = std::chrono::high_resolution_clock::now();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, arg.N, arg.M, arg.K, &alpha, B, arg.N, A, arg.K, &beta, C_cublas, arg.N);
    cudaDeviceSynchronize();
    auto blas_end = std::chrono::high_resolution_clock::now();
    cublasDestroy(handle);

    std::chrono::duration<float, std::milli> blas_duration = blas_end - blas_start;
    std::cout << "Method " << "cublas" << " time: " << blas_duration.count() << " ms" << std::endl;
#endif
    auto start = std::chrono::high_resolution_clock::now();

    switch (method)
    {
    case 1:
        v1(arg, A, B, C);
        break;
    case 2:
        v2(arg, A, B, C);
        break;
    case 3:
        v3(arg, A, B, C);
        break;
    case 4:
        v4(arg, A, B, C);
        break;
    case 5:
        v5(arg, A, B, C);
        break;
    case 6:
        v6(arg, A, B, C);
        break;
    case 7:
        v7(arg, A, B, C);
        break;
    case 8:
        v8(arg, A, B, C);
        break;
    default:
        std::cerr << "Invalid method!" << std::endl;
        break;
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Method " << method << " time: " << duration.count() << " ms" << std::endl;

    // 比较结果
    bool match = true;
    for (int i = 0; i < arg.M * arg.N; i++)
    {
        if (fabs(C[i] - C_cublas[i]) > 1e-3)
        {
            match = false;
            std::cout << "Results do not match at index " << i << ": " << C[i] << " != " << C_cublas[i] << std::endl;
            break;
        }
    }

    if (match)
    {
        std::cout << "Results match!" << std::endl;
    }
    else
    {
        std::cout << "Results do not match!" << std::endl;
    }
    std::cout << std::endl;

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_cublas);

    return 0;
}