#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "args.h"
#include "origin_gemm.cuh"
#include"v1.cuh"
void init_matrix(args arg, float **A, float **B, float **C) {
    int M = arg.M;
    int K = arg.K;
    int N = arg.N;
    cudaError_t err;
    err = cudaMallocManaged((void**)A, M * K * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for A: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMallocManaged((void**)B, K * N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for B: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMallocManaged((void**)C, M * N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for C: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) {
        (*A)[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        (*B)[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        (*C)[i] = 0.0f;
    }
    std::cout << "matrix initialized with random values!" << std::endl;
}

int main() {
    args arg;

    float *A, *B, *C;
    init_matrix(arg, &A, &B, &C);
    int* origin_output = origin_gemm(arg, reinterpret_cast<int*>(A), reinterpret_cast<int*>(B), reinterpret_cast<int*>(C));
    v1(arg, A, B, C);

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}