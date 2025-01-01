#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include"args.h"


__global__ void origin_gemm_thread(){
    // printf can be used instead of std::cout in device code
    printf("Hello from origin_gemm!\n");
}


void init_matrix(args arg, int **A, int **B, int **C){
    int M = arg.M;
    int K = arg.K;
    int N = arg.N;
    int bk = arg.bk;
    int rk = arg.rk;
    int grid_size = arg.grid_size;
    int block_size = arg.block_size;
    cudaError_t err;
    err = cudaMallocManaged(A, M * K * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for A: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMallocManaged(B, K * N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for B: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMallocManaged(C, M * N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed for C: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    for(int i = 0; i < M * K; i++){
        (*A)[i] = i;
    }
    for(int i = 0; i < K * N; i++){
        (*B)[i] = i;
    }
    for(int i = 0; i < M * N; i++){
        (*C)[i] = 0;
    }
    std::cout<<"matrix intialized!"<<std::endl;
}

void origin_gemm(args arg ,int *A, int *B, int *C){
    origin_gemm_thread<<<1, 16>>>();
    cudaDeviceSynchronize();
}

int main(){
    args arg;
    int *A, *B, *C;
    init_matrix(arg, &A, &B, &C);
    origin_gemm(arg, A, B, C);
     

    // 调用内核函数
    

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}