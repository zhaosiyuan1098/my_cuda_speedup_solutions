#!/bin/bash

# 定义矩阵大小数组
matrix_sizes=(32 64 100 128 256 500 512 1024 1500 2048 2560 3000 3479 4096 4667 5120 6400)

# 清空benchmark文件
> result.txt

# 遍历每个矩阵大小
for size in ${matrix_sizes[@]}; do
    # 遍历每个方法（0-4）
    for method in $(seq 1 5); do
        echo "Running matrix multiplication for size $size with method $method" >> result.txt
        xmake run my_cuda_speedup_solutions $size $method >> result.txt
    done
done
