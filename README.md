# 记录本人的cuda加速实现
在一次面试后，深感自己对cuda的理解和使用还是不够深入，故开此新坑 ----20241218

## 使用方法
```bash
curl -fsSL https://xmake.io/shget.text | bash

git clone https://github.com/zhaosiyuan1098/my_cuda_speedup_solutions.git

cd ./my_cuda_speedup_solutions

xmake 
```

### gemm
* v1:最原始的实现
* v2:使用分块矩阵+共享内存
* v3:减少寄存器使用
* v4:对B矩阵转置（**效果较差**，虽然避免了bankconflict，但改变了原有的内存访问顺序，可能导致内存访问不连续+的转置访问模式可能导致缓存命中率降低+内存访问模式不再符合 GPU 的合并访问模式）
* v5:使用padding避免bank conflict


## todolist

### gemm
    
* 使用寄存器加速

### reduce
* 原始实现
* 避免分支分化
* 避免bank conflict

### softmax