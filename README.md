# 记录本人的cuda加速实现
在一次面试后，深感自己对cuda的理解和使用还是不够深入，故开此新坑 ----20241218

## 使用方法
```bash
curl -fsSL https://xmake.io/shget.text | bash

git clone https://github.com/zhaosiyuan1098/my_cuda_speedup_solutions.git

cd ./my_cuda_speedup_solutions

xmake 
```

## todolist

### gemm
    
* 使用共享内存加速
* 使用寄存器加速
* 避免bank confict

### reduce
* 原始实现
* 避免分支分化
* 避免bank conflict

### softmax