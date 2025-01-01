#include "args.h"
#include <iostream>

args::args(int M, int K, int N, int bk, int rk) : M(M), K(K), N(N), bk(bk), rk(rk)
{
    grid_size = (M + bk - 1) / bk;
    block_size = (K + rk - 1) / rk;
    std::cout << "grid_size: " << grid_size << std::endl;
    std::cout << "block_size: " << block_size << std::endl;
}