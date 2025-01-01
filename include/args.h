class args {
public:
    const int M;
    const int K;
    const int N;
    const int bk;
    const int rk;
    int grid_size;
    int block_size;
    args(int M=2048, int K=2048, int N=2048, int bk=128, int rk=8);
};