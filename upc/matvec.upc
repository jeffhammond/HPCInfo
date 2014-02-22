#include <upc_relaxed.h>

#define N 1000

shared double A[N][N], X[N], Y[N];

int main(int argc, char** argv)
{
    //int i,j;
    for(int i=0; i<N; i++)
        if (i%THREADS==MYTHREAD)
            for (int j=0; j<N; j++)
                Y[i] += A[i][j] * X[j];

    return 0;
}
