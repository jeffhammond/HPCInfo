#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <omp.h>

#define N 100*1000*1000

int main(void)
{
    int n = N;
    double y[N], x;

    #pragma omp target teams distribute parallel for
    for (int i=0; i<n; ++i) {
        y[i] = i;
    }

    #pragma omp target teams distribute parallel for reduction(+:x)
    for (int i=0; i<n; ++i) {
        x += y[i]
    }
    printf("y=%f\n",y);
    unsigned long long r = n;
    r *= (n-1);
    r /= 2;
    printf("r=%lld\n",r);

    return 0;
}
