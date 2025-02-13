#include <stdio.h>
#include <math.h>

void foo(int x[1024])
{
    #pragma omp simd simdlen(8)
    for (int i=0; i<1024; i++) {
        const int j = i % 4;
        switch (j) {
            case 0: x[i]++;            break;
            case 1: x[i]--;            break;
            case 2: x[i]/=2;           break;
            case 3: x[i]*=2;           break;
            case 4: x[i]=exp(x[i]);    break;
            case 5: x[i]=sin(x[i]);    break;
            case 6: x[i]=cos(x[i]);    break;
            case 7: x[i]=x[1024-i];    break;
        }
    }
}
