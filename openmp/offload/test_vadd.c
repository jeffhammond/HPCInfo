#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

//#include <cilk/cilk.h>
//#include <omp.h>

void vadd1(int n, float * restrict a, float * restrict b, float * restrict c)
{
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd2(int n, float * restrict a, float * restrict b, float * restrict c)
{
    _Cilk_for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd3(int n, float * restrict a, float * restrict b, float * restrict c)
{
#pragma offload target(gfx) pin(a, b, c : length(n))
    _Cilk_for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd4(int n, float * restrict a, float * restrict b, float * restrict c)
{
#pragma omp target map(to:n,a[0:n],b[0:n]) map(from:c[0:n])
#pragma omp parallel for simd
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main(int argc, char * argv[])
{
    int n = (argc > 1 ) ? atoi(argv[1]) : 1000;
    float * x  = malloc(n*sizeof(float)); assert(x !=NULL);
    float * y  = malloc(n*sizeof(float)); assert(y !=NULL);
    float * z1 = malloc(n*sizeof(float)); assert(z1!=NULL);
    float * z2 = malloc(n*sizeof(float)); assert(z2!=NULL);
    float * z3 = malloc(n*sizeof(float)); assert(z3!=NULL);
    float * z4 = malloc(n*sizeof(float)); assert(z4!=NULL);

    for (int i=0; i<n; i++) {
        x[i] = (float)i;
    }

    for (int i=0; i<n; i++) {
        y[i] = (float)(-i);
    }

    vadd1(n,x,y,z1);
    vadd2(n,x,y,z2);
    //vadd3(n,x,y,z3);
    vadd4(n,x,y,z4);

#if 0
    for (int i=0; i<n; i++) {
        printf("y[%d] = %lf\n", i, y[i]);
    }
#endif

    free(z4);
    free(z3);
    free(z2);
    free(z1);
    free(y);
    free(x);
    return 0;
}
