#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

//#include <cilk/cilk.h>
//#include <omp.h>

#define RESTRICT

void vadd1(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd2(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    _Cilk_for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd3(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
#pragma offload target(gfx) in(a,b : length(n)) out(c : length(n)) //pin(a, b, c : length(n))
    _Cilk_for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd4(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
#pragma omp target map(to:n,a[0:n],b[0:n]) map(from:c[0:n])
#pragma omp parallel for simd
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

#include <gfx/gfx_rt.h>

//_declspec(target(gfx_kernel))
void gfx_vadd5(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    _Cilk_for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd5(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    _GFX_share(a, sizeof(float)*n);
    _GFX_share(b, sizeof(float)*n);
    _GFX_share(c, sizeof(float)*n);
    //_GFX_enqueue("gfx_vadd5", a, b, c, n);
    GfxTaskId id = _GFX_offload("gfx_vadd5", a, b, c, n);
    _GFX_wait(id,-1);
    _GFX_unshare(a);
    _GFX_unshare(b);
    _GFX_unshare(c);
}

int main(int argc, char * argv[])
{
    int n = (argc > 1 ) ? atoi(argv[1]) : 1000;
    float * x  = calloc(n,sizeof(float)); assert(x !=NULL);
    float * y  = calloc(n,sizeof(float)); assert(y !=NULL);
    float * z1 = calloc(n,sizeof(float)); assert(z1!=NULL);
    float * z2 = calloc(n,sizeof(float)); assert(z2!=NULL);
    float * z3 = calloc(n,sizeof(float)); assert(z3!=NULL);
    float * z4 = calloc(n,sizeof(float)); assert(z4!=NULL);
    float * z5 = calloc(n,sizeof(float)); assert(z5!=NULL);

    for (int i=0; i<n; i++) {
        x[i] = (float)i;
    }

    for (int i=0; i<n; i++) {
        y[i] = (float)(-i);
    }

    vadd1(n,x,y,z1);
    vadd2(n,x,y,z2);
    //vadd3(n,x,y,z3);
    //vadd4(n,x,y,z4);
    vadd5(n,x,y,z5);

#if 0
    for (int i=0; i<n; i++) {
        printf("y[%d] = %lf\n", i, y[i]);
    }
#endif

    free(z5);
    free(z4);
    free(z3);
    free(z2);
    free(z1);
    free(y);
    free(x);

    printf("Success\n");

    return 0;
}
