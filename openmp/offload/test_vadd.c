#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

//#include <cilk/cilk.h>
#include <omp.h>

#define RESTRICT

double vdiff(int n, const float * RESTRICT a, const float * RESTRICT b)
{
    double d = 0.0;
    for(int i = 0; i < n; i++) {
        d += (a[i] - b[i]);
    }
    return d;
}

void vadd0(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd1(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
#if defined(_OPENMP) && (_OPENMP >= 201307)
    #pragma omp parallel for simd
#elif defined(_OPENMP)
    #warning No OpenMP simd support!
    #pragma omp parallel for
#else
    #warning No OpenMP support!
#endif
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd2(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
#ifdef __cilk
    _Cilk_for(int i = 0; i < n; i++)
#else
    #warning No Cilk support.  Using sequential for loop.
    for(int i = 0; i < n; i++)
#endif
        c[i] = a[i] + b[i];
}

void vadd3(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
#ifdef __cilk
    #if defined(__INTEL_COMPILER) && defined(__INTEL_OFFLOAD)
    #pragma offload target(gfx) in(a,b : length(n)) out(c : length(n)) //pin(a, b, c : length(n))
    #else
    #warning No Cilk offload support!
    #endif
    _Cilk_for(int i = 0; i < n; i++)
#else
    #warning No Cilk support.  Using sequential for loop.
    for(int i = 0; i < n; i++)
#endif
        c[i] = a[i] + b[i];
}

void vadd4(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
#if defined(_OPENMP) && (_OPENMP >= 201307)
    #pragma omp target map(to:n,a[0:n],b[0:n]) map(from:c[0:n])
    #pragma omp parallel for simd
#else
    #warning No OpenMP target/simd support!
    #pragma omp parallel for
#endif
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

#if USE_GFX

#include <gfx/gfx_rt.h>

__attribute__((target(gfx_kernel)))
void gfx_vadd5(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    _Cilk_for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

void vadd5(int n, float * RESTRICT a, float * RESTRICT b, float * RESTRICT c)
{
    int rc = 0;
    rc = _GFX_share(a, sizeof(float)*n);
    if (rc) printf("_GFX_share returned %#06x\n", -_GFX_get_last_error());
    rc = _GFX_share(b, sizeof(float)*n);
    if (rc) printf("_GFX_share returned %#06x\n", -_GFX_get_last_error());
    rc = _GFX_share(c, sizeof(float)*n);
    if (rc) printf("_GFX_share returned %#06x\n", -_GFX_get_last_error());
    GfxTaskId id = _GFX_offload("gfx_vadd5", a, b, c, n);
    rc = _GFX_wait(id,1e9);
    if (rc) printf("_GFX_wait returned %#06x\n", -_GFX_get_last_error());
    rc = _GFX_unshare(a);
    if (rc) printf("_GFX_unshare returned %#06x\n", -_GFX_get_last_error());
    rc = _GFX_unshare(b);
    if (rc) printf("_GFX_unshare returned %#06x\n", -_GFX_get_last_error());
    rc = _GFX_unshare(c);
    if (rc) printf("_GFX_unshare returned %#06x\n", -_GFX_get_last_error());
}

#endif /* USE_GFX */

int main(int argc, char * argv[])
{
    int n = (argc > 1 ) ? atoi(argv[1]) : 1000;
    float * x  = calloc(n,sizeof(float)); assert(x !=NULL);
    float * y  = calloc(n,sizeof(float)); assert(y !=NULL);
    float * z0 = calloc(n,sizeof(float)); assert(z0!=NULL);
    float * z1 = calloc(n,sizeof(float)); assert(z1!=NULL);
    float * z2 = calloc(n,sizeof(float)); assert(z2!=NULL);
    float * z3 = calloc(n,sizeof(float)); assert(z3!=NULL);
    float * z4 = calloc(n,sizeof(float)); assert(z4!=NULL);
#if USE_GFX
    float * z5 = calloc(n,sizeof(float)); assert(z5!=NULL);
#endif

    for (int i=0; i<n; i++) {
        x[i] = (float)i;
    }

    for (int i=0; i<n; i++) {
        y[i] = (float)i;
    }

    double t0 = omp_get_wtime();
    vadd0(n,x,y,z0);
    double t1 = omp_get_wtime();
    vadd1(n,x,y,z1);
    double t2 = omp_get_wtime();
    vadd2(n,x,y,z2);
    double t3 = omp_get_wtime();
    vadd3(n,x,y,z3);
    double t4 = omp_get_wtime();
    vadd4(n,x,y,z4);
    double t5 = omp_get_wtime();
#if USE_GFX
    vadd5(n,x,y,z5);
    double t6 = omp_get_wtime();
#endif
    printf("%20s time = %lf             \n", "for",                      t1-t0);
    printf("%20s time = %lf (error=%lf) \n", "OpenMP for",               t2-t1, vdiff(n,z0,z1));
    printf("%20s time = %lf (error=%lf) \n", "_Cilk_for",                t3-t2, vdiff(n,z0,z2));
    printf("%20s time = %lf (error=%lf) \n", "offload _Cilk_for",        t4-t3, vdiff(n,z0,z3));
    printf("%20s time = %lf (error=%lf) \n", "OpenMP offload for",       t5-t4, vdiff(n,z0,z4));
#if USE_GFX
    printf("%20s time = %lf (error=%lf) \n", "GFX RT offload _Cilk_for", t6-t5, vdiff(n,z0,z5));
#endif

#if USE_GFX
    for (int i=0; i<n; i++) {
        printf("%d z0=%f z5=%f\n", i, z0[i], z5[i]);
    }
#endif

    /* prevent compiler from optimizing away anything */
    double junk = 0.0;
    for (int i=0; i<n; i++) {
        junk += z0[i] + z1[i] + z2[i] + z3[i] + z4[i]; // + z5[i];
    }
    printf("junk=%lf\n", junk);

#if USE_GFX
    free(z5);
#endif
    free(z4);
    free(z3);
    free(z2);
    free(z1);
    free(z0);
    free(y);
    free(x);

    printf("Success\n");

    return 0;
}
