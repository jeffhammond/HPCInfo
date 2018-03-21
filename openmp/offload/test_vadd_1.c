#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if USE_GFX
#include <gfx/gfx_rt.h>
#endif

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
#if defined(_OPENMP) && (_OPENMP >= 201307)
    //#pragma omp target teams distribute map(to:n,a[0:n],b[0:n]) map(from:c[0:n])
    #pragma omp target map(to:n,a[0:n],b[0:n]) map(from:c[0:n])
    #pragma omp parallel for simd
#else
    #warning No OpenMP target/simd support!
    #pragma omp parallel for
#endif
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

int main(int argc, char * argv[])
{
    int n = (argc > 1 ) ? atoi(argv[1]) : 1000;
    float * x  = calloc(n,sizeof(float)); assert(x !=NULL);
    float * y  = calloc(n,sizeof(float)); assert(y !=NULL);
    float * z0 = calloc(n,sizeof(float)); assert(z0!=NULL);
    float * z1 = calloc(n,sizeof(float)); assert(z1!=NULL);
    float * z2 = calloc(n,sizeof(float)); assert(z2!=NULL);

    for (int i=0; i<n; i++) {
        y[i] = x[i] = (float)i;
    }

    for (int iter=0; iter<10; iter++) {
        double t0 = omp_get_wtime();
        vadd0(n,x,y,z0);
        double t1 = omp_get_wtime();
        vadd1(n,x,y,z1);
        double t2 = omp_get_wtime();
        vadd2(n,x,y,z2);
        double t3 = omp_get_wtime();
        printf("%20s time = %lf             \n", "for",                      t1-t0);
        printf("%20s time = %lf (error=%lf) \n", "OpenMP for",               t2-t1, vdiff(n,z0,z1));
        printf("%20s time = %lf (error=%lf) \n", "OpenMP offload for",       t3-t2, vdiff(n,z0,z2));

        /* prevent compiler from optimizing away anything */
        double junk = t0+t1+t2+t3;
        for (int i=0; i<n; i++) {
            junk += z0[i] + z1[i] + z2[i];
        }
        printf("junk=%lf\n", junk);
    }

    free(z2);
    free(z1);
    free(z0);
    free(y);
    free(x);

    printf("Success\n");

    return 0;
}
