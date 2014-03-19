#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    int n = (argc>1 ? atoi(argv[1]) : 400);

    double * A = malloc(n*n*sizeof(double));
    double * B = malloc(n*n*sizeof(double));
    double * C = malloc(n*n*sizeof(double));
    assert(A!=NULL);
    assert(B!=NULL);
    assert(C!=NULL);

    double t0, t1;

#pragma omp parallel
{
    #pragma omp parallel for
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            A[i*n+j] = 1.0/(i+j+1);

    #pragma omp parallel for
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            B[i*n+j] = 1.0/(i+j+1);

    #pragma omp parallel for
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            C[i*n+j] = 0.0;

    t0 = omp_get_wtime();
    #pragma omp parallel for
    for (int k=0; k<n; k++)
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
                C[i*n+j] += A[i*n+k] * B[k*n+j];

    t1 = omp_get_wtime();
}

    double x = 0.0;
    #pragma omp parallel for
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
        {
             //printf("C(%d,%d) = %lf \n", i, j, C[i*n+j]);
             x += C[i*n+j];
        }

    double dt = t1-t0;
    printf("x = %lf dt = %lf \n", x, dt);

    return 0;
}

