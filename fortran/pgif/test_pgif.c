#include <stdio.h>
#include <stdlib.h>

#include "pgif90.h"

#define M 3
#define N 4

extern void foo(double *, int, int);
extern void bar(double *, int, int, F90_Desc *);

int main(void)
{
    double x[M][N];
    double * y = malloc(M*N*sizeof(*y));

    int z = 0;
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            x[i][j] = y[i*N+j] = z++;
        }
    }

    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            printf("C: %d,%d,%f,%f\n",i,j,x[i][j],y[i*N+j]);
        }
    }

    foo(y,N,M);
    bar(y,N,M,NULL);

    free(y);
    return 0;
}
