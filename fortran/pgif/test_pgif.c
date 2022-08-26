#include <stdio.h>
#include <stdlib.h>

#include "pgif90.h"

#define M 7
#define N 3

extern void foo(double *, int, int);
extern void bar(double *, int, int, F90_Desc_la *);

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

    F90_Desc_la d = {0};
    d.tag   = 35;         // it's always 35
    d.rank  =  2;         // matrix
    d.kind  = 28;         // double
    d.len   = sizeof(*y); // element size
    d.flags = 0x20000000; // sequential
    d.lsize = M*N;        // total size
    d.gsize = M*N;        // total size
    d.lbase = -N;         // need to understand better to do rank>2 correctly
    d.gbase = NULL;       // always (nul)?

    d.dim[0].lbound  = 1;
    d.dim[0].extent  = N;
    d.dim[0].sstride = 0;
    d.dim[0].soffset = 0;
    d.dim[0].lstride = 1;
    d.dim[0].ubound  = 0;

    // if lbase is 0, lbound below has to be 0
    d.dim[1].lbound  = 1; // correlated to lbase
    d.dim[1].extent  = M;
    d.dim[1].sstride = 0;
    d.dim[1].soffset = 0;
    d.dim[1].lstride = N;
    d.dim[1].ubound  = 0;

    bar(y,N,M,&d);

    free(y);
    return 0;
}
