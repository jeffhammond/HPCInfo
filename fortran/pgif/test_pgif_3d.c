#include <stdio.h>
#include <stdlib.h>

#include "pgif90.h"

#define M 11
#define N 3
#define P 7

extern void foo(double *, int, int, int);
extern void bar(double *, int, int, int, F90_Desc_la *);

int main(void)
{
    double x[M][N][P];
    double * y = malloc(M*N*P*sizeof(*y));

    int z = 0;
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<P; k++) {
                x[i][j][k] = y[i*N*P+j*P+k] = z++;
            }   
        }
    }

    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<P; k++) {
                printf("C: %d,%d,%d,%f,%f\n",i,j,k,x[i][j][k],y[i*N*P+j*P+k]);
            }
        }
    }

    foo(y,P,N,M);

    F90_Desc_la d = {0};
    d.tag   = 35;         // it's always 35
    d.rank  =  3;         // matrix
    d.kind  = 28;         // double
    d.len   = sizeof(*y); // element size
    d.flags = 0x20000000; // sequential
    d.lsize = M*N*P;      // total size
    d.gsize = M*N*P;      // total size
    d.lbase = 0;          // set to 0 and lbound[>0] = 0 for simplicity
    d.gbase = NULL;       // always (nul)?

    // NOTE: lbound and ubound here have no impact
    //       on the result of lbound and ubound in
    //       Fortran

    d.dim[0].lbound  = 1;
    d.dim[0].extent  = P;
    //d.dim[0].sstride = 0;
    //d.dim[0].soffset = 0;
    d.dim[0].lstride = 1;
    //d.dim[0].ubound  = 0;

    // if lbase is 0, lbound below has to be 0
    d.dim[1].lbound  = 0; // correlated to lbase
    d.dim[1].extent  = N;
    //d.dim[1].sstride = 0;
    //d.dim[1].soffset = 0;
    d.dim[1].lstride = P;
    //d.dim[1].ubound  = 0;

    // if lbase is 0, lbound below has to be 0
    d.dim[2].lbound  = 0; // correlated to lbase
    d.dim[2].extent  = M;
    //d.dim[2].sstride = 0;
    //d.dim[2].soffset = 0;
    d.dim[2].lstride = N*P;
    //d.dim[2].ubound  = 0;

    bar(y,P,N,M,&d);

    free(y);
    return 0;
}
