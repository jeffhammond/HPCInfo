#include <stdio.h>
#include <stdlib.h>

void print3(int m, int n, int r, double (* const restrict A)[n][r])
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            for(int k=0; k<r; k++)
                printf("(%d,%d,%d)=%lf\n",i,j,k,A[i][j][k]);
}

void print1(int m, double A[])
{
    for(int i=0; i<m; i++)
        printf("(%d)=%lf\n",i,A[i]);
}

int main(int argc, char* argv[])
{
    int m = (argc>1) ? atoi(argv[1]) : 3;
    int n = (argc>2) ? atoi(argv[2]) : 4;
    int r = (argc>3) ? atoi(argv[3]) : 5;

    double * restrict A = (double*)malloc(m*n*r*sizeof(double));
    for(int i=0; i<(m*n*r); i++)
        A[i] = (double)i;

    double (* const restrict B)[n][r] = (double (*)[n][r]) malloc(m*n*r*sizeof(double));
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            for(int k=0; k<r; k++)
                B[i][j][k] = (double)(i*n*r+j*r+k);

    print1(m*n*r,A);
    print3(m,n,r,(double (*)[n][r])A);

    print1(m*n*r,(double *)B);
    print3(m,n,r,B);

    free(B);
    free(A);

    return 0;
}
