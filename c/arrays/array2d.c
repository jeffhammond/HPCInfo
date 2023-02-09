#include <stdio.h>
#include <stdlib.h>

void print2x(int m, int n, const double A[restrict m][n])
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            printf("*(%d,%d)=%lf\n",i,j,A[i][j]);
}

void print2(int m, int n, const double (* const restrict A)[n])
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            printf("(%d,%d)=%lf\n",i,j,A[i][j]);
}

void print1(int m, int n, const double A[restrict])
{
    for(int i=0; i<(m*n); i++)
        printf("(%d)=%lf\n",i,A[i]);
}

int main(int argc, char* argv[])
{
    int m = (argc>1) ? atoi(argv[1]) : 10;
    int n = (argc>2) ? atoi(argv[2]) : 5;

    double * restrict A = (double*)malloc(m*n*sizeof(double));
    for(int i=0; i<(m*n); i++)
        A[i] = (double)i;

    double (* const restrict B)[n] = (double (*)[n]) malloc(m*n*sizeof(double));
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            B[i][j] = (double)(i*n+j);

    double (*C)[m][n] = malloc(m*n*sizeof(double));
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            (*C)[i][j] = (double)(i*n+j);

    printf("A\n");
    print1(m,n,A);
    print2(m,n,(double (*)[n])A);

    printf("B\n");
    print1(m,n,(double * restrict)B);
    print2(m,n,B);

    printf("C\n");
    print1(m,n,(double * restrict)C);
    print2(m,n,(double (*)[n])C);
    print2x(m,n,(double (*)[n])C);

    free(C);
    free(B);
    free(A);

    return 0;
}
