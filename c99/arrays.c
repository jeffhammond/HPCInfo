#include <stdio.h>
#include <stdlib.h>

/* Both of these signatures are acceptable.
 * The first one looks more like the cast below.
 * The second format does not work as a cast below. */
#if 1
void print2(int m, int n, double (* const restrict A)[n])
#else
void print2(int m, int n, double (A[restrict])[n])
#endif
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            printf("(%d,%d)=%lf\n",i,j,A[i][j]);

}

void print1(int m, int n, double A[])
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

    print1(m,n,A);
    print2(m,n,(double (*)[n])A);

    print1(m,n,(double * restrict)B);
    print2(m,n,B);

    free(B);
    free(A);

    return 0;
}
