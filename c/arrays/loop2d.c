#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

const int iter = 100;

int main(int argc, char * argv[])
{
    int n = (argc > 1 ? atoi(argv[1]) : 1000);

    double * x = malloc(n*n*sizeof(double));
    double * y = malloc(n*n*sizeof(double));

    size_t t = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            x[i*n+j] = t++;
            y[i*n+j] = 0;
        }
    }

    double t0=0, t1;
    double z;

    for (int k=-2; k<=iter; k++) {

        if (k==1) t0 = omp_get_wtime();

        z = 0;
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                //y[i*n+j] = x[j*n+i];
                //y[i*n+j] = x[i*n+j];
                z += i/exp(j);
            }
        }
    }
    t1 = omp_get_wtime(); 
    printf("2D dt=%lf\n",t1-t0);

    const int nn = n*n;
    for (int k=0; k<=iter; k++) {

        if (k==1) t0 = omp_get_wtime();

        z = 0;
        for (int ij=0; ij<nn; ij++) {
            int i = ij / n;
            int j = ij % n;
            //y[i*n+j] = x[j*n+i];
            //y[i*n+j] = x[i*n+j];
            z += i/exp(j);
        }
    }
    t1 = omp_get_wtime(); 
    printf("1D dt=%lf\n",t1-t0);

    return 0;
}
