#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cilk/cilk.h>

int main(int argc, char * argv[])
{
    int n = (argc > 1 ) ? atoi(argv[1]) : 1000;
    double * x = malloc(n*sizeof(double)); assert(x!=NULL);
    double * y = malloc(n*sizeof(double)); assert(y!=NULL);

    for (int i=0; i<n; i++) {
        x[i] = (double)i;
    }

    for (int i=0; i<n; i++) {
        y[i] = (double)(-i);
    }

    cilk_for (int i=0; i<n; i++) {
        y[i] += x[i];
    }

    for (int i=0; i<n; i++) {
        printf("y[%d] = %lf\n", i, y[i]);
    }

    free(y);
    free(x);
    return 0;
}
