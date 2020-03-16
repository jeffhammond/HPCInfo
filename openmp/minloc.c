#include <stdio.h>
#include <stdlib.h>

#include <float.h>

#include <omp.h>

#define MIN(x,y) ((x)<(y) ? (x) : (y))

int main(int argc, char* argv[])
{
    size_t n = (argc > 1) ? atol(argv[1]) : 100;

    double * v = malloc(n * sizeof(double));

    double min = DBL_MAX;
    size_t loc = SIZE_MAX;

    #pragma omp parallel firstprivate(n) shared(v, min, loc)
    {
        #pragma omp for
        for (size_t i=0; i<n; ++i) {
        }


    }

    return 0;
}
