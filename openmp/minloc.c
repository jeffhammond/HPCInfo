#include <stdio.h>
#include <stdlib.h>

#include <float.h>

#include <omp.h>

int main(int argc, char* argv[])
{
    size_t n = (argc > 1) ? atol(argv[1]) : 100;

    double * v = malloc(n * sizeof(double));
    if (v==NULL) abort();

    // g = global
    double gmin = DBL_MAX;
    size_t gloc = SIZE_MAX;

    const int mt = omp_get_max_threads();

    printf("MINLOC of %zu elements with %d threads\n", n, mt);

    // t = thread
    double * tmin = malloc(mt * sizeof(double));
    size_t * tloc = malloc(mt * sizeof(size_t));
    if (tmin==NULL || tloc==NULL) abort();

    for (int i=0; i<mt; ++i) {
        tmin[i] = DBL_MAX;
        tloc[i] = SIZE_MAX;
    }

    double dt = 0.0;

    #pragma omp parallel firstprivate(n) shared(v, tmin, tloc, gmin, gloc, dt)
    {
        const int me = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        unsigned int seed = (unsigned int)me;
        srand(seed);
        #pragma omp for
        for (size_t i=0; i<n; ++i) {
            // this is not a _good_ random number generator, but it does not matter for this use case
            double r = (double)rand_r(&seed) / (double)RAND_MAX;
            v[i] = r;
        }

        double t0 = 0.0;

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        // thread-private result
        double mymin = DBL_MAX;
        double myloc = SIZE_MAX;

        #pragma omp for
        for (size_t i=0; i<n; ++i) {
            if (v[i] < mymin) {
                mymin = v[i];
                myloc = i;
            }
        }

        // write thread-private results to shared
        tmin[me] = mymin;
        tloc[me] = myloc;
        #pragma omp barrier

        // find global result
        #pragma omp master
        {
            for (int i=0; i<nt; ++i) {
                if (tmin[i] < gmin) {
                    gmin = tmin[i];
                    gloc = tloc[i];
                }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            double t1 = omp_get_wtime();
            dt = t1 - t0;
        }

#if 0
        #pragma omp critical
        {
            printf("%d: mymin=%f, myloc=%zu\n", me, mymin, myloc);
            fflush(stdout);
        }
#endif
    }

    printf("OpenMP: dt=%f, gmin=%f, gloc=%zu\n", dt, gmin, gloc);
    fflush(stdout);

    // sequential reference timing
    {
        double t0 = omp_get_wtime();

        double mymin = DBL_MAX;
        double myloc = SIZE_MAX;

        for (size_t i=0; i<n; ++i) {
            if (v[i] < mymin) {
                mymin = v[i];
                myloc = i;
            }
        }

        gmin = mymin;
        gloc = myloc;

        double t1 = omp_get_wtime();
        dt = t1 - t0;
    }

    printf("Sequential: dt=%f, gmin=%f, gloc=%zu\n", dt, gmin, gloc);
    fflush(stdout);

    // debug printing for toy inputs
    if (n<100) {
        for (size_t i=0; i<n; ++i) {
            printf("v[%zu]=%f\n", i , v[i]);
        }
        fflush(stdout);
    }

    free(v);

    printf("SUCCESS\n");

    return 0;
}
