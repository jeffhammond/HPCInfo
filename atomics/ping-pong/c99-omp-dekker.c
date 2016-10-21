#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

#if ( _OPENMP < 201307 )
# error You need OpenMP 4+ for seq_cst atomics.
#else
# define OMP_ATOMIC_LOAD_SC  _Pragma("omp atomic read seq_cst")
# define OMP_ATOMIC_STORE_SC _Pragma("omp atomic write seq_cst")
# define OMP_ATOMIC_LOAD     _Pragma("omp atomic read")
# define OMP_ATOMIC_STORE    _Pragma("omp atomic write")
#endif

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
    if (nt != 2) omp_set_num_threads(2);

    int iterations = (argc>1) ? atoi(argv[1]) : 1000000;

    printf("Dekker benchmark\n");
    printf("num threads  = %d\n", omp_get_max_threads());
    printf("iterations   = %d\n", iterations);
    fflush(stdout);

    int x=0, y=0;
    int lcount=0;
    int rcount=0;

    #pragma omp parallel
    {
        int me = omp_get_thread_num();
        int temp = -1;

        /// START TIME
        #pragma omp barrier
        double t0 = omp_get_wtime();

        for (int i=0; i<iterations; ++i) {
            if (me==0) {
                OMP_ATOMIC_STORE
                x=1;
                OMP_ATOMIC_LOAD_SC
                temp=y;
                if (temp==0) {
                    lcount++;
                    temp=0;
                }
            } else {
                OMP_ATOMIC_STORE
                y=1;
                OMP_ATOMIC_LOAD_SC
                temp=x;
                if (temp==0) {
                    rcount++;
                    temp=0;
                }
            }
            /// reset flags
            #pragma omp single
            {
                x=0;
                y=0;
            }
        }

        /// STOP TIME
        #pragma omp barrier
        double t1 = omp_get_wtime();

        /// PRINT TIME
        double dt = t1-t0;
        #pragma omp critical
        {
            printf("total time elapsed = %lf\n", dt);
            printf("time per iteration = %e\n", dt/iterations);
            printf("lcount=%d, rcount=%d\n", lcount, rcount);
        }
    }

    return 0;
}
