#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_ATOMICS__)

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <stdatomic.h>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

int main(int argc, char * argv[])
{
    /// ensure we use an even number of threads
    {
        int nt = omp_get_max_threads();
        if (nt == 1) {
            printf("You must use more than %d threads\n", nt);
            abort();
        }
        if (nt % 2 != 0) omp_set_num_threads(nt-1);
#if 1
        /// temporary limitation
        if (nt != 2) omp_set_num_threads(2);
#endif
    }

    int iterations = (argc>1) ? atoi(argv[1]) : 100;

    _Atomic int left_ready  = -1;
    _Atomic int right_ready = -1;

    int left_payload  = 0;
    int right_payload = 0;

    #pragma omp parallel
    {
        int me      = omp_get_thread_num();
        /// 0=left 1=right
        bool parity = (me % 2 == 0);

        int junk = 0;

        /// START TIME
        #pragma omp barrier
        double t0 = omp_get_wtime();

        for (int i=0; i<iterations; ++i) {

            if (parity) {

                /// send to left
                left_payload = i;
                atomic_store_explicit( &left_ready, i, memory_order_release);

                /// recv from right
                while (i != atomic_load_explicit( &right_ready, memory_order_acquire));
                //printf("%d: left received %d\n", i, right_payload);
                junk += right_payload;

            } else {

                /// recv from left
                while (i != atomic_load_explicit( &left_ready, memory_order_acquire));
                //printf("%d: right received %d\n", i, left_payload);
                junk += left_payload;

                ///send to right
                right_payload = i;
                atomic_store_explicit( &right_ready, i, memory_order_release);

            }

        }

        /// STOP TIME
        #pragma omp barrier
        double t1 = omp_get_wtime();

        /// PRINT TIME
        double dt = t1-t0;
        #pragma omp critical
        {
            printf("total time elapsed = %e\n", dt);
            printf("time per iteration = %e\n", dt/iterations);
            printf("%d\n", junk);
        }
    }

    return 0;
}

#else  // C11
#error You need C11 atomics for this test!
#endif // C11
