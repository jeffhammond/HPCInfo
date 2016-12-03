/* Guessing what compilers are sufficient here... */
/* On Blue Gene/Q, we tried 4.7.2 and 4.4.7 only. */
#if (defined(__GNUC__) && (__GNUC__ >= 5)) || \
    (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ >= 7)) || \
    (defined(__clang__) && defined(__clang_major__) && (__clang_major__ >= 3)) || \
    (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1400)) || 1

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

#ifdef SEQUENTIAL_CONSISTENCY
int load_model  = __ATOMIC_SEQ_CST;
int store_model = __ATOMIC_SEQ_CST;
#else
int load_model  = __ATOMIC_ACQUIRE;
int store_model = __ATOMIC_RELEASE;
#endif

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
#if 1
    if (nt != 2) omp_set_num_threads(2);
#else
    if (nt < 2)      omp_set_num_threads(2);
    if (nt % 2 != 0) omp_set_num_threads(nt-1);
#endif

    int iterations = (argc>1) ? atoi(argv[1]) : 1000000;

    printf("thread ping-pong benchmark\n");
    printf("num threads  = %d\n", omp_get_max_threads());
    printf("iterations   = %d\n", iterations);
#ifdef SEQUENTIAL_CONSISTENCY
    printf("memory model = %s\n", "seq_cst");
#else
    printf("memory model = %s\n", "acq-rel");
#endif
    fflush(stdout);

    int left_ready  = -1;
    int right_ready = -1;

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
                __atomic_store_n( &left_ready, i, store_model);

                /// recv from right
                while (i != __atomic_load_n( &right_ready, load_model));
                //printf("%d: left received %d\n", i, right_payload);
                junk += right_payload;

            } else {

                /// recv from left
                while (i != __atomic_load_n( &left_ready, load_model));
                //printf("%d: right received %d\n", i, left_payload);
                junk += left_payload;

                ///send to right
                right_payload = i;
                __atomic_store_n( &right_ready, i, store_model);

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
            printf("%d\n", junk);
        }
    }

    return 0;
}

#else  // GCC 5+
#error Your compiler probably does not support __atomic functions.
#endif // GCC 5+
