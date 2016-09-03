#if defined(__GNUC__) && (__GNUC__ <= 6)
#error GCC will not compile this code because of "https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65467"
#else

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

#ifdef SEQUENTIAL_CONSISTENCY
int load_model  = memory_order_seq_cst;
int store_model = memory_order_seq_cst;
#else
int load_model  = memory_order_acquire;
int store_model = memory_order_release;
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
                atomic_store_explicit( &left_ready, i, store_model);

                /// recv from right
                while (i != atomic_load_explicit( &right_ready, load_model));
                //printf("%d: left received %d\n", i, right_payload);
                junk += right_payload;

            } else {

                /// recv from left
                while (i != atomic_load_explicit( &left_ready, load_model));
                //printf("%d: right received %d\n", i, left_payload);
                junk += left_payload;

                ///send to right
                right_payload = i;
                atomic_store_explicit( &right_ready, i, store_model);

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

#else  // C11
#error You need C11 atomics for this test!
#endif // C11

#endif // GCC <= 6
