#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

#ifdef SEQUENTIAL_CONSISTENCY
#define OMP_ATOMIC_LOAD  _Pragma("omp atomic seq_cst")
#define OMP_ATOMIC_STORE _Pragma("omp atomic seq_cst")
#else
#define OMP_ATOMIC_LOAD  _Pragma("omp atomic read")
#define OMP_ATOMIC_STORE _Pragma("omp atomic write")
#endif
#define OMP_FLUSH _Pragma("omp flush")

int main(int argc, char * argv[])
{
    int nt = omp_get_max_threads();
#if 1
    if (nt != 2) omp_set_num_threads(2);
#else
    if (nt < 2)      omp_set_num_threads(2);
    if (nt % 2 != 0) omp_set_num_threads(nt-1);
#endif

    int iterations = (argc>1) ? atoi(argv[1]) : 100;

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
                OMP_ATOMIC_STORE
                left_ready = i;

                /// recv from right
                while (1) {
                    OMP_FLUSH
                    int temp;
                    OMP_ATOMIC_LOAD
                    temp = i;
                    if (temp == right_ready) break;
                }
                //printf("%d: left received %d\n", i, right_payload);
                junk += right_payload;

            } else {

                /// recv from left
                while (1) {
                    OMP_FLUSH
                    int temp;
                    OMP_ATOMIC_LOAD
                    temp = i;
                    if (temp == left_ready) break;
                }
                //printf("%d: right received %d\n", i, left_payload);
                junk += left_payload;

                ///send to right
                right_payload = i;
                OMP_ATOMIC_STORE
                right_ready = i;

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
