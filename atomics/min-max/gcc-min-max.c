#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <omp.h>

#define ITERATIONS 10000
#define ELEMENTS 1024

#define OMP_MIN(x,y) (x<y)?x:y
#define OMP_MAX(x,y) (x>y)?x:y

static inline void atomic_min_i32(int32_t * target, int32_t value)
{
  int32_t desired;

  int32_t old = __atomic_load_n(target,__ATOMIC_SEQ_CST);

  // early exit when no update required
  if (old < value) return;

  do {
    old = __atomic_load_n(target,__ATOMIC_SEQ_CST);
    desired = OMP_MIN(old, value);
  } while (!__atomic_compare_exchange_n(target, &old, desired, false /* strong */, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) );
}

static inline void atomic_max_i32(int32_t * target, int32_t value)
{
  int32_t desired;

  int32_t old = __atomic_load_n(target,__ATOMIC_SEQ_CST);

  // early exit when no update required
  if (old > value) return;

  do {
    old = __atomic_load_n(target,__ATOMIC_SEQ_CST);
    desired = OMP_MAX(old, value);
  } while (!__atomic_compare_exchange_n(target, &old, desired, false /* strong */, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) );
}

int main(int argc, char * argv[])
{
    int32_t i32[ELEMENTS];
    int64_t i64[ELEMENTS];
    float   r32[ELEMENTS];
    double  r64[ELEMENTS];

    int32_t i32_min =  100000;
    int64_t i64_min =  100000;
    float   r32_min =  100000;
    double  r64_min =  100000;

    int32_t i32_max = -100000;
    int64_t i64_max = -100000;
    float   r32_max = -100000;
    double  r64_max = -100000;

    double  t0, t1, dt;

    for (int j=0; j<ELEMENTS; j++) {
      i32[j] = j+1;
      i64[j] = j+1LL;
      r32[j] = j+1.0f;
      r64[j] = j+1.0;
    }

    #pragma omp parallel
    {
        // MIN

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { atomic_min_i32(&i32_min,i32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { i32_min = OMP_MIN(i32_min,i32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10d took %12.7f seconds\n","i32","min",i32_min,dt);
        }

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { i64_min = OMP_MIN(i64_min,i64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { i64_min = OMP_MIN(i64_min,i64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10lld took %12.7f seconds\n","i64","min",i64_min,dt);
        }

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r32_min = OMP_MIN(r32_min,r32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r32_min = OMP_MIN(r32_min,r32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10.3f took %12.7f seconds\n","r32","min",r32_min,dt);
        }

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r64_min = OMP_MIN(r64_min,r64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r64_min = OMP_MIN(r64_min,r64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10.3f took %12.7f seconds\n","r64","min",r64_min,dt);
        }


        // MAX

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { atomic_max_i32(&i32_max,i32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { i32_max = OMP_MAX(i32_max,i32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10d took %12.7f seconds\n","i32","max",i32_max,dt);
        }

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { i64_max = OMP_MAX(i64_max,i64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { i64_max = OMP_MAX(i64_max,i64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10lld took %12.7f seconds\n","i64","max",i64_max,dt);
        }

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r32_max = OMP_MAX(r32_max,r32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r32_max = OMP_MAX(r32_max,r32[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10.3f took %12.7f seconds\n","r32","max",r32_max,dt);
        }

        // warmup
        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r64_max = OMP_MAX(r64_max,r64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t0 = omp_get_wtime();
        }

        for (int i=0; i<ITERATIONS; i++) {
            for (int j=0; j<ELEMENTS; j++) {
                #pragma omp critical
                { r64_max = OMP_MAX(r64_max,r64[j]); }
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            t1 = omp_get_wtime();
            dt = t1 - t0;
            printf("%3s: %3s=%10.3f took %12.7f seconds\n","r64","max",r64_max,dt);
        }



    }

    return 0;
}
