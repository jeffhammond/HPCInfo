#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <time.h>

static inline double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    time_t s  = ts.tv_sec;
    long   ns = ts.tv_nsec;
    double t  = (double)s + 1.e-9 * (double)ns;
    return t;
}

#include <cuda.h>
#include <cuda_runtime_api.h>

int main(void)
{
    struct cudaPointerAttributes a;

    // warmup
    for (int i=0; i < 100000; i++) {
        cudaError_t e;
        void * p;
        e = cudaMalloc(&p, i);
        e = cudaPointerGetAttributes(&a, p);
    }

    size_t n = 0;
    const double t0 = wtime();
    for (intptr_t i=0; i < INT_MAX; i += 4096) {
        int data;
        CUresult r = cuPointerGetAttribute(&data, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, i);
        n++;
    }
    const double t1 = wtime();
    const double dt = t1 - t0;
    printf("time for %zu calls = %lf, per call = %lf ns\n", n, dt, 1.0e9*dt/(double)n);
    return 0;
}
