#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

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

/*
__host__ Error_t cudaPointerGetAttributes ( cudaPointerAttributes* attributes, const void* ptr )
Returns attributes about a specified pointer.
Parameters
attributes
- Attributes for the specified pointer
ptr
- Pointer to get attributes for
*/

int main(void)
{
    struct cudaPointerAttributes a;

    size_t n = 0;
    const double t0 = wtime();
    for (intptr_t i=0; i < INT_MAX; i += 4096) {
        cudaError_t e = cudaPointerGetAttributes(&a, (void*)i);
        //printf("type=%d device=%d device pointer=%p host pointer=%p\n",
        //        a.type, a.device, a.devicePointer, a.hostPointer);
        n++;
    }
    const double t1 = wtime();
    const double dt = t1 - t0;
    printf("time for %zu calls = %lf, per call = %lf ns\n", n, dt, 1.0e9*dt/(double)n);
    return 0;
}
