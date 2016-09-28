#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h> /* PRIu64 */
#include <assert.h>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

#if defined(__bgq__)
#include <hwi/include/common/bgq_alignment.h>
#include <hwi/include/bqc/A2_inlines.h>
#include <spi/include/kernel/memory.h>
#include <spi/include/l2/barrier.h>
#include <spi/include/l2/atomic.h>
#else
/* if this does not work, use Kaz's getticks() */
static inline uint64_t GetTimeBase(void)
{
#if defined(__x86_64__)
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
#else
#error GetTimeBase not available for your architecture.
#endif
}
#endif

int main(int argc, char * argv[])
{

    int count = (argc>1) ? atoi(argv[1]) : 1000000;

    printf("OpenMP counter test using %d threads \n", omp_get_max_threads() );

    uint64_t counter;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        uint64_t rval;

        #pragma omp barrier
        uint64_t t0 = GetTimeBase();
        for (int i=0; i<count; i++) {
            rval = __sync_fetch_and_add(&counter,1);
        }
        #pragma omp barrier
        uint64_t t1 = GetTimeBase();

        int64_t dt = t1-t0;
        printf("%2d: %d calls to %s took %" PRIu64 " cycles per call (rval=%" PRIu64 ")\n",
               tid, count,
               "__sync_fetch_and_add",
               dt/count, rval);
        fflush(stdout);
    }
    uint64_t rval = counter;

    printf("final value of counter is %" PRIu64 " \n", rval);

    return 0;
}
