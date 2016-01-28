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

static uint64_t counter;

static const int debug = 1;

static void * fight(void * input)
{
    int tid = omp_get_thread_num();

    #pragma omp barrier

    const int count = 1000000;

    uint64_t rval;

    uint64_t t0 = GetTimeBase();
    for (int i=0; i<count; i++) {
#if 0
        #pragma omp atomic capture
        rval = counter++;
#else
        rval = __sync_fetch_and_add(&counter,1);
#endif
    }
    uint64_t t1 = GetTimeBase();

    #pragma omp barrier

    int64_t dt = t1-t0;
    printf("%2d: %d calls to %s took %" PRIu64 " cycles per call (rval=%" PRIu64 ")\n",
           tid, count,
           "OpenMP atomic capture fetch-and-inc",
           dt/count, rval);
    fflush(stdout);

    return NULL;
}

int main(int argc, char * argv[])
{
    printf("OpenMP counter test using %d threads \n", omp_get_max_threads() );

    #pragma omp parallel
    {
        fight(NULL);
    }
    uint64_t rval = counter;

    printf("final value of counter is %" PRIu64 " \n", rval);

    return 0;
}
