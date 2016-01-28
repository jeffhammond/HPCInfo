#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h> /* PRIu64 */
#include <assert.h>

#if 0
# include <pthread.h>
# include <unistd.h> /* sleep() */
#else
# ifdef _OPENMP
#  include <omp.h>
# else
#  error No OpenMP support!
# endif
#endif

#if 0
# ifdef __STDC_NO_ATOMICS__
#  error You need a real C11 compiler!
# else
#  include <stdatomic.h>
# endif
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

#if 0
static int num_threads;
static pthread_t * pool;
static pthread_barrier_t barrier;
#endif

#if 0
static _Atomic uint_fast64_t counter;
#else
static uint64_t counter;
#endif

static const int debug = 1;

static int get_thread_id(void)
{
#if 0
    for (int i=0; i<num_threads; i++)
        if (pthread_self()==pool[i])
            return i;

    return -1;
#else
    return omp_get_thread_num();
#endif
}

static void * fight(void * input)
{
    int tid = get_thread_id();
#if 0
    int rc;

    if (debug) {
        printf("%d: before pthread_barrier_wait 1 \n", tid);
    }
    rc = pthread_barrier_wait(&barrier);
    assert(rc==0);
    if (debug) {
        printf("%d: after  pthread_barrier_wait 1 \n", tid);
        fflush(stdout);
    }
#else
    #pragma omp barrier
#endif

    const int count = 1000000;

    uint64_t rval;

    uint64_t t0 = GetTimeBase();
    for (int i=0; i<count; i++) {
#if 0
        rval = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
#else
        #pragma omp atomic capture
        rval = counter++;
#endif
    }
    uint64_t t1 = GetTimeBase();

#if 0
    if (debug) {
        printf("%d: before pthread_barrier_wait 2 \n", tid);
    }
    rc = pthread_barrier_wait(&barrier);
    assert(rc==0);
    if (debug) {
        printf("%d: after  pthread_barrier_wait 2 \n", tid);
        fflush(stdout);
    }
#else
    #pragma omp barrier
#endif

    int64_t dt = t1-t0;
    printf("%2d: %d calls to %s took %" PRIu64 " cycles per call (rval=%" PRIu64 ")\n",
           tid, count,
#if 0
           "C11 atomic_fetch_add_explicit(memory_order_relaxed)",
#else
           "OpenMP atomic capture fetch-and-inc",
#endif
           dt/count, rval);
    fflush(stdout);

#if 0
    pthread_exit(NULL);
#endif

    return NULL;
}

int main(int argc, char * argv[])
{
#if 0
    int rc;
    num_threads = (argc>1) ? atoi(argv[1]) : 1;
#else
    int num_threads = omp_get_max_threads();
#endif
    printf("C11 counter test using %d threads \n", num_threads );

#if 0
    rc = pthread_barrier_init(&barrier, NULL, num_threads);
    assert(rc==0);

    pool = (pthread_t *) malloc( num_threads * sizeof(pthread_t) );
    assert(pool!=NULL);

    for (int i=0; i<num_threads; i++) {
        int rc = pthread_create(&(pool[i]), NULL, &fight, NULL);
        if (rc!=0) {
            printf("pthread error \n");
            fflush(stdout);
            sleep(1);
        }
        assert(rc==0);
    }

    if (debug) {
        printf("threads created \n");
        fflush(stdout);
    }

    for (int i=0; i<num_threads; i++) {
        void * junk;
        int rc = pthread_join(pool[i], &junk);
        if (rc!=0) {
            printf("pthread error \n");
            fflush(stdout);
            sleep(1);
        }
        assert(rc==0);
    }

    if (debug) {
        printf("threads joined \n");
        fflush(stdout);
    }
    uint64_t rval = atomic_load_explicit(&counter, memory_order_seq_cst);
#else
    #pragma omp parallel
    {
        fight(NULL);
    }
    uint64_t rval = counter;
#endif


    printf("final value of counter is %" PRIu64 " \n", rval);

#if 0
    free(pool);

    rc = pthread_barrier_destroy(&barrier);
    assert(rc==0);
#endif

    return 0;
}
