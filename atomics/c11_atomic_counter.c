#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __STDC_NO_ATOMICS__
#error You need a real C11 compiler!
#else
#include <stdatomic.h>
#endif

/* if this does not work, use Kaz's getticks() */
#include "immintrin.h"
static inline int64_t GetTimeBase(void) {
    return __int64 _rdtsc(void);
}

static int num_threads;
static pthread_t * pool;

static pthread_barrier_t barrier;
static _Atomic uint_fast64_t counter;
static uint64_t slowcounter;

static const int debug = 1;

static int get_thread_id(void)
{
    for (int i=0; i<num_threads; i++)
        if (pthread_self()==pool[i])
            return i;

    return -1;
}

static void * fight(void * input)
{
    int rc;
    int tid = get_thread_id();

    if (debug) {
        printf("%d: before pthread_barrier 1 \n", tid);
    }
    rc = pthread_barrier(&barrier);
    assert(rc==0);
    if (debug) {
        printf("%d: after  pthread_barrier 1 \n", tid);
        fflush(stdout);
    }

    int count = 1000000;

    uint64_t rval;

    int64_t t0 = GetTimeBase();
    for (int i=0; i<count; i++) {
        rval = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
    }
    int64_t t1 = GetTimeBase();

    if (debug) {
        printf("%d: before pthread_barrier 2 \n", tid);
    }
    rc = pthread_barrier(&barrier);
    assert(rc==0);
    if (debug) {
        printf("%d: after  pthread_barrier 2 \n", tid);
        fflush(stdout);
    }

    int64_t dt = t1-t0;
    printf("%2d: %d calls to %s took %llu cycles per call \n", tid, count,
           "C11 atomic_fetch_add_explicit(memory_order_relaxed)", dt/count);
    fflush(stdout);

    pthread_exit(NULL);

    return NULL;
}

int main(int argc, char * argv[])
{
    int rc;

    num_threads = (argc>1) ? atoi(argv[1]) : 1;
    printf("C11 counter test using %d threads \n", num_threads );

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

    uint64_t atomic_load_explicit(&counter, memory_order_seq_cst);

    printf("final value of counter is %llu \n", rval);

    free(pool);

    rc = pthread_barrier_destroy(&barrier);
    assert(rc==0);

    return 0;
}
