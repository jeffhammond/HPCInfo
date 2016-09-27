#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include <hwi/include/common/bgq_alignment.h>
#include <hwi/include/bqc/A2_inlines.h>
#include <spi/include/kernel/memory.h>
#include <spi/include/l2/barrier.h>
#include <spi/include/l2/atomic.h>

typedef struct BGQ_Atomic64_s
{
    volatile uint64_t atom;
}
ALIGN_L1D_CACHE BGQ_Atomic64_t;

/* TODO: test all of these functions
   uint64_t L2_AtomicLoad(volatile uint64_t *ptr)
   uint64_t L2_AtomicLoadClear(volatile uint64_t *ptr)
   uint64_t L2_AtomicLoadIncrement(volatile uint64_t *ptr)
   uint64_t L2_AtomicLoadDecrement(volatile uint64_t *ptr)
   uint64_t L2_AtomicLoadIncrementBounded(volatile uint64_t *ptr)
   uint64_t L2_AtomicLoadDecrementBounded(volatile uint64_t *ptr)
   uint64_t L2_AtomicLoadIncrementIfEqual(volatile uint64_t *ptr)
   void     L2_AtomicStore(volatile uint64_t *ptr, uint64_t value)
   void     L2_AtomicStoreTwin(volatile uint64_t *ptr, uint64_t value)
   void     L2_AtomicStoreAdd(volatile uint64_t *ptr, uint64_t value)
   void     L2_AtomicStoreAddCoherenceOnZero(volatile uint64_t *ptr,
   void     L2_AtomicStoreOr(volatile uint64_t *ptr, uint64_t value)
   void     L2_AtomicStoreXor(volatile uint64_t *ptr, uint64_t value)
   void     L2_AtomicStoreMax(volatile uint64_t *ptr, uint64_t value)
   void     L2_AtomicStoreMaxSignValue(volatile uint64_t *ptr,
*/

int num_threads;
pthread_t * pool;

L2_Barrier_t barrier = L2_BARRIER_INITIALIZER;
BGQ_Atomic64_t counter;
BGQ_Atomic64_t slowcounter;

int debug = 0;

int get_thread_id(void)
{
    for (int i=0; i<num_threads; i++)
        if (pthread_self()==pool[i])
            return i;

    return -1;
}

void * slowfight(void * input)
{
    int tid = get_thread_id();

    if (debug) 
        printf("%d: before L2_Barrier 1 \n", tid);
    L2_Barrier(&barrier, num_threads);
    if (debug) {
        printf("%d: after  L2_Barrier 1 \n", tid);
        fflush(stdout);
    }

    int count = 1000000;

    uint64_t t0 = GetTimeBase();
    for (int i=0; i<count; i++) {
        volatile uint64_t rval = Fetch_and_Add(&(slowcounter.atom), 1);
    }
    uint64_t t1 = GetTimeBase();

    if (debug) 
        printf("%d: before L2_Barrier 2 \n", tid);
    L2_Barrier(&barrier, num_threads);
    if (debug) {
        printf("%d: after  L2_Barrier 2 \n", tid);
        fflush(stdout);
    }
    
    uint64_t dt = t1-t0;
    printf("%2d: %d calls to %s took %llu cycles per call \n", 
           tid, count, "Fetch_and_Add", dt/count);
    fflush(stdout);

    pthread_exit(NULL);

    return NULL;
}

void * fight(void * input)
{
    int tid = get_thread_id();

    if (debug) 
        printf("%d: before L2_Barrier 1 \n", tid);
    L2_Barrier(&barrier, num_threads);
    if (debug) {
        printf("%d: after  L2_Barrier 1 \n", tid);
        fflush(stdout);
    }

    int count = 1000000;

    uint64_t t0 = GetTimeBase();
    for (int i=0; i<count; i++) {
        volatile uint64_t rval = L2_AtomicLoadIncrement(&(counter.atom));
    }
    uint64_t t1 = GetTimeBase();

    if (debug) 
        printf("%d: before L2_Barrier 2 \n", tid);
    L2_Barrier(&barrier, num_threads);
    if (debug) {
        printf("%d: after  L2_Barrier 2 \n", tid);
        fflush(stdout);
    }
    
    uint64_t dt = t1-t0;
    printf("%2d: %d calls to %s took %llu cycles per call \n", 
           tid, count, "L2_AtomicLoadIncrement", dt/count);
    fflush(stdout);

    pthread_exit(NULL);

    return NULL;
}

int main(int argc, char * argv[])
{
    num_threads = (argc>1) ? atoi(argv[1]) : 1;
    printf("L2 counter test using %d threads \n", num_threads );

    //printf("sizeof(BGQ_Atomic64_t) = %zu \n", sizeof(BGQ_Atomic64_t) );

    /* this "activates" the L2 atomic data structures */
    uint64_t rc64 = Kernel_L2AtomicsAllocate(&counter, sizeof(BGQ_Atomic64_t) );
    assert(rc64==0);

    L2_AtomicStore(&(counter.atom), 0);
    out64_sync(&(counter.atom), 0);

    pool = (pthread_t *) malloc( num_threads * sizeof(pthread_t) );
    assert(pool!=NULL);

    /**************************************************/

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

    volatile uint64_t rval = L2_AtomicLoad(&(counter.atom));
    printf("final value of counter is %llu \n", rval);

    /**************************************************/

    for (int i=0; i<num_threads; i++) {
        int rc = pthread_create(&(pool[i]), NULL, &slowfight, NULL);
        if (rc!=0) {
            printf("pthread error \n");
            fflush(stdout);
            sleep(1);
        }
        assert(rc==0);
    }

    printf("threads created \n");
    fflush(stdout);

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
    
    printf("threads joined \n");
    fflush(stdout);

    rval = in64(&(slowcounter.atom));
    printf("final value of slowcounter is %llu \n", rval);

    /**************************************************/

    free(pool);
 
    return 0;   
}
