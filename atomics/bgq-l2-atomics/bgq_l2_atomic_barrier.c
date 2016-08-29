#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>

#include <spi/include/kernel/memory.h>
#include <spi/include/l2/barrier.h>

int num_threads;
pthread_t * pool;

static L2_Barrier_t barrier = L2_BARRIER_INITIALIZER;

int debug = 0;

int get_thread_id(void)
{
    for (int i=0; i<num_threads; i++)
        if (pthread_self()==pool[i])
            return i;

    return -1;
}

void * fight(void * input)
{
    int tid = get_thread_id();

    int count = 100000;

    if (debug) 
        printf("%d: before L2_Barrier \n", tid);

    uint64_t t0 = GetTimeBase();
    for (int i=0; i<count; i++)
        L2_Barrier(&barrier, num_threads);
    uint64_t t1 = GetTimeBase();

    if (debug) {
        printf("%d: after  L2_Barrier \n", tid);
        fflush(stdout);
    }

    uint64_t dt = t1-t0;
    printf("%2d: %d calls to %s took %llu cycles per call \n", 
           tid, count, "L2_Barrier", dt/count);
    fflush(stdout);

    pthread_exit(NULL);

    return NULL;
}

int main(int argc, char * argv[])
{
    num_threads = (argc>1) ? atoi(argv[1]) : 1;
    printf("L2 barrier test using %d threads \n", num_threads );

    /* this "activates" the L2 atomic data structure */
    Kernel_L2AtomicsAllocate(&barrier, sizeof(L2_Barrier_t) );

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

    free(pool);
 
    return 0;   
}
