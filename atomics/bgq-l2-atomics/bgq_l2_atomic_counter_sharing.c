#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
//#include <unistd.h>

#include <hwi/include/common/bgq_alignment.h>
#include <hwi/include/bqc/A2_inlines.h>
#include <spi/include/kernel/memory.h>
#include <spi/include/l2/barrier.h>
#include <spi/include/l2/atomic.h>

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

int main(int argc, char * argv[])
{
    const int n = 1024;
    int count = (argc>1) ? atoi(argv[1]) : 1000000;

    /* this "activates" the L2 atomic data structures */
    uint64_t * l2_counters = NULL;
    int rc = posix_memalign((void**)&l2_counters, 2*1024*1024, n * sizeof(uint64_t) ); 
    assert(rc==0 && l2_counters != NULL);
    Kernel_L2AtomicsAllocate(l2_counters, n * sizeof(uint64_t) );

    for (int i=0; i<n; i++) {
        L2_AtomicStore(&(l2_counters[i]), 0);
    }

    #pragma omp parallel
    {
        int me = omp_get_thread_num();
        int jmax = n/omp_get_num_threads();
        for (int j=0; j<jmax; j++) {
            #pragma omp barrier
            uint64_t t0 = GetTimeBase();
            for (int i=0; i<count; i++) {
                uint64_t rval = L2_AtomicLoadIncrement(&(l2_counters[j*me]));
            }
            #pragma omp barrier
            uint64_t t1 = GetTimeBase();
            printf("threads = %d, stride = %d, ticks = %llu \n",
                   omp_get_num_threads(), j, t1-t0);
            fflush(stdout);
        }
    }

    for (int i=0; i<n; i++) {
        uint64_t rval = L2_AtomicLoad(&(l2_counters[i]));
        printf("final value of counter is %llu \n", rval);
    }

    return 0;   
}
