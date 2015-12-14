#include <stdio.h>
#include <stdlib.h>

#include <shmem.h>
#if (_SHMEM_MAJOR_VERSION==1) && (_SHMEM_MINOR_VERSION<3)
/* #warning Cray does not comply with OpenSHMEM 1.3 yet. */
static void * shmem_malloc(size_t n) { return shmalloc(n); }
static void shmem_free(void* p) { shfree(p); }
#endif

/***************************************************************/
#include <shmem.h>
int main(void) {
    shmem_init();
    if (num_pes()<2) exit(1);
    /* allocate from the global heap */
    int * A = shmem_malloc(sizeof(int));
    int B = 37;
    /* store contents of local data B at PE 0 into A at PE 1 */
    if (my_pe()==0) shmem_int_put(A,&B,1,1);
    /* global synchronization of execution and data */
    shmem_barrier_all();
    /* observe the result of the store */
    if (my_pe()==1) printf("A@1=%d\n",*A);
    shmem_free(A);
    shmem_finalize();
    return 0;
}
/***************************************************************/
