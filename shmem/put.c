#include "myshmem.h"

int main(int argc, char* argv[])
{
    shmem_init();
    int mype = my_pe();
    int npes = num_pes();

    int i;
    int n = ( argc>1 ? atoi(argv[1]) : 5);
    int * sheap = shmalloc(n*sizeof(int));
    if (sheap==NULL) exit(1);
    for (i=0; i<n; i++)
        sheap[i] = -mype;
    /* apparently Cray SHMEM doesn't call a barrier in shmalloc */
    shmem_barrier_all();

    int symm = sheap_is_symmetric((long)sheap);
    if (symm>0); /* do something else */

    int * local = malloc(n*sizeof(int));
    for (i=0; i<n; i++)
        local[i] = mype;
    shmem_barrier_all();

    int target = (mype+1)%npes;
    shmem_int_put(sheap, local, (size_t)n, target);
    //shmem_quiet();
    shmem_barrier_all();
    target = (mype>0 ? mype-1 : npes-1);
    for (i=0; i<n; i++)
        if (sheap[i] != target)
            printf("PE %d, element %d: correct = %d, got %d \n", mype, i, target, sheap[i]);

    free(local);
    /* it is possible that Cray SHMEM doesn't call a barrier in shfree */
    shmem_barrier_all();
    shfree(sheap);
    
    printf("PE %d is done \n", mype);

    shmem_finalize();
    return 0;
}
