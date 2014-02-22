#include <stdio.h>
#include <stdlib.h>
#include <mpp/shmem.h>

#ifdef OPENSHMEM
static void shmem_init(void)
{
    start_pes(0);
    return;
}

static void shmem_finalize(void)
{
    return;
}

static int num_pes(void)
{
    return _num_pes();
}

static int my_pe(void)
{
    return _my_pe();
}
#endif

#define CHECK_SHEAP_IS_SYMMETRIC

static long sheap_base;

static int sheap_is_symmetric(long base)
{
    int errors = 0;
#ifdef CHECK_SHEAP_IS_SYMMETRIC
    int mype = my_pe();
    int npes = num_pes();

    sheap_base = base;

    /* this is an inefficient N^2 implementation of what should be a reduction */
    int i;
    for (i=0; i<npes; i++)
    {
        long remote_sheap_base;
        shmem_long_get(&remote_sheap_base, &sheap_base, (size_t)1, i);
        if (sheap_base != remote_sheap_base)
        {
            printf("PE %d: the symmetric heap is not actually symmetric: my base = %p, PE %d base = %p \n",
                   mype, sheap_base, i, remote_sheap_base);
            errors++;
        }
    }
    shmem_barrier_all();

    if (errors==0)
            printf("PE %d: the symmetric heap is symmetric: my base = %p \n",
                   mype, sheap_base);
#endif
    return errors; /* returns 0 on success */
}

