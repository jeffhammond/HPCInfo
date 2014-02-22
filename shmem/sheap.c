#include "myshmem.h"

int main(int argc, char* argv[])
{
    shmem_init();
    int mype = my_pe();
    int npes = num_pes();

    int n = ( argc>1 ? atoi(argv[1]) : 1000);
    int * sheap = shmalloc(n*sizeof(int));
    if (sheap==NULL) exit(1);
    printf("PE %d: sheap base = %p\n", mype, sheap);
    shfree(sheap);

    shmem_finalize();
    return 0;
}
