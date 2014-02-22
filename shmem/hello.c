#include "myshmem.h"

int main(int argc, char* argv[])
{
    shmem_init();
    printf("Hello world: I am PE %d of %d.\n", my_pe(), num_pes());
    shmem_barrier_all();
    shmem_finalize();
    return 0;
}
