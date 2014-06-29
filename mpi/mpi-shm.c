/* You need to use <tt>mpicc -I/bgsys/drivers/ppcfloor/arch/include/</tt> 
 * to compile this example if you're on Blue Gene/P. */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <pthread.h>
#include <mpi.h>

#ifdef __bgp__
#  include <spi/kernel_interface.h>
#  include <common/bgp_personality.h>
#  include <common/bgp_personality_inlines.h>
#endif

//#define DEV_SHM
#define POSIX_SHM

#define BGP_SMP_SHM_BROKEN

int main(int argc, char* argv[])
{
    int i;

    int world_rank = -1, world_size = -1;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t size = ( argc>1 ? atoi(argv[1]) : getpagesize() ); 
    printf("%d: size = %ld \n", world_rank, (long)size);

    MPI_Barrier(MPI_COMM_WORLD);

#if defined(POSIX_SHM)
    int fd = shm_open("/bar", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if (fd<0) printf("%d: shm_open failed: %d \n", world_rank, fd);
    else      printf("%d: shm_open succeeded: %d \n", world_rank, fd);
#elif defined(DEV_SHM)
    int fd = open("/dev/shm/foo", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if (fd<0) printf("%d: open failed: %d \n", world_rank, fd);
    else      printf("%d: open succeeded: %d \n", world_rank, fd);
#else
    int fd = -1;
    printf("%d: no file backing \n");
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    if (fd>=0 && world_rank==0)
    {
        int rc = ftruncate(fd, size);
        if (rc==0) printf("%d: ftruncate succeeded \n", world_rank );
        else       printf("%d: ftruncate failed \n", world_rank );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    char * ptr = NULL;
#if defined(__bgp__) && defined(BGP_SMP_SHM_BROKEN)
    _BGP_Personality_t pers;
    Kernel_GetPersonality(&pers, sizeof(pers));
    if( BGP_Personality_processConfig(&pers) == _BGP_PERS_PROCESSCONFIG_SMP )
    {
        printf("%d: SMP mode => MAP_PRIVATE | MAP_ANONYMOUS \n");
        ptr = mmap( NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, fd, 0 );
    }
    else
        ptr = mmap( NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
#else
    ptr = mmap( NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
#endif
    if (ptr==NULL) printf("%d: mmap failed \n", world_rank );
    else           printf("%d: mmap succeeded \n", world_rank );

    MPI_Barrier(MPI_COMM_WORLD);
    
    for (i=0; i<size; i++) printf("%d: ptr[%d] = %c \n", world_rank, i, ptr[i] );

    MPI_Barrier(MPI_COMM_WORLD);

    if (0==world_rank)
    {
        memset(ptr, 'X', size);
        printf("%d: memset succeeded \n", world_rank );

        int rc = msync(ptr, size, MS_INVALIDATE | MS_SYNC);
        if (rc==0) printf("%d: msync succeeded \n", world_rank);
        else       printf("%d: msync failed \n", world_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (i=0; i<size; i++) printf("%d: ptr[%d] = %c \n", world_rank, i, ptr[i] );

    MPI_Barrier(MPI_COMM_WORLD);

    if (fd>=0)
    {
        int rc = ftruncate(fd, 0);
        if (rc==0) printf("%d: ftruncate succeeded \n", world_rank );
        else       printf("%d: ftruncate failed \n", world_rank );
    }

    MPI_Barrier(MPI_COMM_WORLD);

#if defined(POSIX_SHM)
    if (fd>=0)
    {
        int rc = shm_unlink("/bar");
        if (rc==0) printf("%d: shm_unlink succeeded \n", world_rank );
        else       printf("%d: shm_unlink failed \n", world_rank );
    }
#elif defined(DEV_SHM)
    if (fd>=0)
    {
        int rc = close(fd);
        if (rc==0) printf("%d: close succeeded \n");
        else       printf("%d: close failed \n");
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);

    if (ptr!=NULL)
    {
        int rc = munmap(ptr, size);
        if (rc==0) printf("%d: munmap succeeded \n", world_rank );
        else       printf("%d: munmap failed \n", world_rank );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
