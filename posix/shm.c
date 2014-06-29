#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <pthread.h>
#include <mpi.h>

//#define DEV_SHM
#define POSIX_SHM

#ifdef __bgp__
#  include <asm-generic/errno-base.h> /* BGP only? */
#  include </bgsys/drivers/ppcfloor/arch/include/spi/kernel_interface.h>
#  include </bgsys/drivers/ppcfloor/arch/include/common/bgp_personality.h>
#  include </bgsys/drivers/ppcfloor/arch/include/common/bgp_personality_inlines.h>
#endif

#ifdef __bgq__
#  include <spi/include/kernel/location.h>
#endif

int main(int argc, char* argv[])
{
    int i,j;

    int world_rank = -1, world_size = -1;
    int mpi_result = MPI_SUCCESS;

    int color = -1;
    int ranks_per_node = -1;
    MPI_Comm IntraNodeComm;

    int node_shmem_bytes; 

    MPI_Init(&argc,&argv);
    mpi_result = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    assert(mpi_result==MPI_SUCCESS);
    mpi_result = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(mpi_result==MPI_SUCCESS);

    if (world_rank==0)
    {
        char * env_char;
        int units = 1;
        int num_count = 0;
        env_char = getenv("NODE_SHARED_MEMORY");
        if (env_char!=NULL)
        {
            if      ( NULL != strstr(env_char,"G") ) units = 1000000000;
            else if ( NULL != strstr(env_char,"M") ) units = 1000000;
            else if ( NULL != strstr(env_char,"K") ) units = 1000;
            else                                     units = 1;

            num_count = strspn(env_char, "0123456789");
            memset( &env_char[num_count], ' ', strlen(env_char)-num_count);

            node_shmem_bytes = units * atoi(env_char);
            printf("%7d: NODE_SHARED_MEMORY = %d bytes \n", world_rank, node_shmem_bytes );
        }
        else
        {
            node_shmem_bytes = getpagesize();
            printf("%7d: NODE_SHARED_MEMORY = %d bytes \n", world_rank, node_shmem_bytes );
        }
    }
    mpi_result = MPI_Bcast( &node_shmem_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD );
    assert(mpi_result==MPI_SUCCESS);

    int node_shmem_count = node_shmem_bytes/sizeof(double);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

#if defined(__bgp__)
    uint32_t xSize, ySize, zSize, tSize;
    uint32_t xRank, yRank, zRank, tRank;

    MPIX_rank2torus( world_size-1, &xSize, &ySize, &zSize, &tSize );
    xSize++; ySize++; zSize++;

    MPIX_rank2torus( world_rank, &xRank, &yRank, &zRank, &tRank );
    color = xRank + yRank*xSize + zRank*ySize*xSize;
#else
    color = 0;
#endif

    mpi_result = MPI_Comm_split(MPI_COMM_WORLD, color, 0, &IntraNodeComm);
    assert(mpi_result==MPI_SUCCESS);

    int subcomm_rank = -1;
    mpi_result = MPI_Comm_rank(IntraNodeComm, &subcomm_rank);
    assert(mpi_result==MPI_SUCCESS);

#if defined(POSIX_SHM)
    int fd;
    if (subcomm_rank==0)
        fd = shm_open("/foo", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );

    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

    if (subcomm_rank!=0)
        fd = shm_open("/foo", O_RDWR, S_IRUSR | S_IWUSR );

    if (fd<0) printf("%7d: shm_open failed: %d \n", world_rank, fd);
    else      printf("%7d: shm_open succeeded: %d \n", world_rank, fd);
#elif defined(DEV_SHM)
    int fd = open("/dev/shm/foo", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if (fd<0) printf("%7d: open failed: %d \n", world_rank, fd);
    else      printf("%7d: open succeeded: %d \n", world_rank, fd);
#else
    int fd = -1;
    printf("%7d: no file backing \n", world_rank);
#endif
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

    if (fd>=0 && subcomm_rank==0)
    {
        int rc = ftruncate(fd, node_shmem_bytes);
        if (rc==0) printf("%7d: ftruncate succeeded \n", world_rank);
        else       printf("%7d: ftruncate failed \n", world_rank);
    }
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

#ifdef __bgp__
    double * ptr = NULL;
    _BGP_Personality_t pers;
    Kernel_GetPersonality(&pers, sizeof(pers));

    if( BGP_Personality_processConfig(&pers) == _BGP_PERS_PROCESSCONFIG_SMP )
    {
        printf("SMP mode => MAP_PRIVATE | MAP_ANONYMOUS \n");
        ptr = mmap( NULL, node_shmem_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, fd, 0 );
    }
    else
    {
        if (node_shmem_bytes>pers.Kernel_Config.SharedMemMB)
        {
            printf("node_shmem_bytes (%d) greater than pers.Kernel_Config.SharedMemMB (%d) - allocating the latter \n", 
                   node_shmem_bytes, pers.Kernel_Config.SharedMemMB );
            node_shmem_bytes = pers.Kernel_Config.SharedMemMB;
        }
        ptr = mmap( NULL, node_shmem_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
    }
#else
    double * ptr = mmap( NULL, node_shmem_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
#endif
    if (ptr==NULL) printf("%7d: mmap failed \n", world_rank);
    else           printf("%7d: mmap succeeded \n", world_rank);
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

    mpi_result = MPI_Comm_size(IntraNodeComm, &ranks_per_node );
    assert(mpi_result==MPI_SUCCESS);
    if (0==subcomm_rank) printf("%7d: ranks_per_node = %d \n", world_rank, ranks_per_node );
    fflush(stdout);

    for (i=0; i<ranks_per_node; i++)
    {
        if (i==subcomm_rank)
       {
            printf("%7d: subcomm_rank %d setting the buffer \n", world_rank, subcomm_rank );
            for (j=0; j<node_shmem_count; j++ ) ptr[j] = (double)i;
            printf("%7d: memset succeeded \n", world_rank);

            int rc = msync(ptr, node_shmem_bytes, MS_INVALIDATE | MS_SYNC);
            if (rc==0) printf("%7d: msync succeeded \n", world_rank);
            else       printf("%7d: msync failed \n", world_rank);
        }

        fflush(stdout);
        mpi_result = MPI_Barrier(MPI_COMM_WORLD);
        assert(mpi_result==MPI_SUCCESS);

        printf("%7d: ptr = %lf ... %lf \n", world_rank, ptr[0], ptr[node_shmem_count-1]);
        fflush(stdout);

        mpi_result = MPI_Barrier(MPI_COMM_WORLD);
        assert(mpi_result==MPI_SUCCESS);
    }
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

#if defined(POSIX_SHM)
    //if (fd>=0)
    if (fd>=0 && subcomm_rank==0)
    {
        int rc = -1;

#if 0
        // cannot truncate shm files apparently
        rc = ftruncate(fd, 0);
        if (rc==0) printf("%7d: ftruncate succeeded \n", world_rank);
        else       printf("%7d: ftruncate failed \n", world_rank);
#endif
        rc = shm_unlink("/foo");
        if (rc==0) printf("%7d: shm_unlink succeeded \n", world_rank);
        else       printf("%7d: shm_unlink failed \n", world_rank);
    }
#elif defined(DEV_SHM)
    if (fd>=0 && subcomm_rank==0)
    {
        int rc = -1;

        rc = ftruncate(fd, 0);
        if (rc==0) printf("%7d: ftruncate succeeded \n", world_rank);
        else       printf("%7d: ftruncate failed \n", world_rank);

        rc = close(fd);
        if (rc==0) printf("%7d: close succeeded \n", world_rank);
        else       printf("%7d: close failed \n", world_rank);
    }
#endif
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

    if (ptr!=NULL)
    {
        int rc = munmap(ptr, node_shmem_bytes);
        if (rc==0) printf("%7d: munmap succeeded \n", world_rank);
        else       printf("%7d: munmap failed \n", world_rank);
    }
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

    if (world_rank==0) printf("%7d: all done! \n", world_rank );
    fflush(stdout);
    mpi_result = MPI_Barrier(MPI_COMM_WORLD);
    assert(mpi_result==MPI_SUCCESS);

    MPI_Finalize();
    return 0;
}
