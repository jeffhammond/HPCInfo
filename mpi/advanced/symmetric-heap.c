#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <asm-generic/errno-base.h>
#include <pthread.h>

#include <mpi.h>

#if defined(__bgp__) || defined(__bgq__)
#  include <mpix.h>
#elif defined(__CRAYXT) || defined(__CRAYXE)
#  include <pmi.h> 
#elif MPI_VERSION >= 3
// MPICH provides MPI(X)_Comm_split_type
#else
// fall back to Jeff's memory-intensive implementation 
static inline int xstrcmp(const void *a, const void *b) 
{ 
    const char **ia = (const char **)a;
    const char **ib = (const char **)b;
    return strcmp(*ia, *ib);
} 

int MPE_Comm_split_node(MPI_Comm ParentComm, MPI_Comm * NodeComm)
{
    int world_rank = -1, world_size = -1;

    MPI_Comm_rank(ParentComm, &world_rank);
    MPI_Comm_size(ParentComm, &world_size);

    int namelen = 0;
    char procname[MPI_MAX_PROCESSOR_NAME];
    memset(procname, '\0', MPI_MAX_PROCESSOR_NAME);
    MPI_Get_processor_name( procname, &namelen );

    char ** procname_array   = NULL;
    char *  procname_storage = NULL;
    int  *  procname_colors  = NULL;

    if (world_rank==0)
    {
        procname_array = malloc( world_size * sizeof(char *) );
        assert( procname_array != NULL );

        procname_storage = malloc( world_size * namelen * sizeof(char) );
        assert( procname_storage != NULL );

        memset( procname_storage, '\0', world_size * namelen);

        int i;
        for (i=0; i<world_size; i++)
            procname_array[i] = &procname_storage[i*namelen];
    }

    MPI_Gather( procname, namelen, MPI_CHAR, procname_storage, namelen, MPI_CHAR, 0, ParentComm );

    int color = world_rank;

    if (world_rank==0)
    {
        int i;
        qsort(procname_array, world_size, sizeof(char *), (void*) &xstrcmp);

        procname_colors = malloc( world_size * sizeof(int) );
        assert( procname_colors != NULL );

        color = 0;
        procname_colors[0] = color;
        for (i=1; i<world_size; i++)
        {
            if (0!=strncmp(procname_array[i], procname_array[i-1], namelen)) color++;
            procname_colors[i] = color;
        }

        free(procname_storage);
        free(procname_array);
    }
    MPI_Scatter( procname_colors, 1, MPI_INT, &color, 1, MPI_INT, 0, ParentComm );

    if (world_rank==0) 
        free(procname_colors);

    MPI_Comm_split(ParentComm, color, 0, NodeComm);

    return MPI_SUCCESS;
}
#endif

int main(int argc, char* argv[])
{
    int mpi_result = MPI_SUCCESS;

    MPI_Init(&argc,&argv);

    int world_rank= -1, world_size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int sheap_size = 10*1000*1000; /* 10 MB by default */
    if (world_rank==0)
    {
        char * env_char;
        int units = 1;
        int num_count = 0;
        env_char = getenv("BG_SYMMETRIC_HEAP");
        if (env_char!=NULL) 
        {
            if      ( NULL != strstr(env_char,"G") ) units = 1000000000;
            else if ( NULL != strstr(env_char,"M") ) units = 1000000;
            else if ( NULL != strstr(env_char,"K") ) units = 1000;
            else                                     units = 1;

            num_count = strspn(env_char, "0123456789");
            memset( &env_char[num_count], ' ', strlen(env_char)-num_count);
            sheap_size = units * atoi(env_char);
        } else {
            sheap_size = 1024;
        }
        printf("%7d: BG_SYMMETRIC_HEAP = %d bytes \n", world_rank, sheap_size );
    }
    mpi_result = MPI_Bcast( &sheap_size, 1, MPI_INT, 0, MPI_COMM_WORLD );
    assert(mpi_result==MPI_SUCCESS);

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm NodeComm;
#if defined(__bgp__)
    int nodeid, rpn;
    unsigned x,y,z,t;

    /* this is a hack for getting ranks per node */
    MPIX_rank2torus(world_size, &x, &y, &z, &t);
    rpn = t+1;

    MPIX_rank2torus(world_rank, &x, &y, &z, &t);
    nodeid = MPIX_torus2rank(x,y,z,0)/rpn;
    MPI_Comm_split(MPI_COMM_WORLD, nodeid, 0, &NodeComm);
#elif defined(__bgq__)
    int nodeid;
    MPIX_Hardware_t hw;
    MPIX_Hardware(&hw);

    nodeid = hw.Coords[0] * hw.Size[1] * hw.Size[2] * hw.Size[3] * hw.Size[4]
           + hw.Coords[1] * hw.Size[2] * hw.Size[3] * hw.Size[4]
           + hw.Coords[2] * hw.Size[3] * hw.Size[4]
           + hw.Coords[3] * hw.Size[4]
           + hw.Coords[4];
    MPI_Comm_split(MPI_COMM_WORLD, nodeid, 0, &NodeComm);
#elif defined(__CRAYXT) || defined(__CRAYXE)
    int nodeid;
#  if defined(__CRAYXT)
    PMI_Portals_get_nid(world_rank, &nodeid);
#  elif defined(__CRAYXE)
    PMI_Get_nid(world_rank, &nodeid);
#  endif
    MPI_Comm_split(MPI_COMM_WORLD, nodeid, 0, &NodeComm);
#elif MPI_VERSION >= 3
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &NodeComm);
#elif defined (MPICH2) && (MPICH2_NUM_VERSION >= MPICH2_CALC_VERSION(1,5,0,0,0))
    MPIX_Comm_split_type(MPI_COMM_WORLD, MPIX_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &NodeComm);
#else
    MPE_Comm_split_node(MPI_COMM_WORLD, &NodeComm);
#endif

    int rank_in_node = -1, ranks_per_node = 0, num_nodes = 0, my_node = -1;

    MPI_Comm_rank(NodeComm, &rank_in_node);
    MPI_Comm_size(NodeComm, &ranks_per_node);

    num_nodes = world_size/ranks_per_node;
    my_node = (world_rank - rank_in_node)/ranks_per_node;

    printf("%7d: rank_in_node = %2d, ranks_per_node = %2d, my_node = %5d, num_nodes = %5d, world_rank = %7d, world_size = %7d \n",
            world_rank, rank_in_node, ranks_per_node, my_node, num_nodes, world_rank, world_size);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm NodeRankComm;
    int color = rank_in_node;
    int key   = my_node;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &NodeRankComm);

    int subcomm_rank, subcomm_size;
    MPI_Comm_rank(NodeRankComm, &subcomm_rank);
    MPI_Comm_size(NodeRankComm, &subcomm_size);

    void * ptr  = NULL;
    void * addr = NULL;

    /* void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset); */
    if (subcomm_rank==0) /* on node 0 */
    {
        ptr = mmap( 0, sheap_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0 );
        printf("%7d: mmap %d bytes at address %p \n", world_rank, sheap_size, ptr );
        fflush(stdout);
        addr = ptr;
    }

    /* int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) */
    MPI_Bcast( &addr, sizeof(void*), MPI_BYTE, 0, NodeRankComm );

    if (subcomm_rank>0) /* not on node 0 */
    {
        ptr = mmap( addr , sheap_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0 );
        printf("%7d: trying to mmap the same address as the root: wanted %p, got %p (%s) \n", world_rank, addr, ptr, ptr==addr ? "SUCCESS" : "FAILURE" );
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank==0) printf("%7d: all done! \n", world_rank );

    MPI_Finalize();

    return 0;
}

