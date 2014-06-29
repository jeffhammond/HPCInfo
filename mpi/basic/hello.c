#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MPI_THREAD_STRING(level)  \
        ( level==MPI_THREAD_SERIALIZED ? "THREAD_SERIALIZED" : \
                ( level==MPI_THREAD_MULTIPLE ? "THREAD_MULTIPLE" : \
                        ( level==MPI_THREAD_FUNNELED ? "THREAD_FUNNELED" : \
                                ( level==MPI_THREAD_SINGLE ? "THREAD_SINGLE" : "THIS_IS_IMPOSSIBLE" ) ) ) )

int main(int argc, char ** argv)
{
    /* These are the desired and available thread support.
       A hybrid code where all MPI calls are made from the main thread can used FUNNELED.
       If threads are making MPI calls, MULTIPLE is appropriate. */
    int requested = MPI_THREAD_FUNNELED, provided;

    /* MPICH2 will be substantially more efficient than OpenMPI 
       for MPI_THREAD_{FUNNELED,SERIALIZED} but this is unlikely
       to be a serious bottleneck. */
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided<requested)
    {
        printf("MPI_Init_thread provided %s when %s was requested.  Exiting. \n",
               MPI_THREAD_STRING(provided), MPI_THREAD_STRING(requested) );
        exit(1);
    }

    int world_size, world_rank;

    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    printf("Hello from %d of %d processors\n", world_rank, world_size);

    MPI_Finalize();
    return 0;
}
