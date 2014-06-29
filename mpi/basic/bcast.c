#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MPI_THREAD_STRING(level)  \
    ( level==MPI_THREAD_SERIALIZED ? "THREAD_SERIALIZED" : \
        ( level==MPI_THREAD_MULTIPLE ? "THREAD_MULTIPLE" : \
            ( level==MPI_THREAD_FUNNELED ? "THREAD_FUNNELED" : \
                ( level==MPI_THREAD_SINGLE ? "THREAD_SINGLE" : "THIS_IS_IMPOSSIBLE" ) ) ) )

#ifdef CHECK_MPI_ERRORS
#define CHECK_MPI(rc) \
    do {                                                \
        if (rc!=MPI_SUCCESS) {                          \
            printf("MPI call failed.  Exiting. \n");    \
            exit(1);                                    \
        }                                               \
    } while (0) 
#else
#define CHECK_MPI(rc)
#endif

int main(int argc, char ** argv)
{
    int rc = MPI_SUCCESS;

    /* These are the desired and available thread support.
       A hybrid code where all MPI calls are made from the main thread can used FUNNELED.
       If threads are making MPI calls, MULTIPLE is appropriate. */
    int requested = MPI_THREAD_FUNNELED, provided;

    /* MPICH2 will be substantially more efficient than OpenMPI 
       for MPI_THREAD_{FUNNELED,SERIALIZED} but this is unlikely
       to be a serious bottleneck. */
    rc = MPI_Init_thread(&argc, &argv, requested, &provided); CHECK_MPI(rc);
    if (provided<requested)
    {
        printf("MPI_Init_thread provided %s when %s was requested.  Exiting. \n",
               MPI_THREAD_STRING(provided), MPI_THREAD_STRING(requested) );
        exit(1);
    }

    int world_size, world_rank;

    rc = MPI_Comm_size(MPI_COMM_WORLD,&world_size); CHECK_MPI(rc);
    rc = MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); CHECK_MPI(rc);

    int root = 0, count = 1;

    /* the ternary is often branchless... */
    int i, n = (argc>1 ? atoi(argv[1]) : 1000);
    rc = MPI_Bcast(&n, count, MPI_INT, root, MPI_COMM_WORLD); CHECK_MPI(rc);

    /* seed the RNG with something unique to a rank */
    srand(world_rank);

    double x = 0.0;
    for (i=0;i<n;i++)
        x += (double)rand()/(double)RAND_MAX;

    printf("%d: The sum of %d random numbers is %lf.\n", world_rank, n, x);

    MPI_Finalize();
    return 0;
}
