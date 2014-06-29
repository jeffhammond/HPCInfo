#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MPI_THREAD_STRING(level)  \
    ( level==MPI_THREAD_SERIALIZED ? "THREAD_SERIALIZED" : \
        ( level==MPI_THREAD_MULTIPLE ? "THREAD_MULTIPLE" : \
            ( level==MPI_THREAD_FUNNELED ? "THREAD_FUNNELED" : \
                ( level==MPI_THREAD_SINGLE ? "THREAD_SINGLE" : "THIS_IS_IMPOSSIBLE" ) ) ) )

#ifdef CHECK_ERRORS
#define CHECK_MPI(rc) \
    do {                                             \
        if (rc!=MPI_SUCCESS) {                       \
            printf("MPI call failed.  Exiting. \n"); \
            exit(1);                                 \
        }                                            \
    } while (0) 
#else
#define CHECK_MPI(rc)
#endif

int main(int argc, char ** argv)
{
    int rc;

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
    long i, n = (argc>1 ? atol(argv[1]) : 100000);
    rc = MPI_Bcast(&n, count, MPI_LONG, root, MPI_COMM_WORLD); CHECK_MPI(rc);
    if (world_rank==0)
        printf("%d: using %ld samples.\n", world_rank, world_size*n);

    /* seed the RNG with something unique to a rank */
    srand(world_rank);

    long in = 0, total = 0;
    for (i=0;i<n;i++)
    {
        register double x = (double)rand()/(double)RAND_MAX;
        register double y = (double)rand()/(double)RAND_MAX;
        register double z = x*x + y*y;
        if (z<1.0) in++;
    }

    rc = MPI_Reduce(&in, &total, count, MPI_LONG, MPI_SUM, root, MPI_COMM_WORLD); CHECK_MPI(rc);
    double pi = 4.0*(double)total/(world_size*n);
    if (world_rank==0)
        printf("%d: pi = %12.8lf.\n", world_rank, pi);

    MPI_Finalize();
    return 0;
}
