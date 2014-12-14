#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.>

typedef enum {
    int mpi_classic  = 1,
    int mpi_plus_mpi = 2,
    int mpi_plus_omp = 3,
    int mpi_rma      = 4
} pmethod;

typedef struct {
    int blocksize;
    int pepn; /* PEs per node: OpenMP threads or MPI procs */
    MPI_Win winnode;
    MPI_Comm commnode;
    MPI_Comm commcart;
    double * nodeptrs[]; /* pointers to all blocks in node */
} dmatrix;

int allocate_blocked_matrix(MPI_Comm comm, int totaldim, enum pmethod pm, dmatrix * dm)
{
    int commsize;
    MPI_Comm_size(comm, &commsize);

    int sqsize = (int)floor(sqrt((double)commsize));


    int dims[2] = {0,0};
    MPI_Dims_create(commsize, 2, dims);
    MPI_Cart_create(comm, 2, dims, periods, 1 /* reorder */, &(dm->commcart) );

    int xdimc = totaldim / dims[0];
    int xdimr = totaldim % dims[1];

    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0 /* key */, MPI_INFO_NULL, &(dm->commnode) );
    MPI_Win_allocate_shared(blockbytes, sizeof(double), MPI_INFO_NULL, &(dm->commnode), &winptr, &(dm->winnode) );

    if (pm==mpi_classic) {
    } else if (pm==mpi_plus_mpi) {
    } else if (pm==mpi_plus_omp) {
    } else if (pm==mpi_rma) {
    }

    return 0;
}    
