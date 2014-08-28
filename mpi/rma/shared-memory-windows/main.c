#include <stdio.h>
#include <mpi.h>

/* This function synchronizes process i with process j
 * in such a way that this function returns on process j
 * only after it has been called on process i.
 *
 * No additional semantic guarantees are provided.
 *
 * The process ranks are with respect to the input communication. */

int p2p_xsync(int i, int j, MPI_Comm comm)
{
    /* Avoid deadlock for stupid usage. */
    if (i==j) return;

    int tag = 666; /* The number of the beast. */

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank==i) {
        MPI_Send(NULL, 0, MPI_INT, j, tag, comm);
    } else if (rank==j) {
        MPI_Recv(NULL, 0, MPI_INT, i, tag, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int main(int argc, char * argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *   shptr;
    MPI_Win shwin;
    MPI_Win_allocate_shared(rank==0 ? sizeof(int) : 0,sizeof(int), MPI_INFO_NULL,
                            MPI_COMM_WORLD, &shptr, MPI_Win *win)

        fdsjkf


    MPI_Win_free(shwin);

    MPI_Finalize();

    return 0;
}
