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
    if (i==j) return MPI_SUCCESS;

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

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &size);

    int *   shptr;
    MPI_Win shwin;
    MPI_Win_allocate_shared(rank==0 ? sizeof(int) : 0,sizeof(int),
                            MPI_INFO_NULL, MPI_COMM_WORLD,
                            &shptr, &shwin);

    if (rank==0) {
        *shptr = 42; /* Answer to the Ultimate Question of Life, The Universe, and Everything. */
    }

#if RMA_SYNC_MODE == 0
    if (rank==0) {
        MPI_Win_sync(shwin);
    }
    for (int j=1; j<size; j++) {
        p2p_xsync(0, j, MPI_COMM_WORLD);
    }
    if (rank!=0) {
        MPI_Win_sync(shwin);
    }
#elif RMA_SYNC_MODE == 1
#warning NO SYNC
#else
#error Invalid choice of RMA_SYNC_MODE
#endif

    int lint = *shptr; /* Okay here, but not in one's bellybutton... */
    printf("rank %d: lint = %d \n", rank, lint);

    MPI_Win_free(&shwin);

    MPI_Finalize();

    return 0;
}
