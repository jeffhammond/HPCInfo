#include <stdio.h>
#include <mpi.h>

/* This function synchronizes process rank i with process rank j
 * in such a way that this function returns on process rank j
 * only after it has been called on process rank i.
 *
 * No additional semantic guarantees are provided.
 *
 * The process ranks are with respect to the input communicat (comm). */

int p2p_xsync(int i, int j, MPI_Comm comm)
{
    /* Avoid deadlock. */
    if (i==j) {
        return MPI_SUCCESS;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    int tag = 666; /* The number of the beast. */

    if (rank==i) {
        MPI_Send(NULL, 0, MPI_INT, j, tag, comm);
    } else if (rank==j) {
        MPI_Recv(NULL, 0, MPI_INT, i, tag, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

/* If val is the same at all MPI processes in comm,
 * this function returns 1, else 0. */

int coll_check_equal(int val, MPI_Comm comm)
{
    int minmax[2] = {-val,val};
    MPI_Allreduce(MPI_IN_PLACE, minmax, 2, MPI_INT, MPI_MAX, comm);
    return ((-minmax[0])==minmax[1] ? 1 : 0);
}

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *   shptr = NULL;
    MPI_Win shwin;
    MPI_Win_allocate_shared(rank==0 ? sizeof(int) : 0,sizeof(int),
                            MPI_INFO_NULL, MPI_COMM_WORLD,
                            &shptr, &shwin);

    /* l=local r=remote */
    MPI_Aint rsize = 0;
    int rdisp;
    int * rptr = NULL;
    int lint = -999;
    MPI_Win_shared_query(shwin, 0, &rsize, &rdisp, &rptr);
    if (rptr==NULL || rsize!=sizeof(int)) {
        printf("rptr=%p rsize=%zu \n", rptr, (size_t)rsize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /*******************************************************/

    MPI_Win_lock_all(0 /* assertion */, shwin);

    if (rank==0) {
        *shptr = 42; /* Answer to the Ultimate Question of Life, The Universe, and Everything. */
        MPI_Win_sync(shwin);
    }
    for (int j=1; j<size; j++) {
        p2p_xsync(0, j, MPI_COMM_WORLD);
    }
    if (rank!=0) {
        MPI_Win_sync(shwin);
    }
    lint = *rptr;

    MPI_Win_unlock_all(shwin);

    /*******************************************************/

    if (1==coll_check_equal(lint,MPI_COMM_WORLD)) {
        if (rank==0) {
            printf("SUCCESS!\n");
        }
    } else {
        printf("rank %d: lint = %d \n", rank, lint);
    }

    MPI_Win_free(&shwin);

    MPI_Finalize();

    return 0;
}
