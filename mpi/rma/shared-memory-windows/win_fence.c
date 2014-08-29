#include <stdio.h>
#include <mpi.h>

/* If val is the same at all MPI processes in comm,
 * this function returns 1, else 0. */

int coll_check_equal(int val, MPI_Comm comm)
{
    int min, max;
    MPI_Allreduce(&val, &min, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&val, &max, 1, MPI_INT, MPI_MAX, comm);
    return (min==max ? 1 : 0);
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

    /*******************************************************/

    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, shwin);
    if (rank==0) {
        *shptr = 42; /* Answer to the Ultimate Question of Life, The Universe, and Everything. */
    }
    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, shwin);

    //MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, shwin);
    if (rptr!=NULL && rsize>0) {
        lint = *rptr;
    }
    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, shwin);

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
