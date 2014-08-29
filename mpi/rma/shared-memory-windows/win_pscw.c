#include <stdio.h>
#include <mpi.h>

int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Group MPI_GROUP_WORLD;
    MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size<2) {
        printf("You must use at least 2 processes for this test.\n");
        MPI_Group_free(&MPI_GROUP_WORLD);
    }

    int ranks[1] = {1};
    MPI_Group MPI_GROUP_ONE;
    MPI_Group_incl(MPI_GROUP_WORLD, 1, ranks, &MPI_GROUP_ONE);
    ranks[0] = 0;
    MPI_Group MPI_GROUP_ZERO;
    MPI_Group_incl(MPI_GROUP_WORLD, 1, ranks, &MPI_GROUP_ZERO);

    int *   shptr = NULL;
    MPI_Win shwin;
    MPI_Win_allocate_shared(rank==0 ? sizeof(int) : 0,sizeof(int),
                            MPI_INFO_NULL, MPI_COMM_WORLD,
                            &shptr, &shwin);

    MPI_Aint rsize = 0;
    int rdisp = 0;
    int * rptr = NULL;
    MPI_Win_shared_query(shwin, 0, &rsize, &rdisp, &rptr);

    /*******************************************************/

    if (rank==0) {
        MPI_Win_post(MPI_GROUP_ONE, 0, shwin);
        *shptr = 42; /* Answer to the Ultimate Question of Life, The Universe, and Everything. */
        MPI_Win_wait(shwin);

        MPI_Win_post(MPI_GROUP_ONE, 0, shwin);
        MPI_Win_wait(shwin);
    } else if (rank==1) {
        int lint;

        MPI_Win_start(MPI_GROUP_ZERO, 0, shwin);
        MPI_Win_complete(shwin);

        MPI_Win_start(MPI_GROUP_ZERO, 0, shwin);
        if (rptr!=NULL && rsize>0) {
            lint = *rptr;
        } else {
            lint = -911;
            printf("rptr=%p rsize=%zu \n", rptr, (size_t)rsize);
        }
        MPI_Win_complete(shwin);

        if (lint==42) {
            printf("SUCCESS!\n");
        } else {
            printf("lint=%d\n", lint);
        }
    }

    /*******************************************************/

    MPI_Win_free(&shwin);

    MPI_Group_free(&MPI_GROUP_ONE);
    MPI_Group_free(&MPI_GROUP_ZERO);
    MPI_Group_free(&MPI_GROUP_WORLD);
    MPI_Finalize();

    return 0;
}
