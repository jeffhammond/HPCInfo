#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int np, me;
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&me);

    int rlen = 0;
    char pname[MPI_MAX_PROCESSOR_NAME+1] = {0};
    MPI_Get_processor_name(pname, &rlen);

    printf("Hello from %d of %d processors (name=%s)\n", me, np, pname);

    /* create the shared-memory (per-node) communicator */
    MPI_Comm comm_shared = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shared);

    int localnp, localme;
    MPI_Comm_size(comm_shared,&localnp);
    MPI_Comm_rank(comm_shared,&localme);

    /* allocate the shared-memory window */
    MPI_Aint nbytes = (argc>1) ? atol(argv[1]) : 1000;
    MPI_Win win_shared = MPI_WIN_NULL;
    void * baseptr = NULL;
    MPI_Win_allocate_shared(nbytes, 1, MPI_INFO_NULL, comm_shared, &baseptr, &win_shared);

    for (int i=0; i<localnp; i++) {
        int rank = i;
        MPI_Aint lsize = 0;
        int ldisp = 0;
        void * lbase = NULL;
        MPI_Win_shared_query(win_shared, rank, &lsize, &ldisp, &lbase);
        printf("global %d of %d, local %d of %d, size=%zu, disp=%d, base=%p\n",
                me, np, localme, localnp, lsize, ldisp, lbase);
    }



    MPI_Win_free(&win_shared);

    MPI_Finalize();
    return 0;
}
