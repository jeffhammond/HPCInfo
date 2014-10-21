#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(int argc, char* argv[])
{
    int requested=MPI_THREAD_MULTIPLE, provided;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    if (provided<requested) MPI_Abort(MPI_COMM_WORLD, provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Win win;
    double * base;
    MPI_Win_allocate(sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &base, &win); 
    /* the following is erroneous usage, which precludes the use of passive target
     * synchronization inside of threaded regions... */
    MPI_Win_lock(MPI_LOCK_SHARED, 0 /* rank */, 0 /* assert */, win); 
    MPI_Win_lock(MPI_LOCK_SHARED, 0 /* rank */, 0 /* assert */, win); 
    MPI_Win_unlock(0 /* rank */, win); 
    MPI_Win_free(&win);
    MPI_Win_free(&win);

    MPI_Finalize();
    return 0;
}
