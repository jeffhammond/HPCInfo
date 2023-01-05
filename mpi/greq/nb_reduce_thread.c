#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <mpi.h>




int main(int argc, char * argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) MPI_Abort(MPI_COMM_WORLD, provided);



    MPI_Finalize();
    return 0;
}
