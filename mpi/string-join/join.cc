#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int me=0, np=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&np);


    MPI_Finalize();
    return 0;
}
