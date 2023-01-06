#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(void)
{
    MPI_Init(NULL,NULL);

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    MPI_Fint i = me;
    MPI_Allreduce(MPI_IN_PLACE, &i, 1, MPI_INTEGER, MPI_LAND, MPI_COMM_WORLD);

    return MPI_Finalize();
}
