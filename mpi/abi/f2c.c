#include <mpi.h>

int main(void)
{
    MPI_Init(NULL,NULL);
    MPI_Comm c = MPI_Comm_f2c( MPI_Comm_c2f(MPI_COMM_WORLD) );
    MPI_Finalize();
    return 0;
}
