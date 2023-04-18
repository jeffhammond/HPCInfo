#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>

void print_affinity(void)
{


}

int main(int argc, char* argv[])
{
    int required = MPI_THREAD_SERIALIZED, provided;
    MPI_Init_thread(&argc, &argv, required, &provided);

    MPI_Finalize();
    return 0;
}
