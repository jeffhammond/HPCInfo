#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>

void print_affinity(int rank)
{

#ifdef _OPENMP

#endif

}

int main(int argc, char* argv[])
{
    int required = MPI_THREAD_SERIALIZED, provided;
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) abort();

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    printf("C: I am %d of %d\n",me,np);

    fflush(0);
    MPI_Barrier(MPI_COMM_WORLD);

    {
        MPI_Comm node;
        int node_me, node_np;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &node_me);
        MPI_Comm_size(node, &node_np);
        printf("C: rank %d is the %d of %d\n",me,node_me,node_np);
    }

    fflush(0);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
