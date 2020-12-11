#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int n = (argc>1) ? atoi(argv[1]) : 1000;

    int me = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    MPI_Request * r = malloc( 2 * n * sizeof(MPI_Request) );

    for (int i=0; i<n; i++) {
        MPI_Irecv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &r[i]);
    }
    for (int i=0; i<n; i++) {
        MPI_Isend(NULL, 0, MPI_CHAR, me, i, MPI_COMM_WORLD, &r[n+i]);
    }
    MPI_Waitall(n, r, MPI_STATUSES_IGNORE);

    free(r);

    MPI_Finalize();
}
