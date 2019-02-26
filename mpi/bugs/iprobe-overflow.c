#include <stdio.h>
#include <limits.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

/*
*   Icc threw a diagnostic when i declared in for().
*/
    long i;
    double t0 = MPI_Wtime();

/*
*   Set limit of for() to 3G since bug breaks after 2.14G.
*/
    
    for (i=0; i <= 3000000000L; i++) {
        int flag = 0;
        MPI_Status status;
        int rc = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag!=0) MPI_Abort(MPI_COMM_WORLD, flag);
        if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
        if ((i%1000000L)==0) {
            double t1 = MPI_Wtime();
            printf("%ld iterations, %lf seconds\n", i, t1-t0);
        }
    }

    MPI_Finalize();
}
