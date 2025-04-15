    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #include <mpi.h>

    int main(int argc, char * argv[])
    {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size!=2) {
            printf("use 2 processes\n");
            MPI_Abort(MPI_COMM_WORLD, size);
        }

        char buf[1024];

        if (rank==0) {
            MPI_Status status;
            MPI_Recv(buf, sizeof(buf), MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_CHAR, &count);
            printf("recv count=%d\n",count);
        } else if (rank==1) {
            MPI_Send(NULL, 0, MPI_CHAR, 0, 9, MPI_COMM_WORLD);
        }

        if (rank==0) {
            printf("successful termination\n");
        }

        MPI_Finalize();
    }
