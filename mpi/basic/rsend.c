#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

int main(int argc, char * argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size!=2) {
        printf("use 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, size);
    }

    if (argc<2) {
        printf("need an argument for number of bytes\n");
        MPI_Abort(MPI_COMM_WORLD,argc);
    }
    int count = atoi(argv[1]);

    char * buf = malloc(count);
    if (buf==NULL) {
        printf("malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    memset(buf, '\0', count);

    MPI_Request req = MPI_REQUEST_NULL;

    /* tags */
    int data = 0;
    int sync = 1;

    int src = 1;
    int dst = 0;

    if (rank==0) {
        MPI_Irecv(buf, count, MPI_CHAR, src, data, MPI_COMM_WORLD, &req);
        MPI_Send(NULL, 0, MPI_CHAR, src, sync, MPI_COMM_WORLD);
    }

    if (rank==1) {
        MPI_Recv(NULL, 0, MPI_CHAR, dst, sync, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Rsend(buf, count, MPI_CHAR, dst, data, MPI_COMM_WORLD);
    }

    if (rank==0) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    if (rank==0) {
        printf("successful termination\n");
    }

    MPI_Finalize();
}
