#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

void fn(void * invec, void * inoutvec, int * len, MPI_Datatype * dt)
{
    printf("fn: dt=%p\n",dt); 
}

int main(int argc, char * argv[])
{
    int provided;
    MPI_Init(&argc, &argv);

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Op op;
    MPI_Op_create(fn,0,&op);

    int count = 100;
    double * x = malloc( count * sizeof(double) );
    double * y = malloc( count * sizeof(double) );
    for (int i=0; i<count; i++) {
        x[i] = i;
        y[i] = 0;
    }

    MPI_Datatype dt = MPI_DATATYPE_NULL;
    MPI_Type_contiguous(count, MPI_DOUBLE, &dt);
    MPI_Type_commit(&dt);

    printf("main: &dt=%p\n", &dt);

    MPI_Request r = MPI_REQUEST_NULL;
    MPI_Iallreduce(x, y, count, dt, op, MPI_COMM_WORLD, &r);
    //sleep(1);
    MPI_Wait(&r, MPI_STATUS_IGNORE);

    MPI_Type_free(&dt);
    MPI_Op_free(&op);
    MPI_Finalize();

    //free(x);
    //free(y);

    return 0;
}
