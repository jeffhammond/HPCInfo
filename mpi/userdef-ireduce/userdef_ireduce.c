#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

int me, np;

void fn(void * invec, void * inoutvec, int * len, MPI_Datatype * dt)
{
    printf("%d: fn: dt=%p\n",me,dt);
#ifdef MPICH
    printf("%d: fn: *dt=0x%x\n",me,*(int*)dt);
#else
    printf("%d: fn: *dt=%p\n",me,*(void**)dt);
#endif
    fflush(0);
    usleep(1);
}

int main(int argc, char * argv[])
{
    int provided;
    MPI_Init(&argc, &argv);

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN);

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

    printf("%d: main: &dt=%p\n",me,&dt);
#ifdef MPICH
    printf("%d: main: dt=0x%x\n",me,(int)dt);
#else
    printf("%d: main: dt=%p\n",me,(void*)dt);
#endif

    MPI_Request r = MPI_REQUEST_NULL;
    MPI_Iallreduce(x, y, 1, dt, op, MPI_COMM_WORLD, &r);
    MPI_Type_free(&dt);
    MPI_Op_free(&op);
    usleep(1);
    MPI_Wait(&r, MPI_STATUS_IGNORE);
    fflush(0);
    usleep(1);
    MPI_Finalize();

    free(x);
    free(y);

    return 0;
}
