#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void user_logical_and(void * invec, void * inoutvec, int * len, MPI_Datatype * datatype)
{
    if (*datatype != MPI_INT) {
        char name[MPI_MAX_OBJECT_NAME+1] = {0};
        int n;
        MPI_Type_get_name(*datatype, name, &n);
        printf("datatype (%s) does not match arguments!\n",name);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    for (int i=0; i<*len; i++) {
        ((int*)inoutvec)[i] = (((int*)invec)[i] && ((int*)inoutvec)[i]);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);

    const int n = (argc > 1) ? atoi(argv[1]) : 1000;

    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Op op;
    MPI_Op_create(&user_logical_and, 1, &op);

    int * a = calloc(n,sizeof(int));
    int * b = calloc(n,sizeof(int));

    for (int i=0; i<n; i++) {
        a[i] = (int)(np > 1);
        b[i] = 0;
    }

    // correctness
    MPI_Allreduce(a, b, n, MPI_INT, op, MPI_COMM_WORLD);

    for (int i=0; i<n; i++) {
        if (b[i] != (np > 1)) MPI_Abort(MPI_COMM_SELF,np);
    }

    // timing
    double t0 = MPI_Wtime();
    for (int i=0; i<1000000; i++) {
        MPI_Allreduce(a, b, n, MPI_INT, op, MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();
    printf("C time=%lf (for 1000000 calls)\n",t1-t0);

    free(a);
    free(b);

    MPI_Op_free(&op);

    if (me == 0) printf("OK\n");

    MPI_Finalize();

    return 0;
}
