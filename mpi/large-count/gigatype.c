#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char * argv[])
{
    const long n = (argc > 1) ? atol(argv[1]) : 100;

    int rc = MPI_Init(NULL,NULL);

    srand((unsigned)MPI_Wtime());

    int * ab = malloc(n*sizeof(int));
    for (long i=0; i<n; i++) ab[i] = 1;

    int * ad = malloc(n*sizeof(int));
    for (long i=0; i<n; i++) ad[i] = rand() % n;

    MPI_Datatype dt;
    double t0,t1;

    t0 = MPI_Wtime();
    rc =  MPI_Type_indexed(n,ab,ad,MPI_CHAR,&dt);
    t1 = MPI_Wtime();
    if (rc != MPI_SUCCESS) printf("bad=%d\n",rc);
    printf("MPI_Type_indexed=%lf\n",t1-t0);

    t0 = MPI_Wtime();
    rc = MPI_Type_commit(&dt);
    t1 = MPI_Wtime();
    if (rc != MPI_SUCCESS) printf("bad=%d\n",rc);
    printf("MPI_Type_commit=%lf\n",t1-t0);

    t0 = MPI_Wtime();
    rc = MPI_Type_free(&dt);
    t1 = MPI_Wtime();
    if (rc != MPI_SUCCESS) printf("bad=%d\n",rc);
    printf("MPI_Type_free=%lf\n",t1-t0);

    MPI_Finalize();

    return 0;
}
