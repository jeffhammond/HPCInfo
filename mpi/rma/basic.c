#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    assert(provided==MPI_THREAD_SINGLE);

    int me;
    int nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    int status;
    double t0,t1,t2,t3,t4,t5;
    double tt0,tt1,tt2,tt3,tt4;

    int bufSize = ( argc>1 ? atoi(argv[1]) : 1000000 );
    if (me==0) printf("%d: bufSize = %d doubles\n",me,bufSize);

    /* allocate RMA buffers for windows */
    double* m1;
    double* m2;
    status = MPI_Alloc_mem(bufSize * sizeof(double), MPI_INFO_NULL, &m1);
    status = MPI_Alloc_mem(bufSize * sizeof(double), MPI_INFO_NULL, &m2);

    /* register remote pointers */
    MPI_Win w1;
    MPI_Win w2;
    status = MPI_Win_create(m1, bufSize * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w1);
    status = MPI_Win_create(m2, bufSize * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w2);
    MPI_Barrier(MPI_COMM_WORLD);

    /* allocate RMA buffers */
    double* b1;
    double* b2;
    status = MPI_Alloc_mem(bufSize * sizeof(double), MPI_INFO_NULL, &b1);
    status = MPI_Alloc_mem(bufSize * sizeof(double), MPI_INFO_NULL, &b2);

    /* initialize buffers */
    int i;
    for (i=0;i<bufSize;i++) b1[i]=1.0*me;
    for (i=0;i<bufSize;i++) b2[i]=-1.0;

    status = MPI_Win_fence( MPI_MODE_NOPRECEDE | MPI_MODE_NOSTORE , w1 );
    status = MPI_Win_fence( MPI_MODE_NOPRECEDE | MPI_MODE_NOSTORE , w2);
    status = MPI_Put(b1, bufSize, MPI_DOUBLE, me, 0, bufSize, MPI_DOUBLE, w1);
    status = MPI_Put(b2, bufSize, MPI_DOUBLE, me, 0, bufSize, MPI_DOUBLE, w2);
    status = MPI_Win_fence( MPI_MODE_NOSTORE , w1);
    status = MPI_Win_fence( MPI_MODE_NOSTORE , w2);

    int target;
    int j;
    double dt,bw;
    MPI_Barrier(MPI_COMM_WORLD);
    if (me==0){
        printf("MPI_Get performance test for buffer size = %d doubles\n",bufSize);
        printf("  jump    host   target       get (s)       BW (MB/s)\n");
        printf("===========================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (j=0;j<nproc;j++){
        target = (me+j) % nproc;
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        status = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, MPI_MODE_NOCHECK, w1);
        t1 = MPI_Wtime();
        status = MPI_Get(b2, bufSize, MPI_DOUBLE, target, 0, bufSize, MPI_DOUBLE, w1);
        t2 = MPI_Wtime();
        status = MPI_Win_unlock(target, w1);
        t3 = MPI_Wtime();
        for (i=0;i<bufSize;i++) assert( b2[i]==(1.0*target) );
        dt = t3 - t0;
        bw = (double)bufSize*sizeof(double)*(1e-6)/dt;
        printf("%4d     %4d     %4d       %9.6f     %9.3f\n",j,me,target,dt,bw);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    status = MPI_Win_free(&w2);
    status = MPI_Win_free(&w1);

    status = MPI_Free_mem(b2);
    status = MPI_Free_mem(b1);

    status = MPI_Free_mem(m2);
    status = MPI_Free_mem(m1);

    MPI_Barrier(MPI_COMM_WORLD);

    if (me==0) printf("%d: MPI_Finalize\n",me);
    MPI_Finalize();

    return(0);
}
