#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "armci.h"

double start,finish,timing;

/***************************************************************************
 *                                                                         *
 * simple_put:                                                             *
 *       -demonstrates how to allocate some shared segements with ARMCI    *
 *       -demonstrates how to do one-sided point-to-point communication    *
 *                                                                         *
 ***************************************************************************/

int main(int argc, char **argv)
{
    int me,nproc;
    int test;
    int status;

    int desired = MPI_THREAD_MULTIPLE;
    int provided;
    MPI_Init_thread(&argc, &argv, desired, &provided);

    ARMCI_Init();

    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);

    int len = ( argc > 1 ? atoi(argv[1]) : 1000 );
    int status;
    int n,i;
    double t0,t1;

    double** addr_vec = (double **) malloc( nproc * sizeof(double *) );
    ARMCI_Malloc((void **) addr_vec, len*sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);

    /* initialization of local segments */
    for( i=0 ; i<len ; i++ ){
       addr_vec[me][i] = (double) (1000*me+i);    
    }

    /* print before exchange */
    for( n=0 ; n<nproc ; n++){
       MPI_Barrier(MPI_COMM_WORLD);
       if (n==me){
          printf("values before exchange\n");
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec[%d][%d] = %f\n", n, n, i, addr_vec[n][i]);
          }
          fflush(stdout);
       }
       MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* even processes put from odd right neighbors */
    if (me%2 == 0){
       t0 = MPI_Wtime();
       status = ARMCI_Put(addr_vec[me], addr_vec[me+1], len*sizeof(double), me+1);
       t1 = MPI_Wtime();
       if(status != 0){
    	  if (me == 0) printf("%s: ARMCI_Put failed at line %d\n",,);
       }
       printf("Proc %d: Put Latency=%lf microseconds\n",me,1e6*(t1-t0)/len);
       fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    /* print after exchange */
    for( n=0 ; n<nproc ; n++){
       MPI_Barrier(MPI_COMM_WORLD);
       if (n==me){
          printf("values after exchange\n");
          for( i=0 ; i<len ; i++ ){
             printf("proc %d: addr_vec[%d][%d] = %f\n", n, n, i, addr_vec[n][i]);
          }
          fflush(stdout);
       }
       MPI_Barrier(MPI_COMM_WORLD);
    }

    ARMCI_Finalize();
    MPI_Finalize();

    return(0);
}
