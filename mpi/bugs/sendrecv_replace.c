/* from https://lists.mpich.org/pipermail/discuss/2019-January/011136.html */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
int main(void)
{
  int rank, size,i;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double* buf = malloc(323262400L*sizeof(double));

  for(i=(1<<28)-4; i< (1<<28)+4; i++){
    printf("Size: %i\n", i);
    MPI_Sendrecv_replace(buf, i, MPI_DOUBLE, size-rank-1, 1111, size-rank-1, 1111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Sendrecv_replace(buf, 323262400, MPI_DOUBLE, size-rank-1, 1111, size-rank-1, 1111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Finalize();
  return 0;
}
