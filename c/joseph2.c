#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

typedef struct {
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
    int MPI_internal[5];
} MPI_Status;

int main(void)
{
  MPI_Status * s = malloc(sizeof(MPI_Status));
  s->MPI_SOURCE = 1;
  s->MPI_TAG    = 2;
  s->MPI_ERROR  = 3;
  s->MPI_internal[0] = 4;
  s->MPI_internal[1] = 5;
  s->MPI_internal[2] = 6;
  s->MPI_internal[3] = 7;
  s->MPI_internal[4] = 8;
  int64_t count = *(int64_t*)&(s->MPI_internal[1]);
  printf("%lld\n",count);
  printf("%lld\n",6LL*(1LL<<32)+5LL);
  return 0;
}
