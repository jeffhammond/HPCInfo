#include <stdio.h>
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
  MPI_Status s = {1,2,3,{4,5,6,7,8}};
  int64_t count = *(int64_t*)&(s.MPI_internal[1]);
  printf("%lld\n",count);
  printf("%lld\n",6LL*(1LL<<32)+5LL);
  return 0;
}
