#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/***************************************************************/
#include <omp.h>
void ** ompx_calloc(size_t bytes)
{
  int np = omp_get_max_threads();
  void ** ptrs = malloc(np*sizeof(void*));
  #pragma omp parallel shared(ptrs)
  {
    int me = omp_get_thread_num();
    ptrs[me] = malloc(bytes);
    memset(ptrs[me],0,bytes);
  }
  return ptrs;
}
void ompx_free(void ** ptrs)
{
  #pragma omp parallel shared(ptrs)
  {
    int me = omp_get_thread_num();
    free(ptrs[me]);
  }
  free(ptrs);
}
int main(int argc, char* argv[]) {
  int n = (argc>1) ? atoi(argv[1]) : 1<<20;
  int np = omp_get_max_threads();
  if (np<2) exit(1);
  int ** A = ompx_calloc(n*sizeof(int));
  #pragma omp parallel shared(A)
  {
     /* threaded computation */
  }
  ompx_free(A);
  return 0;
}

/***************************************************************/
