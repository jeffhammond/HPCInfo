#include <stdio.h>
#include <stdlib.h>

/***************************************************************/
#include <omp.h>
int main(void) {
    /* allocate from the heap with global visibility */
    int * A = malloc(omp_get_max_threads()*sizeof(int));
    #pragma omp parallel shared(A)
    {
        if (omp_get_num_threads()<2) exit(1);
        int B = 37;
        /* store local data B at PE 0 into A at PE 1 */
        if (omp_get_thread_num()==0) A[1] = B;
        /* global synchronization of execution and data */
        #pragma omp flush
        #pragma omp barrier
        /* observe the result of the store */
        if (omp_get_thread_num()==1) printf("A@1=%d\n",A[1]);
    }
    free(A);
    return 0;
}
/***************************************************************/
