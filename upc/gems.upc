#include <stdio.h>
#include <stdlib.h>
/***************************************************************/
#include <upc.h>
int main(void) {
    if (THREADS<2) exit(1);
    /* allocate from the global heap */
    shared int * A = upc_all_alloc(THREADS,sizeof(int));
    int B = 37;
    /* store contents of local data B at PE 0 into A at PE 1 */
    if (MYTHREAD==0) A[1] = B;
    /* global synchronization of execution and data */
    upc_barrier;
    /* observe the result of the store */
    if (MYTHREAD==1) printf("A@1=%d\n",A[1]);
    upc_all_free(A);
    return 0;
}
/***************************************************************/
