#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(void) {

    int a = 0, b = 0;
    #pragma omp parallel
    {
        int me = omp_get_thread_num();
        if (me==0) a=1;
        if (me==1) b=1;
        #pragma omp flush (a,b)
    }
    printf("%d,%d\n",a,b);
    return 0;
}
/***************************************************************/
