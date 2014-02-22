#include <stdio.h>
#include <stdlib.h>
#ifdef RELAXED
#include <upc_relaxed.h>
#else
#include <upc.h>
#endif

int main(int argc, char* argv[])
{
    static shared int array[THREADS];
    array[MYTHREAD] = MYTHREAD;

    upc_barrier;

    if (THREADS<100)
    {
        int i;
        for (i=0;i<THREADS;i++)
            printf("%d: array[%d] = %d is owned by thread %ld \n", 
                   MYTHREAD, i, array[i], (long) upc_threadof(&array[i]) );
    }
    else if (MYTHREAD==0)
    {
        printf("I'm not printing %ld lines to stdout. \n", 
               (long)THREADS*THREADS);
    }

    return 0;
}

