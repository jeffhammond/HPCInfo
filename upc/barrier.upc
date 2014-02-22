#include <stdio.h>
#include <stdlib.h>
#ifdef RELAXED
#include <upc_relaxed.h>
#else
#include <upc.h>
#endif

int main(int argc, char* argv[])
{
    printf("Hello world: I am thread %d.\n", MYTHREAD);
    upc_barrier;
    return 0;
}

