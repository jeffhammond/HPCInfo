#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sched.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if 1
void foo(void)
{
    #pragma omp target
    #pragma omp parallel for
    for (int i=0; i<1; i++)
    sched_yield();
}
#endif

#if 0
void foo(void)
{
    #pragma omp target
    #pragma omp parallel for
    for (int i=0; i<1; i++)
    printf("Bob W is great.\n");
}
#endif

#if 0
void foo(void)
{
    #pragma omp target
    #pragma omp parallel for
    for (int i=0; i<1; i++)
    puts("Rolf R is great\n");
}
#endif

int main(int argc, char * argv[])
{
    foo();

    printf("Success\n");

    return 0;
}
