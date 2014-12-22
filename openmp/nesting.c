#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

void foo(void)
{
    printf("foo called by %d of %d\n", omp_get_thread_num(), omp_get_num_threads() );
#pragma omp parallel
    {
        printf("foo parallel reached by %d of %d\n", omp_get_thread_num(), omp_get_num_threads() );
    }
    return;
}

void bar(void)
{
    printf("bar called by %d of %d\n", omp_get_thread_num(), omp_get_num_threads() );
#pragma omp parallel
    {
        printf("bar parallel reached by %d of %d\n", omp_get_thread_num(), omp_get_num_threads() );
    }
    return;
}

int main(int argc, char* argv[])
{
#pragma omp parallel
    {
        printf("main parallel 1 from %d of %d\n", omp_get_thread_num(), omp_get_num_threads() );
        foo();
        bar();
    }
    fflush(stdout);
#pragma omp parallel
    {
        printf("main parallel 2 from %d of %d\n", omp_get_thread_num(), omp_get_num_threads() );
    }
    foo();
    bar();
    fflush(stdout);
    return 0;
}
