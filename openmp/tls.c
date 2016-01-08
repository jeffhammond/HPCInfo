#if (__STDC_VERSION__ >= 201112L) && !defined(__STDC_NO_THREADS__)

#include <stdio.h>
#include <stdlib.h>
//#include <threads.h>
#include <omp.h>

int p;
#pragma omp threadprivate(p)
int s;

int main(int argc, char **argv)
{
    #pragma omp parallel
    {
        printf("%d: %p %p\n", omp_get_thread_num(), &p, &s );
    }
    return 0;
}

#endif
