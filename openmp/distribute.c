#include <stdio.h>
#include <omp.h>

/* This program does not work, because I am not using distribute properly. */

int main(int argc, char* argv[])
{
#ifdef DISTRIBUTE
#pragma omp distribute parallel for
#else
#pragma omp parallel for
#endif
    for (int i=0; i<100; i++) {
        printf("tid=%d\n", omp_get_thread_num());
    }
    return 0;
}
