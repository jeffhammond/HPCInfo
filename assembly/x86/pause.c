#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static inline void cpu_relax(void)
{
    asm ("pause" ::: "memory");
}

int main(int argc, char* argv[])
{
    const int n = (argc>1) ? atoi(argv[1]) : 1000;
    double t0 = omp_get_wtime();
    for (int i=0; i<n; i++) {
        cpu_relax();
    }
    double t1 = omp_get_wtime();
    printf("dt = %lf\n", t1-t0);
    return 0;
}
