// icx -fiopenmp -fopenmp-targets=spir64 spmd.c && ./a.out

#include <stdio.h>
#include <omp.h>

int main(void)
{
#if JEFF
#warning JEFF
    #pragma omp target teams parallel
    {
        int team = omp_get_team_num();
        int thrd = omp_get_thread_num();
        printf("team %6d thread %6d\n", team, thrd);
    }
#else
    #pragma omp target teams //parallel
    {
        int team = omp_get_team_num();
        #pragma omp parallel
        {
            int thrd = omp_get_thread_num();
            printf("team %6d thread %6d\n", team, thrd);
        }
    }
#endif
    return 0;
}
