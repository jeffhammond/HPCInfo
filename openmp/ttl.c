// icx -fiopenmp -fopenmp-targets=spir64 slm.c && ./a.out
#include <stdio.h>
#include <omp.h>
int main(void)
{
    int mxtd;
    #pragma omp target teams reduction(max:mxtd)
    {
        mxtd = omp_get_max_threads();
    }
    printf("omp_get_max_threads = %d\n",mxtd);
    printf("omp_get_teams_thread_limit = %d\n",omp_get_teams_thread_limit());
    return 0;
}
