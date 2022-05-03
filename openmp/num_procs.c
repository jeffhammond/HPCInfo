#include <stdio.h>
#include <omp.h>

int main(void)
{
    printf("L0: %d\n", omp_get_num_procs() );
    #pragma omp parallel num_threads(8)
    {
        int me = omp_get_thread_num();
        #pragma omp critical
        {
            printf("%d: L1: %d\n", me, omp_get_num_procs() );
        }
        #pragma omp parallel num_threads(4) 
        {
            #pragma omp critical
            {
                printf("%d: L2: %d\n", me, omp_get_num_procs() );
            }
            #pragma omp parallel num_threads(2) 
            {
                #pragma omp critical
                {
                    printf("%d: L3: %d\n", me, omp_get_num_procs() );
                }
            }
        }
    }
    return 0;
}
