#include <stdio.h>
#include <omp.h>

int main(void)
{
    int x = 0;
    #pragma omp parallel
    #pragma omp master
    {
        #pragma omp task
        {
            #pragma omp parallel for
            for (int i=0; i<100000; i++) {
                #pragma omp atomic update
                x++;
            }
        }
        #pragma omp task
        {
            #pragma omp target map(tofrom:x)
            {
                #pragma omp parallel for
                for (int i=0; i<100000; i++) {
                    #pragma omp atomic update
                    x++;
                }
            }
        }
        #pragma omp taskwait
    }
    printf("x=%d\n", x);
    return 0;
}
