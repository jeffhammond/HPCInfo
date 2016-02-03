/* associated with http://stackoverflow.com/q/35175957/2189128 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    int iter = (argc>1) ? atoi(argv[1]) : 50000;
    int r=0, c=0, a=0;

    printf("OpenMP threads = %d\n", omp_get_max_threads() );

    #pragma omp parallel reduction(+:r) shared(c,a)
    {
        #pragma omp for
        for (int i = 0; i < iter; i++ ) {
            r++;
            #pragma omp critical
            c++;
            #pragma omp atomic
            a++;
        }
    }
    printf("reduce      = %d\n"
           "critical    = %d\n"
           "atomic      = %d\n", r, c, a);
    return 0;
}
