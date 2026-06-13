// this has to be above stdio.h for sched_getaffinity to be declared
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#include <sched.h>
#include <unistd.h>

#include <omp.h>

int main(void)
{

    {
        cpu_set_t mask;
        CPU_ZERO(&mask);

        int rc = sched_getaffinity(getpid(), sizeof(cpu_set_t), &mask);
        if (rc) abort();

        printf("CPU set=");
        for (int i=0; i<sizeof(cpu_set_t); i++) {
            int on = CPU_ISSET(i, &mask);
            if (on) printf(" %d", i);
        }
        printf("\n");
    }

    #pragma omp parallel
    {
        int me = omp_get_thread_num();

        cpu_set_t mask;
        CPU_ZERO(&mask);

        int rc = sched_getaffinity(0, sizeof(cpu_set_t), &mask);
        if (rc) abort();

        #pragma omp critical
        {
            printf("%d: CPU set=", me);
            for (int i=0; i<sizeof(cpu_set_t); i++) {
                int on = CPU_ISSET(i, &mask);
                if (on) printf(" %d", i);
            }
            printf("\n");
        }
    }

    return 0;
}
