#include <cpuid.h>
#include <stdio.h>
#include <stdint.h>

#if 1

int main (void)
{
    int32_t e[4];
    char n[4] = {'A','B','C','D'};
    __cpuid (0 /* vendor string */, e[0], e[1], e[2], e[3]);
    printf("------------------------------------------------------\n");
    printf("(hi-to-lo)   |1098|7654|3210|9876|5432|1098|7654|3210|\n");
    printf("------------------------------------------------------\n");
    for (int j=0; j<4; j++) {
        printf("E%cX=%8x=|",n[j],e[j]);
        for (int32_t i=31; i>=0; i--) {
            printf("%d", e[j] & (1<<i) ? 1 : 0);
            if ((i%4)==0) printf("|");
        }
        printf("\n");
    }
    printf("------------------------------------------------------\n");
    return 0;
}

#else

#include <omp.h>

int main (void)
{
    double dt = omp_get_wtime();
    int n = 100000000;
    int32_t a, b, c, d;
    int64_t junk = 0;
    double t0 = omp_get_wtime();
    for (int i=0; i<n; i++) {
        __cpuid (0 /* vendor string */, a, b, c, d);
        junk += (a+b+c+d);
    }
    double t1 = omp_get_wtime();
    dt = t1-t0;
    printf("cpuid dt=%lf s dt/n=%lf ns\n", dt, 1.e9*(dt/n));
    printf("EAX: %x\nEBX: %x\nECX: %x\nEDX: %x\n", a, b, c, d);
    printf("junk=%zu\n", junk);
    return 0;
}

#endif
