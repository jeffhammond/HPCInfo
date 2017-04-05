#include <cpuid.h>
#include <stdio.h>
#include <stdint.h>

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
    a = 1024;
    for (uint32_t i=1; i<=32; i++) {
        printf("%d", (a >> i) & 0x00000001 );
        //printf("\n");
        //printf("i=%d i mod 4=%d\n", i, i%4);
        if ((i%4)==0) printf("|");
    }
    printf("\n");
    printf("junk=%zu\n", junk);
    return 0;
}
