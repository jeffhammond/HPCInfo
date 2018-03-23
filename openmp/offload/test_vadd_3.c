#include <stdio.h>
#include <stdlib.h>

#ifdef __cilk
#include <cilk/cilk.h>
#endif

void vadd4(int n, float * a, float * b, float * c)
{
    #if defined(__INTEL_COMPILER) && defined(__INTEL_OFFLOAD)
    #pragma offload target(gfx) in(a,b : length(n)) out(c : length(n)) //pin(a, b, c : length(n))
    #else
    #error No Cilk offload support!
    #endif
    _Cilk_for(int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

int main(int argc, char * argv[])
{
    int n = (argc > 1 ) ? atoi(argv[1]) : 1000;
    float * x  = calloc(n,sizeof(float));
    float * y  = calloc(n,sizeof(float));
    float * z4 = calloc(n,sizeof(float));

    for (int i=0; i<n; i++) {
        x[i] = y[i] = (float)i;
    }

    vadd4(n,x,y,z4);
    double junk = 0.0;
    for (int i=0; i<n; i++) {
        junk += z4[i];
    }
    printf("junk=%lf\n", junk);

    free(z4);
    free(y);
    free(x);

    return 0;
}
