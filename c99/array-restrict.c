#include <stdio.h>
#include <stdlib.h>

int heavy_calcs(int n, float* restrict range1, float* restrict range2)
{
    if (n>1000) return 1;
    float tmpvalues[1000] = {0};
    {
        float * restrict ptv = tmpvalues;
        for (int i=0; i<n; i++) {
            ptv[i] = range1[i] + range2[i];
        }
    }
    return 0;
}

int main(int argc, char * argv[])
{
    int n = (argc>1) ? atoi(argv[1]) : 1000;
    float * r1 = (float*)malloc(n*sizeof(float));
    float * r2 = (float*)malloc(n*sizeof(float));
    int rc = heavy_calcs(n,r1,r2);
    free(r1);
    free(r2);
    return rc;
}
