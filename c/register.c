#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const int n = 1000;
    register double * restrict x = malloc(n * sizeof(double));

    for (int i=0; i<n; i++) {
        register int j = i;
        //const int j = i;
        //register double y = x[j];
        const double y = x[j];
        printf("&y = %p\n", (void*)&y);
        printf("&x[j] = %p\n", (void*)&x[j]);
        //printf("&p = %p\n", &j);
        //printf("&x[i] = %p\n", (void*)&x[i]);
    }

    return 0;
}
