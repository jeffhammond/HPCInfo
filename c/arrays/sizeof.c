// https://nullprogram.com/blog/2019/10/27/

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[])
{
    int n = (argc>1 ? atoi(argv[1]) : 100);

    float (*z)[n] = malloc(sizeof(*z) * n);
    printf("sizeof(*z) = %zu\n", sizeof(*z) );

    if (z) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                z[j][i] = (i == j);
            }
        }
    }
    return z[argc][argc];    
}
