#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int nf = argc - 1;
    if (nf < 1) {
        printf("you need to provide some floating-point numbers as arguments!\n");
    }

    double * fs = malloc(nf * sizeof(double));
    for (int i=0; i<nf; i++) {    
        fs[i] = atof(argv[i+1];
        printf("fs[%d]=%f\n",i,fs[i]);
    }

    free(fs);
    printf("all done\n");
    return 0;
}
