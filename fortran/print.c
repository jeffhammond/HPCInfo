#include <stdio.h>

void p(int * i)
{
    printf("sint: %d\n",*i);
    printf("uint: %u\n",(unsigned)*i);
    printf("ptr:  %p\n",i);
}
