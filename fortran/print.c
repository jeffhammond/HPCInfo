#include <stdio.h>

void p(int * i)
{
    printf("sint: %d\n",*i);
    printf("uint: %u\n",(unsigned)*i);
    printf("hex:  %x\n",*i);
    printf("addr:  %p\n",i);
}
