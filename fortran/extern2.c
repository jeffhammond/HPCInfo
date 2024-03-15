#include <stdio.h>
#include <stdint.h>

void * evp;

void p(void)
{
    printf("evp=%p &evp=%p &evp=%zu\n",evp,&evp,(intptr_t)&evp);
}
