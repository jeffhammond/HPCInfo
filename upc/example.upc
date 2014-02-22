#include <stdio.h>

#include <upc_relaxed.h>

int main(void)
{
    int a=0, b=0;
    static shared int x=0;

    if (MYTHREAD==0)
        x=1;
    if (MYTHREAD==1)
        x=2;

    upc_barrier;

    if (MYTHREAD==0)
    {
        a=x;
        printf("%d: a=%d \n", MYTHREAD, a);
    }

    if (MYTHREAD==1)
    {
        b=x;
        printf("%d: b=%d \n", MYTHREAD, b);
    }


    return 0;
} 
