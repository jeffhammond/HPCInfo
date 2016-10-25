#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void foo(double src)
{
    double dst;
    asm ("mov %1, %0"
       : "=r" (dst)
       : "r" (src));
    printf("%lf\n", dst);
}

void bar(double src)
{
    double dst;
    asm ("movnti %1, %0"
       : "=m" (dst)
       : "r" (src));
    asm ("sfence" ::: "memory");
    printf("%lf\n", dst);
}

int main(int argc, char* argv[])
{
    double src = 7777.;
    foo(src);
    bar(src);
    return 0;
}
