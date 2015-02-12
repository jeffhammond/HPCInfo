#include <stdio.h>
#include <stdlib.h>

int * foo(void)
{
    int a = 7;
    return &a;
}

int main(int argc, char* argv[])
{
    int * a = foo();
    int b = *a;
    printf("%d\n", b);
    return 0;
}
