#include <stdio.h>
#include <complex.h>
#include <stdatomic.h>

_Atomic long double _Complex x;

int main(void)
{
    x = 0.0 + 0.0*I;
    for (int i=0; i<1000; i++) {
        x += 1.0;
    }

    printf("(%Lf,%Lf)\n",creall(x), cimagl(x));

    return 0;
}
