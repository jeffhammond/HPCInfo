#include <stdio.h>
#include <complex.h>
//#include <stdatomic.h>

int main(void)
{
    _Atomic long double _Complex x = 0.0 + 0.0*I;
    for (int i=0; i<1000; i++) {
        x += 1.0;
    }
    printf("(%Lf,%Lf)\n",creall(x), cimagl(x));
    printf("%zu\n",sizeof(x));
    return 0;
}
