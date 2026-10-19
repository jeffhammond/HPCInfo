#include <math.h>

int main(void)
{
    volatile double x = 1.0;
    return (int)(fmaximum(x, 2.0) + fminimum(x, 2.0) +
                 fmaximum_num(x, 2.0) + fminimum_num(x, 2.0)) > 100;
}
