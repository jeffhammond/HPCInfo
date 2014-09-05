#include <stdio.h>
#include <quadmath.h>

int main(void)
{
  char tmp[128];
  __float128 x = sqrtq(2.0Q);

  quadmath_snprintf(tmp, sizeof tmp, "%.45Qf",x);
  printf("%s\n",tmp);
  printf("%zu \n",sizeof(__float128));

  return 0;
}
