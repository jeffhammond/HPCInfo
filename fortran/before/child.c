#include <stdio.h>

void before(void) __attribute__((constructor));

void before(void)
{
  printf ("BEFORE\n");
}
