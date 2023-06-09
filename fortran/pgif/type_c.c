#include <stdio.h>
#include "pgif90.h"

typedef struct {
    int * m;
    long long d[17];
    int i;
} ta_c;

typedef struct {
    int m[100];
    int i;
} tb_c;

typedef struct {
    ta_c a;
    int i;
} ca_c;

typedef struct {
    tb_c b;
    int i;
} cb_c;

void fa(ca_c * pa)
{
    printf("pa=%zu &(pa->a)=%zu &(pa->i)=%zu pa->i=%d pa->a->m=%zu pa->a->i=%d\n", 
            pa, &(pa->a), &(pa->i), pa->i, pa->a.m, pa->a.i);
}

void fb(cb_c * pb)
{
    printf("pb=%zu &(pb->b)=%zu &(pb->i)=%zu pb->i=%d pb->b->m=%zu pb->b->i=%d\n", 
            pb, &(pb->b), &(pb->i), pb->i, pb->b.m, pb->b.i);
}
