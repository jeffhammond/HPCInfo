#include <stdio.h>
#include "pgif90.h"

typedef struct {
    int * m;
    void * wtf; 
    // this is 16 because d is a F90_Desc_la where
    // F90_DescDim_la is instantiated with MAXDIMS=1,
    // because the compiler is optimizing away unused
    // metadata.
    long long d[16];
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

void fa_(ca_c * pa)
{
    printf("pa=%zu &(pa->a)=%zu &(pa->i)=%zu pa->i=%d pa->a->m=%zu pa->a->i=%d\n", 
            pa, &(pa->a), &(pa->i), pa->i, pa->a.m, pa->a.i);
    printf("pa->a.wtf = %p\n", pa->a.wtf);
    for (int i=0; i<16; i++) {
        printf("pa->a.d[%d] = %zu\n", i, pa->a.d[i]);
    }
}

void fb(cb_c * pb)
{
    printf("pb=%zu &(pb->b)=%zu &(pb->i)=%zu pb->i=%d pb->b->m=%zu pb->b->i=%d\n", 
            pb, &(pb->b), &(pb->i), pb->i, pb->b.m, pb->b.i);
}
